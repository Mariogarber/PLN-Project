import torch
from typing import List, Dict, Optional
from datasets import Dataset
from toxicity_logits_processor import ToxicityLogitsProcessor


class DetoxInference:
    """
    Full inference pipeline for detoxification using:
    - mT5 (with or without LoRA)
    - Multilingual lexicons
    - Toxicity-aware LogitsProcessor

    This class produces detoxified outputs and returns them
    in a format ready for the Evaluator.
    """

    def __init__(
        self,
        model,
        tokenizer,
        lexicons: Dict[str, List[str]],
        device: Optional[str] = None,
        penalty: float = 12.0,
        block_extra_ids: bool = True,
    ):
        """
        Parameters
        ----------
        model : PreTrainedModel
            Your trained mT5 (optionally with LoRA merged).
        tokenizer : PreTrainedTokenizer
            Matching tokenizer.
        lexicons : Dict[str, List[str]]
            Dict mapping languages to toxic lexicons.
        device : str
            "cuda" or "cpu". If None, auto-detect.
        penalty : float
            Logits penalty for toxic tokens.
        block_extra_ids : bool
            Whether to block <extra_id_X> tokens (recommended: True)
        """

        self.model = model
        self.tokenizer = tokenizer
        self.lexicons = lexicons

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.penalty = penalty
        self.block_extra_ids = block_extra_ids

    # ----------------------------------------------------------------------
    def _run_batch(
        self,
        toxic_list: List[str],
        language: str,
        max_length: int,
        num_beams: int,
        batch_size: int,
    ):
        """Runs detox inference for a batch in a single language."""

        processor = ToxicityLogitsProcessor(
            tokenizer=self.tokenizer,
            toxic_lexicons=self.lexicons,
            language=language,
            penalty=self.penalty,
            block_extra_ids=self.block_extra_ids,
        )

        encoded = self.tokenizer(
            toxic_list,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                num_beams=num_beams,
                max_length=max_length,
                logits_processor=[processor],
            )

        decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return decoded

    # ----------------------------------------------------------------------
    def run(
        self,
        dataset: Dataset,
        toxic_col: str = "toxic_sentence",
        gold_col: str = "neutral_sentence",
        lang_col: str = "language",
        batch_size: int = 16,
        max_length: int = 128,
        num_beams: int = 4,
    ):
        """
        Full multilingual detoxified inference.

        Parameters
        ----------
        dataset : Dataset
            Tokenized or untokenized HF dataset.
        toxic_col : str
            Column holding toxic input sentences.
        gold_col : str
            Column with neutral gold sentences (optional).
        lang_col : str
            Column specifying language code.
        batch_size : int
            Batch size for inference.
        max_length : int
            Maximum generated length.
        num_beams : int
            Beam search parameter.

        Returns
        -------
        List[Dict]
            Items with keys: language, toxic, gold, pred
        """

        results = []

        # Group by language for more efficient decoding
        languages = set(dataset[lang_col])
        print(f"Running detox inference for languages: {languages}\n")

        for lang in languages:
            if lang not in self.lexicons:
                print(f"WARNING: No lexicon found for '{lang}'. Skipping.")
                continue

            subset = dataset.filter(lambda x: x[lang_col] == lang)

            print(f"→ Language '{lang}' with {len(subset)} samples")

            # Process in mini-batches
            for start in range(0, len(subset), batch_size):
                batch = subset[start:start + batch_size]

                toxic_list = batch[toxic_col]
                gold_list = batch[gold_col]

                preds = self._run_batch(
                    toxic_list=toxic_list,
                    language=lang,
                    max_length=max_length,
                    num_beams=num_beams,
                    batch_size=batch_size,
                )

                for t, g, p in zip(toxic_list, gold_list, preds):
                    results.append({
                        "language": lang,
                        "toxic": t,
                        "gold": g,
                        "pred": p.strip(),
                    })

        return results
