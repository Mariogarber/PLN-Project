import re
from typing import Dict, Optional, Callable

from datasets import Dataset, DatasetDict
from transformers import T5TokenizerFast


class DataTokenizer:
    """
    A clean, modular tokenizer class for mT5 detoxification.

    Responsibilities:
    - Load tokenizer
    - Clean text (optional)
    - Tokenize HF datasets (train/val/test)
    - Tokenize curriculum datasets (easy/medium/hard/full)
    - Output: input_ids, attention_mask, labels
    """

    def __init__(
        self,
        tokenizer_name: str = "google/mt5-base",
        prefix: str = "detoxify_keep_meaning: ",
        max_input_length: int = 128,
        max_target_length: int = 128,
        clean_fn: Optional[Callable[[str], str]] = None,
        logger=None,
    ):
        self.tokenizer = T5TokenizerFast.from_pretrained(tokenizer_name)
        self.prefix = prefix
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.clean_fn = clean_fn
        self.logger = logger

        if self.logger:
            self.logger.info(f"Loaded tokenizer {tokenizer_name}")

    # ----------------------------------------------------
    #                 TEXT CLEANING
    # ----------------------------------------------------
    def clean_text(self, text: str) -> str:
        """Allow custom cleaning function or default behavior."""
        if self.clean_fn:
            return self.clean_fn(text)

        # Default cleaning: remove double spaces, trim
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("’", "'")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ----------------------------------------------------
    #                 TOKENIZATION LOGIC
    # ----------------------------------------------------
    def _tokenize_row(self, batch):
        """
        Tokenizes one batch of HF Dataset rows.
        Expects columns:
            - toxic_sentence
            - neutral_sentence
        """
        inputs = [
            self.prefix + self.clean_text(t) for t in batch["toxic_sentence"]
        ]
        targets = [self.clean_text(t) for t in batch["neutral_sentence"]]

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
        )

        labels = self.tokenizer(
            targets,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
        )["input_ids"]

        # Replace pad token ids with -100 for T5 loss masking
        labels = [
            [(token if token != self.tokenizer.pad_token_id else -100) for token in seq]
            for seq in labels
        ]

        model_inputs["labels"] = labels
        return model_inputs

    # ----------------------------------------------------
    #           TOKENIZE SIMPLE DATASET
    # ----------------------------------------------------
    def tokenize_dataset(
        self,
        dataset: Dataset,
        num_proc: int = 4,
        remove_columns: bool = True,
    ) -> Dataset:
        """
        Tokenizes a HF Dataset containing textual detox samples.
        """

        if self.logger:
            self.logger.info(f"Tokenizing dataset with {len(dataset)} samples...")

        cols_to_remove = (
            dataset.column_names if remove_columns else []
        )

        tokenized = dataset.map(
            self._tokenize_row,
            batched=True,
            num_proc=num_proc,
            remove_columns=cols_to_remove,
            desc="Tokenizing dataset",
        )

        return tokenized

    # ----------------------------------------------------
    #           TOKENIZE STANDARD SPLITS
    # ----------------------------------------------------
    def tokenize_splits(
        self,
        splits: Dict[str, Dataset],
        num_proc: int = 4,
    ) -> Dict[str, Dataset]:
        """
        Tokenize a dict of datasets: {"train": ..., "val": ..., "test": ...}
        """

        if self.logger:
            self.logger.info("Tokenizing train/val/test splits...")

        tokenized_splits = {}
        for key, ds in splits.items():
            if self.logger:
                self.logger.info(f" → Tokenizing split `{key}` ({len(ds)} samples)")
            tokenized_splits[key] = self.tokenize_dataset(ds, num_proc=num_proc)

        return tokenized_splits

    # ----------------------------------------------------
    #           TOKENIZE CURRICULUM DATASETS
    # ----------------------------------------------------
    def tokenize_curriculum(
        self,
        curriculum: Dict[str, Dataset],
        num_proc: int = 4,
    ) -> Dict[str, Dataset]:
        """
        Tokenizes curriculum datasets:
            - easy
            - medium
            - hard
            - full

        Returns a dict with the same keys and tokenized datasets.
        """

        if self.logger:
            self.logger.info("Tokenizing curriculum datasets...")

        tokenized = {}
        for key, ds in curriculum.items():
            if self.logger:
                self.logger.info(f" → Tokenizing `{key}` ({len(ds)} samples)")
            tokenized[key] = self.tokenize_dataset(ds, num_proc=num_proc)

        return tokenized
