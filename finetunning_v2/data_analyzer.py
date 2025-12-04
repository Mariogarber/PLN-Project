import logging
from typing import Optional, Union, Dict, List, Tuple
import io
import base64

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase
from bert_score import score as bertscore
from sentence_transformers import SentenceTransformer
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

class DatasetAnalyzer:
    """
    DatasetAnalyzer: non-destructive analysis tool for detoxification datasets.

    - Works with HF Dataset / DatasetDict / pandas DataFrame
    - Multilingual-aware (expects optional 'language' column)
    - Compatible with DataManager outputs (raw_dataset, splits)
    - Focused on:
        * basic stats
        * length stats
        * overlap stats ('equal_percentage')
        * difficulty buckets (easy/medium/hard)
        * noise detection (empty, duplicates, identical pairs)
    """

    def __init__(
        self,
        data: Union[Dataset, DatasetDict, pd.DataFrame],
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        split_name: Optional[str] = None,
        log_level: int = logging.INFO,
    ):
        """
        Parameters
        ----------
        data:
            HF Dataset / DatasetDict / pandas DataFrame.
            If DatasetDict is provided, `split_name` must be specified (e.g. 'train').

        tokenizer:
            Optional tokenizer. Used for token-level stats if desired.

        split_name:
            Name of the split to select if `data` is a DatasetDict.

        log_level:
            Logging level (e.g. logging.INFO, logging.DEBUG).
        """
        # ----------------------
        # LOGGING
        # ----------------------
        self.logger = logging.getLogger("DatasetAnalyzer")
        self.logger.setLevel(log_level)

        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        if not self.logger.handlers:
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # ----------------------
        # STORE TOKENIZER
        # ----------------------
        self.tokenizer = tokenizer

        # ----------------------
        # CONVERT INPUT TO DATAFRAME
        # ----------------------
        self.df = self._to_dataframe(data, split_name)

        if self.df.empty:
            self.logger.warning("Initialized DatasetAnalyzer with an EMPTY dataset.")

        self.logger.info(
            f"DatasetAnalyzer initialized with {len(self.df)} samples "
            f"and columns: {list(self.df.columns)}"
        )

    # ============================================================
    # =============== INTERNAL CONVERSION UTILITIES ==============
    # ============================================================

    def _to_dataframe(
        self,
        data: Union[Dataset, DatasetDict, pd.DataFrame],
        split_name: Optional[str],
    ) -> pd.DataFrame:
        """
        Convert any supported input (Dataset, DatasetDict, DataFrame) into a pandas DataFrame.
        Does NOT modify the original object.
        """
        # If it's already a DataFrame, copy it to avoid side-effects
        if isinstance(data, pd.DataFrame):
            self.logger.debug("Input data is already a pandas DataFrame.")
            return data.copy()

        # If it's a DatasetDict, require a split name
        if isinstance(data, DatasetDict):
            if split_name is None:
                raise ValueError(
                    "data is a DatasetDict; 'split_name' must be provided "
                    f"(available splits: {list(data.keys())})"
                )
            if split_name not in data:
                raise ValueError(
                    f"Split '{split_name}' not found in DatasetDict. "
                    f"Available: {list(data.keys())}"
                )
            self.logger.info(f"Using split '{split_name}' from DatasetDict.")
            ds = data[split_name]
            return self._dataset_to_df(ds)

        # If it's a single Dataset
        if isinstance(data, Dataset):
            self.logger.info("Converting HF Dataset to pandas DataFrame.")
            return self._dataset_to_df(data)

        raise TypeError(
            f"Unsupported data type: {type(data)}. "
            "Expected Dataset, DatasetDict, or pandas DataFrame."
        )

    @staticmethod
    def _dataset_to_df(dataset: Dataset) -> pd.DataFrame:
        """
        Safe conversion from HF Dataset to pandas DataFrame.
        """
        # as_pandas() returns a DataFrame-like object; ensure we get a plain DataFrame
        df = dataset.to_pandas()
        return df.reset_index(drop=True)

    # ============================================================
    # ====================== BASIC STATS =========================
    # ============================================================

    def basic_stats(self) -> Dict[str, Union[int, Dict]]:
        """
        Basic high-level statistics:
        - total samples
        - available columns
        - language distribution (if 'language' column exists)
        """
        stats: Dict[str, Union[int, Dict]] = {
            "num_samples": len(self.df),
            "columns": list(self.df.columns),
        }

        if "language" in self.df.columns:
            lang_counts = self.df["language"].value_counts().to_dict()
            stats["language_counts"] = lang_counts
        else:
            stats["language_counts"] = {}

        self.logger.info(f"Total samples: {stats['num_samples']}")
        if stats["language_counts"]:
            self.logger.info(f"Samples per language: {stats['language_counts']}")
        else:
            self.logger.info("No 'language' column found. Skipping language stats.")

        return stats

    # ============================================================
    # ==================== LENGTH STATS ==========================
    # ============================================================

    def length_stats(
        self,
        by_language: bool = True,
        use_tokens: bool = False,
    ) -> Dict[str, Dict]:
        """
        Length statistics for toxic and neutral sentences.

        Parameters
        ----------
        by_language:
            If True, compute stats per language (requires 'language' column).

        use_tokens:
            If True and tokenizer is provided, compute token-level lengths.
            Otherwise, use whitespace-separated word counts.

        Returns
        -------
        Dict with global and optional per-language stats.
        """
        if "toxic_sentence" not in self.df.columns or "neutral_sentence" not in self.df.columns:
            raise ValueError("Expected columns 'toxic_sentence' and 'neutral_sentence' in dataset.")

        self.logger.info(
            f"Computing length stats "
            f"({'tokens' if use_tokens and self.tokenizer else 'words'})..."
        )

        if use_tokens and self.tokenizer:
            toxic_lens = self.df["toxic_sentence"].astype(str).apply(
                lambda x: len(self.tokenizer.encode(x, add_special_tokens=False))
            )
            neutral_lens = self.df["neutral_sentence"].astype(str).apply(
                lambda x: len(self.tokenizer.encode(x, add_special_tokens=False))
            )
        else:
            toxic_lens = self.df["toxic_sentence"].astype(str).apply(
                lambda x: len(x.split())
            )
            neutral_lens = self.df["neutral_sentence"].astype(str).apply(
                lambda x: len(x.split())
            )

        result: Dict[str, Dict] = {
            "global": {
                "toxic_mean": float(toxic_lens.mean()),
                "toxic_median": float(toxic_lens.median()),
                "toxic_std": float(toxic_lens.std(ddof=0)),
                "neutral_mean": float(neutral_lens.mean()),
                "neutral_median": float(neutral_lens.median()),
                "neutral_std": float(neutral_lens.std(ddof=0)),
            }
        }

        self.logger.info(
            f"[GLOBAL] toxic len mean={result['global']['toxic_mean']:.2f}, "
            f"neutral len mean={result['global']['neutral_mean']:.2f}"
        )

        if by_language and "language" in self.df.columns:
            lang_stats: Dict[str, Dict] = {}
            for lang, group in self.df.groupby("language"):
                if use_tokens and self.tokenizer:
                    t_l = group["toxic_sentence"].astype(str).apply(
                        lambda x: len(self.tokenizer.encode(x, add_special_tokens=False))
                    )
                    n_l = group["neutral_sentence"].astype(str).apply(
                        lambda x: len(self.tokenizer.encode(x, add_special_tokens=False))
                    )
                else:
                    t_l = group["toxic_sentence"].astype(str).apply(
                        lambda x: len(x.split())
                    )
                    n_l = group["neutral_sentence"].astype(str).apply(
                        lambda x: len(x.split())
                    )

                lang_stats[lang] = {
                    "toxic_mean": float(t_l.mean()),
                    "toxic_median": float(t_l.median()),
                    "neutral_mean": float(n_l.mean()),
                    "neutral_median": float(n_l.median()),
                    "num_samples": int(len(group)),
                }

                self.logger.info(
                    f"[{lang}] toxic mean={lang_stats[lang]['toxic_mean']:.2f}, "
                    f"neutral mean={lang_stats[lang]['neutral_mean']:.2f}, "
                    f"samples={lang_stats[lang]['num_samples']}"
                )

            result["by_language"] = lang_stats

        return result

    # ============================================================
    # ==================== OVERLAP STATS =========================
    # ============================================================

    def overlap_stats(
        self,
        by_language: bool = True,
        quantiles: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
    ) -> Dict[str, Dict]:
        """
        Stats for 'equal_percentage' (lexical overlap toxic→neutral).

        Requires:
        - column 'equal_percentage' in dataset.
        """
        if "equal_percentage" not in self.df.columns:
            raise ValueError(
                "Column 'equal_percentage' not found. "
                "You should compute it in DataManager before analyzing overlap."
            )

        overlap = self.df["equal_percentage"].astype(float)

        q_values = overlap.quantile(list(quantiles)).to_dict()

        global_stats: Dict[str, float] = {
            "mean": float(overlap.mean()),
            "std": float(overlap.std(ddof=0)),
            "min": float(overlap.min()),
            "max": float(overlap.max()),
        }
        global_stats.update({f"q{int(q*100)}": float(v) for q, v in q_values.items()})

        self.logger.info(
            f"[OVERLAP GLOBAL] mean={global_stats['mean']:.2f}, "
            f"std={global_stats['std']:.2f}, "
            f"min={global_stats['min']:.2f}, "
            f"max={global_stats['max']:.2f}"
        )

        result: Dict[str, Dict] = {"global": global_stats}

        if by_language and "language" in self.df.columns:
            lang_stats: Dict[str, Dict] = {}
            for lang, group in self.df.groupby("language"):
                ov = group["equal_percentage"].astype(float)
                if len(ov) == 0:
                    continue

                q_vals_lang = ov.quantile(list(quantiles)).to_dict()
                stats_lang: Dict[str, float] = {
                    "mean": float(ov.mean()),
                    "std": float(ov.std(ddof=0)),
                    "min": float(ov.min()),
                    "max": float(ov.max()),
                    "num_samples": int(len(ov)),
                }
                stats_lang.update(
                    {f"q{int(q*100)}": float(v) for q, v in q_vals_lang.items()}
                )

                lang_stats[lang] = stats_lang

                self.logger.info(
                    f"[OVERLAP {lang}] mean={stats_lang['mean']:.2f}, "
                    f"q50={stats_lang.get('q50', np.nan):.2f}, "
                    f"samples={stats_lang['num_samples']}"
                )

            result["by_language"] = lang_stats

        return result

    # ============================================================
    # ================== DIFFICULTY BUCKETS ======================
    # ============================================================

    def difficulty_buckets(
        self,
        low: float = 30.0,
        high: float = 70.0,
    ) -> Dict[str, int]:
        """
        Classify examples into difficulty buckets based on 'equal_percentage':

        - 'hard':   overlap < low
        - 'medium': low <= overlap < high
        - 'easy':   overlap >= high

        Returns
        -------
        Dict with counts per bucket: {'hard': ..., 'medium': ..., 'easy': ...}

        Note: This method uses an internal column 'difficulty' in the analyzer's
        DataFrame copy, but does NOT modify the original dataset object.
        """
        if "equal_percentage" not in self.df.columns:
            raise ValueError(
                "Column 'equal_percentage' not found. Required for difficulty buckets."
            )

        overlap = self.df["equal_percentage"].astype(float)

        conditions = [
            overlap < low,
            (overlap >= low) & (overlap < high),
            overlap >= high,
        ]
        choices = ["hard", "medium", "easy"]
        difficulty = np.select(conditions, choices, default="medium")

        # Internal only (copy within analyzer)
        self.df["difficulty"] = difficulty

        counts = pd.Series(difficulty).value_counts().to_dict()
        # Ensure all keys exist
        for k in ["hard", "medium", "easy"]:
            counts.setdefault(k, 0)

        self.logger.info(
            f"Difficulty buckets (low={low}, high={high}): "
            f"{counts}"
        )

        return counts

    # ============================================================
    # ====================== NOISE STATS =========================
    # ============================================================

    def noise_stats(self) -> Dict[str, int]:
        """
        Detect basic noise patterns:
        - empty toxic_sentence
        - empty neutral_sentence
        - toxic == neutral
        - duplicates (toxic, neutral) pairs

        Returns
        -------
        Dict with counts of each kind of noise.
        """
        required_cols = {"toxic_sentence", "neutral_sentence"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(
                f"Expected columns {required_cols} to compute noise stats."
            )

        toxic = self.df["toxic_sentence"].astype(str)
        neutral = self.df["neutral_sentence"].astype(str)

        # empty after stripping
        empty_toxic = (toxic.str.strip() == "").sum()
        empty_neutral = (neutral.str.strip() == "").sum()

        identical_pairs = (toxic == neutral).sum()

        # duplicates as exactly same (toxic, neutral) pair
        dup_pairs = self.df.duplicated(subset=["toxic_sentence", "neutral_sentence"]).sum()

        stats = {
            "empty_toxic": int(empty_toxic),
            "empty_neutral": int(empty_neutral),
            "identical_pairs": int(identical_pairs),
            "duplicate_pairs": int(dup_pairs),
        }

        self.logger.info(f"Noise stats: {stats}")

        return stats

    # ============================================================
    # ===================== BERTSCORE STATS =======================
    # ============================================================

    def bertscore_stats(
        self,
        model_type: str = "microsoft/deberta-xlarge-mnli",
        batch_size: int = 32,
        lang: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Compute BERTScore precision, recall, F1 for toxic → neutral pairs.

        Parameters
        ----------
        model_type:
            BERTScore model to use (DeBERTa models yield best performance).

        batch_size:
            Batching size for efficient GPU/CPU usage.

        lang:
            Optional language code. BERTScore supports language hints (e.g. 'en').

        Returns
        -------
        dict with mean P/R/F1
        """

        if "toxic_sentence" not in self.df or "neutral_sentence" not in self.df:
            raise ValueError("Dataset must contain columns 'toxic_sentence' and 'neutral_sentence'.")

        hyp = self.df["neutral_sentence"].astype(str).tolist()
        ref = self.df["toxic_sentence"].astype(str).tolist()

        self.logger.info(f"Computing BERTScore using model '{model_type}' ...")

        with torch.no_grad():
            P, R, F1 = bertscore(
                cands=hyp,
                refs=ref,
                lang=lang,
                model_type=model_type,
                batch_size=batch_size,
                rescale_with_baseline=False,
            )

        stats = {
            "precision": float(P.mean().item()),
            "recall": float(R.mean().item()),
            "f1": float(F1.mean().item()),
        }

        self.logger.info(
            f"BERTScore — P={stats['precision']:.4f}, "
            f"R={stats['recall']:.4f}, F1={stats['f1']:.4f}"
        )

        return stats

    # ============================================================
    # ================== EMBEDDING SIMILARITY ====================
    # ============================================================

    def sentence_embedding_similarity(
        self,
        model_name: str = "sentence-transformers/LaBSE",
        batch_size: int = 32,
        device: str = None,
    ) -> Dict[str, float]:
        """
        Compute cosine similarity between toxic and neutral sentences
        using sentence embeddings (LaBSE recommended for multilingual).

        Parameters
        ----------
        model_name:
            SentenceTransformer model, e.g.:
            - "sentence-transformers/LaBSE" (multilingual)
            - "all-mpnet-base-v2" (English only)
        
        batch_size:
            Embedding batch size.

        device:
            Force device (cpu/cuda), else detect automatically.

        Returns
        -------
        dict with mean, median, std cosine similarity
        """

        if "toxic_sentence" not in self.df.columns:
            raise ValueError("Column 'toxic_sentence' required.")
        if "neutral_sentence" not in self.df.columns:
            raise ValueError("Column 'neutral_sentence' required.")

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info(f"Loading embedding model '{model_name}' on {device} ...")

        model = SentenceTransformer(model_name).to(device)

        toxic = self.df["toxic_sentence"].astype(str).tolist()
        neutral = self.df["neutral_sentence"].astype(str).tolist()

        self.logger.info("Computing embeddings...")
        emb_toxic = model.encode(toxic, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
        emb_neutral = model.encode(neutral, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)

        self.logger.info("Computing cosine similarities...")
        cos_sim = torch.nn.functional.cosine_similarity(emb_toxic, emb_neutral)

        stats = {
            "mean": float(cos_sim.mean().item()),
            "median": float(cos_sim.median().item()),
            "std": float(cos_sim.std().item()),
        }

        self.logger.info(
            f"Embedding similarity — mean={stats['mean']:.4f}, "
            f"median={stats['median']:.4f}, std={stats['std']:.4f}"
        )

        return stats

    # ============================================================
    # ================= SEMANTIC OUTLIERS =========================
    # ============================================================

    def semantic_outliers(
        self,
        model_name: str = "sentence-transformers/LaBSE",
        batch_size: int = 32,
        threshold: float = 0.40,
    ) -> pd.DataFrame:
        """
        Identify semantic outliers: pairs with very low semantic similarity.

        Parameters
        ----------
        threshold:
            Similarity below this is flagged as a semantic mismatch.
            Typical values: 0.30 – 0.45 depending on dataset.

        Returns
        -------
        DataFrame with suspicious examples.
        """

        sims = self.sentence_embedding_similarity(
            model_name=model_name,
            batch_size=batch_size,
        )

        # Compute all similarities directly (no summary)
        model = SentenceTransformer(model_name)
        toxic = self.df["toxic_sentence"].astype(str).tolist()
        neutral = self.df["neutral_sentence"].astype(str).tolist()

        emb_t = model.encode(toxic, convert_to_tensor=True)
        emb_n = model.encode(neutral, convert_to_tensor=True)
        cos_sim = torch.nn.functional.cosine_similarity(emb_t, emb_n)

        df_out = self.df.copy()
        df_out["semantic_similarity"] = cos_sim.cpu().numpy()

        outliers = df_out[df_out["semantic_similarity"] < threshold]

        self.logger.info(
            f"Found {len(outliers)} semantic outliers (similarity < {threshold})"
        )

        return outliers

    # ============================================================
    # ================== VIS: LENGTH DISTRIBUTIONS ===============
    # ============================================================

    def plot_length_distribution(self, by_language: bool = True, bins: int = 40):
        """
        Plot distributions of sentence lengths (word-based).
        """

        if "toxic_sentence" not in self.df:
            raise ValueError("Dataset lacks 'toxic_sentence' column.")

        df = self.df.copy()
        df["toxic_len"] = df["toxic_sentence"].astype(str).apply(lambda x: len(x.split()))
        df["neutral_len"] = df["neutral_sentence"].astype(str).apply(lambda x: len(x.split()))

        plt.figure(figsize=(10, 5))
        sns.histplot(df["toxic_len"], color="red", label="Toxic", kde=True, bins=bins, alpha=0.5)
        sns.histplot(df["neutral_len"], color="blue", label="Neutral", kde=True, bins=bins, alpha=0.5)
        plt.title("Length Distribution (words)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        if by_language and "language" in df:
            g = sns.FacetGrid(df, col="language", col_wrap=3, sharex=False)
            g.map(sns.histplot, "toxic_len", kde=True, color="red", alpha=0.5)
            g.map(sns.histplot, "neutral_len", kde=True, color="blue", alpha=0.5)
            g.fig.suptitle("Length Distributions by Language")
            plt.tight_layout()
            plt.show()

    # ============================================================
    # ================== VIS: OVERLAP DISTRIBUTIONS ==============
    # ============================================================

    def plot_overlap_distribution(self, by_language: bool = True, bins: int = 40):
        """
        Plot histograms and KDE for equal_percentage.
        """

        if "equal_percentage" not in self.df:
            raise ValueError("equal_percentage missing. Compute it first.")

        plt.figure(figsize=(10, 5))
        sns.histplot(self.df["equal_percentage"], bins=bins, kde=True, color="purple", alpha=0.6)
        plt.title("Overlap Distribution (equal_percentage)")
        plt.xlabel("Overlap (%)")
        plt.tight_layout()
        plt.show()

        if by_language and "language" in self.df:
            g = sns.FacetGrid(self.df, col="language", col_wrap=3, sharex=False)
            g.map(sns.histplot, "equal_percentage", kde=True, color="purple", alpha=0.6)
            g.fig.suptitle("Overlap Distribution by Language")
            plt.tight_layout()
            plt.show()

    # ============================================================
    # =================== VIS: DIFFICULTY MAP ====================
    # ============================================================

    def plot_difficulty_by_language(self, low=30, high=70):
        """
        Bar plot showing count of easy/medium/hard per language.
        """

        if "equal_percentage" not in self.df:
            raise ValueError("equal_percentage missing.")

        df = self.df.copy()
        df["difficulty"] = np.select(
            [df["equal_percentage"] < low,
             (df["equal_percentage"] >= low) & (df["equal_percentage"] < high),
             df["equal_percentage"] >= high],
            ["hard", "medium", "easy"]
        )

        if "language" not in df:
            raise ValueError("No 'language' column present.")

    # ============================================================
    # ===== VIS: SCATTER PLOT (SEMANTIC SIMILARITY VS OVERLAP) ===
    # ============================================================

    def plot_semantic_vs_overlap(
        self,
        model_name="sentence-transformers/LaBSE",
        batch_size=32
    ):
        """
        Scatter plot comparing semantic similarity and overlap.
        Helps detect problematic examples.
        """

        if "equal_percentage" not in self.df:
            raise ValueError("equal_percentage missing.")

        # Compute similarities
        self.logger.info("Computing embedding similarity for scatter plot...")
        model = SentenceTransformer(model_name)
        toxic = self.df["toxic_sentence"].astype(str).tolist()
        neutral = self.df["neutral_sentence"].astype(str).tolist()

        emb_t = model.encode(toxic, convert_to_tensor=True, batch_size=batch_size)
        emb_n = model.encode(neutral, convert_to_tensor=True, batch_size=batch_size)
        cos_sim = torch.nn.functional.cosine_similarity(emb_t, emb_n).cpu().numpy()

        df = self.df.copy()
        df["semantic_similarity"] = cos_sim

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df,
            x="equal_percentage",
            y="semantic_similarity",
            hue="language" if "language" in df else None,
            palette="tab10",
            alpha=0.7,
        )
        plt.title("Semantic Similarity vs Overlap")
        plt.xlabel("Overlap (%)")
        plt.ylabel("Cosine Similarity")
        plt.tight_layout()
        plt.show()

    # ============================================================
    # ===================== VIS: HEATMAP ==========================
    # ============================================================

    def plot_language_similarity_heatmap(
        self,
        model_name="sentence-transformers/LaBSE",
        batch_size=32
    ):
        """
        Heatmap showing mean semantic similarity per language.
        """

        if "language" not in self.df:
            raise ValueError("No 'language' column for heatmap.")

        model = SentenceTransformer(model_name)
        toxic = self.df["toxic_sentence"].astype(str).tolist()
        neutral = self.df["neutral_sentence"].astype(str).tolist()

        emb_t = model.encode(toxic, convert_to_tensor=True, batch_size=batch_size)
        emb_n = model.encode(neutral, convert_to_tensor=True, batch_size=batch_size)
        sims = torch.nn.functional.cosine_similarity(emb_t, emb_n).cpu().numpy()

        df = self.df.copy()
        df["semantic_similarity"] = sims

        heatmap_data = df.groupby("language")["semantic_similarity"].mean().to_frame()

        plt.figure(figsize=(6, 6))
        sns.heatmap(heatmap_data, annot=True, cmap="viridis")
        plt.title("Mean Semantic Similarity per Language")
        plt.tight_layout()
        plt.show()

    # ============================================================
    # =================== REPORT GENERATION =======================
    # ============================================================

    def _fig_to_base64(self, fig: Figure) -> str:
        """
        Convert a matplotlib figure to a base64 string for embedding in HTML.
        """
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        return encoded

    # ============================================================
    # =============== INTERNAL PLOT GENERATION UTILS ==============
    # ============================================================

    def _create_length_plots(self) -> List[Figure]:
        """Generate global and per-language length distribution plots."""
        figs = []

        df = self.df.copy()
        df["toxic_len"] = df["toxic_sentence"].astype(str).apply(lambda x: len(x.split()))
        df["neutral_len"] = df["neutral_sentence"].astype(str).apply(lambda x: len(x.split()))

        # ---- Global plot ----
        fig = plt.figure(figsize=(10, 5))
        sns.histplot(df["toxic_len"], kde=True, color="red", label="Toxic", alpha=0.6)
        sns.histplot(df["neutral_len"], kde=True, color="blue", label="Neutral", alpha=0.6)
        plt.title("Length Distribution (global)")
        plt.legend()
        figs.append(fig)

        # ---- Per-language plots ----
        if "language" in df.columns:
            languages = df["language"].unique()
            for lang in languages:
                dfl = df[df["language"] == lang]

                fig = plt.figure(figsize=(8, 4))
                sns.histplot(dfl["toxic_len"], kde=True, color="red", alpha=0.6, label="Toxic")
                sns.histplot(dfl["neutral_len"], kde=True, color="blue", alpha=0.6, label="Neutral")
                plt.title(f"Length Distribution — {lang}")
                plt.legend()
                figs.append(fig)

        return figs

    def _create_overlap_plots(self) -> List[Figure]:
        """Generate global and per-language overlap distribution plots."""
        figs = []

        df = self.df.copy()

        # ---- Global ----
        fig = plt.figure(figsize=(10, 5))
        sns.histplot(df["equal_percentage"], kde=True, color="purple", alpha=0.7)
        plt.title("Overlap Distribution (global)")
        plt.xlabel("Overlap (%)")
        figs.append(fig)

        # ---- Per-language ----
        if "language" in df.columns:
            for lang in df["language"].unique():
                dfl = df[df["language"] == lang]

                fig = plt.figure(figsize=(8, 4))
                sns.histplot(dfl["equal_percentage"], kde=True, color="purple", alpha=0.7)
                plt.title(f"Overlap Distribution — {lang}")
                plt.xlabel("Overlap (%)")
                figs.append(fig)

        return figs

    def _create_difficulty_plots(self, low=30, high=70) -> List[Figure]:
        """Generate global and per-language difficulty bucket plots."""
        figs = []

        df = self.df.copy()

        if "difficulty" in df.columns:
            df = df.drop(columns=["difficulty"])

        # ---- Create new difficulty labels ----
        df["difficulty"] = np.select(
            [
                df["equal_percentage"] < low,
                (df["equal_percentage"] >= low) & (df["equal_percentage"] < high),
                df["equal_percentage"] >= high,
            ],
            ["hard", "medium", "easy"],
            default="medium"
        )

        # ---- Global ----
        fig = plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x="difficulty", palette="viridis")
        plt.title("Difficulty Distribution (global)")
        figs.append(fig)

        # ---- Per-language ----
        if "language" in df.columns:
            for lang in df["language"].unique():
                dfl = df[df["language"] == lang]

                fig = plt.figure(figsize=(8, 4))
                sns.countplot(data=dfl, x="difficulty", palette="viridis")
                plt.title(f"Difficulty Distribution — {lang}")
                figs.append(fig)

        return figs


    def generate_report(
        self,
        output_path: str = "dataset_report.html",
        include_semantics: bool = True,
        include_difficulty: bool = True,
        include_noise: bool = True,
        model_name: str = "sentence-transformers/LaBSE",
        threshold_outliers: float = 0.45,
    ):
        """
        Generate a complete HTML report including:
        - basic stats
        - length (global + by_language)
        - overlap (global + by_language)
        - difficulty (global + by_language)
        - noise stats
        - semantic similarity (optional)
        """

        self.logger.info("Generating dataset analysis report...")

        html = []
        push = html.append

        push("<html><head><title>Dataset Report</title></head><body>")
        push("<h1>Multilingual Detoxification Dataset Report</h1><hr>")

        # --------------------------
        # BASIC STATS
        # --------------------------
        push("<h2>1. Basic Statistics</h2>")
        basic = self.basic_stats()
        push(f"<pre>{basic}</pre>")

        # --------------------------
        # LENGTH ANALYSIS
        # --------------------------
        push("<h2>2. Length Analysis</h2>")
        lens = self.length_stats(by_language=True)
        push(f"<pre>{lens}</pre>")

        for fig in self._create_length_plots():
            encoded = self._fig_to_base64(fig)
            push(f'<img src="data:image/png;base64,{encoded}" width="600">')
            plt.close(fig)

        # --------------------------
        # OVERLAP ANALYSIS
        # --------------------------
        push("<h2>3. Overlap Analysis</h2>")
        overlap = self.overlap_stats(by_language=True)
        push(f"<pre>{overlap}</pre>")

        for fig in self._create_overlap_plots():
            encoded = self._fig_to_base64(fig)
            push(f'<img src="data:image/png;base64,{encoded}" width="600">')
            plt.close(fig)

        # --------------------------
        # DIFFICULTY
        # --------------------------
        if include_difficulty:
            push("<h2>4. Difficulty</h2>")
            difficulty = self.difficulty_buckets()
            push(f"<pre>{difficulty}</pre>")

            for fig in self._create_difficulty_plots():
                encoded = self._fig_to_base64(fig)
                push(f'<img src="data:image/png;base64,{encoded}" width="500">')
                plt.close(fig)

        # --------------------------
        # NOISE
        # --------------------------
        if include_noise:
            push("<h2>5. Noise & Data Quality</h2>")
            noise = self.noise_stats()
            push(f"<pre>{noise}</pre>")

        # --------------------------
        # SEMANTICS
        # --------------------------
        if include_semantics:
            push("<h2>6. Semantic Analysis</h2>")

            # BERTScore
            try:
                bert = self.bertscore_stats()
                push("<h3>BERTScore</h3>")
                push(f"<pre>{bert}</pre>")
            except Exception as e:
                push(f"<pre>BERTScore failed: {e}</pre>")

            # Embedding similarity
            try:
                sem = self.sentence_embedding_similarity(model_name=model_name)
                push("<h3>Embedding Similarity</h3>")
                push(f"<pre>{sem}</pre>")
            except Exception as e:
                push(f"<pre>Embedding similarity failed: {e}</pre>")

            # Outliers
            try:
                outliers = self.semantic_outliers(model_name=model_name, threshold=threshold_outliers)
                push("<h3>Semantic Outliers</h3>")
                push(outliers.head().to_html())
                push(f"<p>Total outliers: {len(outliers)}</p>")
            except Exception as e:
                push(f"<pre>Outlier detection failed: {e}</pre>")

        # --------------------------
        # FOOTER
        # --------------------------
        push("<hr><p><i>Report generated by DatasetAnalyzer</i></p></body></html>")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))

        self.logger.info(f"Report successfully generated → {output_path}")

