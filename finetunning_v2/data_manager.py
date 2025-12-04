import logging
from typing import List, Dict, Optional

from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from transformers import T5TokenizerFast
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np


class DataManager:
    """
    A comprehensive data management class for multilingual text detoxification using mT5.
    This class handles the complete pipeline for preparing datasets for text detoxification tasks,
    including data loading, preprocessing, feature engineering, stratified splitting, and tokenization
    specifically designed for transformer models like mT5.
    Key Features:
        - Multilingual dataset loading from remote sources
        - Toxic span analysis and forbidden word extraction
        - Text overlap calculation between toxic and neutral pairs
        - Stratified dataset splitting (train/eval/test)
        - HuggingFace-compatible tokenization and formatting
        - Comprehensive logging and dataset statistics
    Attributes:
        languages (List[str]): List of language codes to process (default: 9 languages)
        prefix (str): Task-specific prefix added to input sequences for mT5
        seed (int): Random seed for reproducible splits and shuffling
        tokenizer (T5TokenizerFast): Pre-trained mT5 tokenizer instance
        raw_dataset (Dataset): Original loaded dataset before processing
        splits (DatasetDict): Train/eval/test splits of the raw data
        tokenized_splits (DatasetDict): Final tokenized datasets ready for training
        toxic_spans_dataset (Dataset): Dataset containing toxic span annotations
        forbidden_words (List[str]): List of identified toxic/forbidden words
        forbidden_token_ids (List[int]): Token IDs corresponding to forbidden words
    Usage:
        >>> dm = DataManager(languages=['en', 'es'], seed=42)
        >>> tokenized_data = dm.full_prepare()
        >>> # Ready for model training
    The class follows a pipeline approach where each method builds upon previous steps,
    culminating in the `full_prepare()` method that executes the complete workflow.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        prefix: str = "detoxify_keep_meaning: ",
        tokenizer_name: str = "google/mt5-base",
        seed: int = 42,
        log_level=logging.INFO,
    ):
        # ----------------------
        # LOGGING CONFIGURATION
        # ----------------------
        self.logger = logging.getLogger("DataManager")
        self.logger.setLevel(log_level)

        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(handler)

        # ----------------------
        # BASIC PARAMS
        # ----------------------
        self.languages = languages or ['en', 'am', 'ar', 'de', 'es', 'hi', 'ru', 'uk', 'zh']
        self.prefix = prefix
        self.seed = seed

        self.logger.info(f"Inicializando DataManager para idiomas: {self.languages}")

        # ----------------------
        # TOKENIZER
        # ----------------------
        self.tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained(tokenizer_name)
        self.logger.info(f"Tokenizer cargado: {tokenizer_name}")

        # ----------------------
        # INTERNAL STORAGE
        # ----------------------
        self.raw_dataset: Optional[Dataset] = None
        self.splits: Optional[DatasetDict] = None
        self.tokenized_splits: Optional[DatasetDict] = None

        self.toxic_spans_dataset: Optional[Dataset] = None
        self.forbidden_words: Optional[List[str]] = None
        self.forbidden_token_ids: Optional[List[int]] = None

    # ============================================================
    # ====================== LOADING DATA =========================
    # ============================================================

    def load_main_dataset(self) -> DatasetDict:
        """
        Carga el dataset principal tox → neutral desde GitHub.
        """
        url_template = (
            "https://raw.githubusercontent.com/Mariogarber/PLN-Project/main/"
            "dataset/toxic_nontoxic/multilingual_paradetox_{}.parquet"
        )

        datasets = []

        for lang in self.languages:
            url = url_template.format(lang)
            try:
                ds = load_dataset("parquet", data_files=url, split="train")
                ds = ds.add_column("language", [lang] * len(ds))
                datasets.append(ds)
                self.logger.info(f"Cargado {len(ds)} ejemplos para {lang}")

            except Exception as e:
                self.logger.warning(f"No se pudo cargar {lang}: {e}")

        if not datasets:
            raise RuntimeError("No se pudo cargar ningún dataset.")

        combined = concatenate_datasets(datasets)
        combined = combined.shuffle(self.seed)

        self.raw_dataset = combined
        self.logger.info(f"Dataset total cargado: {len(combined)} muestras")

        return combined

    # ============================================================
    # ================= FEATURES (OVERLAP) ========================
    # ============================================================

    @staticmethod
    def _percent_equal_words(toxic: str, neutral: str) -> float:
        toxic_words = set(str(toxic).split())
        neutral_words = str(neutral).split()

        if not neutral_words:
            return 0.0

        overlap = sum(1 for w in neutral_words if w in toxic_words)
        return overlap / len(neutral_words) * 100

    def compute_overlap_feature(self, threshold_1: float = 70.0, threshold_2: float = 30.0) -> None:
        """
        Inserta la columna "equal_percentage" en self.raw_dataset.
        """
        if self.raw_dataset is None:
            raise ValueError("Dataset no cargado.")

        def _map(batch):
            values = []
            for t, n in zip(batch["toxic_sentence"], batch["neutral_sentence"]):
                values.append(self._percent_equal_words(t, n))
            return {"equal_percentage": values}
        
        def _map_difficulty(batch):
            difficulties = []
            for perc in batch["equal_percentage"]:
                if perc >= threshold_1:
                    difficulties.append("easy")
                elif perc >= threshold_2:
                    difficulties.append("medium")
                else:
                    difficulties.append("hard")
            return {"difficulty": difficulties}

        self.raw_dataset = self.raw_dataset.map(_map, batched=True)
        self.raw_dataset = self.raw_dataset.map(_map_difficulty, batched=True)
        self.logger.info("Cálculo de overlap (equal_percentage) completado.")

    # ============================================================
    # ======================== SPLITS =============================
    # ============================================================

    def make_splits(self, train=0.7, eval=0.15):
        """
        Split estratificado simple usando HF train_test_split.
        """
        if self.raw_dataset is None:
            raise ValueError("Dataset no cargado.")

        self.logger.info("Generando splits train/eval/test...")

        train_test = self.raw_dataset.train_test_split(test_size=1 - train, seed=self.seed)
        train_set = train_test["train"]
        temp = train_test["test"]

        eval_ratio = eval / (1 - train)
        eval_test = temp.train_test_split(test_size=1 - eval_ratio, seed=self.seed)

        self.splits = DatasetDict(
            {
                "train": eval_test["train"],
                "eval": eval_test["test"],
                "test": temp
            }
        )

        for split in self.splits:
            self.logger.info(f"{split.upper()}: {len(self.splits[split])} muestras")

        return self.splits
    
    def get_overlap_splits(self, split_name: str = "train", threshold_1: float = 70.0, threshold_2: float = 30.0):
        """
        Devuelve dos subsets del split indicado:
        - easy: ejemplos con overlap >= threshold_1
        - medium: ejemplos con overlap < threshold_1 y >= threshold_2
        - hard: ejemplos con overlap < threshold_2
        """
        if self.splits is None:
            raise ValueError("Los splits no están generados. Ejecuta make_splits() primero.")

        if split_name not in self.splits:
            raise ValueError(f"El split {split_name} no existe. Opciones: {list(self.splits.keys())}")

        ds = self.splits[split_name]

        if "equal_percentage" not in ds.column_names:
            raise ValueError("No existe la columna 'equal_percentage'. Ejecuta compute_overlap_feature().")

        self.logger.info(f"Separando split '{split_name}' con umbral {threshold_1}%...")

        easy = ds.filter(lambda x: x["equal_percentage"] >= threshold_1)
        medium = ds.filter(lambda x: (x["equal_percentage"] < threshold_1) & (x["equal_percentage"] >= threshold_2))
        hard = ds.filter(lambda x: x["equal_percentage"] < threshold_2)

        self.logger.info(f" - EASY  (>= {threshold_1}%): {len(easy)} ejemplos")
        self.logger.info(f" - MEDIUM ({threshold_2}% - {threshold_1}%): {len(medium)} ejemplos")
        self.logger.info(f" - HARD  (<  {threshold_2}%): {len(hard)} ejemplos")
        return easy, medium, hard
    
    def stratified_split(
        self,
        test_size=0.1,
        val_size=0.1,
        stratify_by=["language", "difficulty"]
    ):
        """
        Perform a safe stratified split for small multilingual detox datasets.

        Splits based on:
            - language
            - difficulty (easy/medium/hard)

        Reason: these are the only two stratification dimensions that keep buckets
        statistically stable with ~400 samples per language.

        Returns:
            self.splits = {"train": ..., "val": ..., "test": ...}
        """

        df = self.raw_dataset.to_pandas()

        # Check required columns exist
        for col in stratify_by:
            if col not in df.columns:
                raise ValueError(f"Missing required column for stratification: {col}")

        # Create combined stratification key
        df["strat_key"] = df[stratify_by].astype(str).agg("_".join, axis=1)

        # First: train+val vs test
        trainval_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["strat_key"],
            random_state=42
        )

        # Second: train vs val
        val_ratio = val_size / (1 - test_size)

        train_df, val_df = train_test_split(
            trainval_df,
            test_size=val_ratio,
            stratify=trainval_df["strat_key"],
            random_state=42
        )

        # Store as HF datasets
        self.splits = {
            "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
            "val": Dataset.from_pandas(val_df.reset_index(drop=True)),
            "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
        }

        self.logger.info(
            f"Stratified splits created:\n"
            f"  train: {len(train_df)}\n"
            f"  val:   {len(val_df)}\n"
            f"  test:  {len(test_df)}"
        )

        return self.splits
    
    def create_curriculum_datasets(self):
        """
        Create curriculum-learning datasets:
            - easy
            - medium
            - hard
            - full
            
        Only applied to TRAIN split.
        
        Requires:
            self.splits["train"] (HF Dataset)
            Column 'difficulty' in train split.
        """

        if not hasattr(self, "splits") or "train" not in self.splits:
            raise ValueError("You must run stratified_split() before creating curriculum datasets.")

        train_df = self.splits["train"].to_pandas()

        if "difficulty" not in train_df.columns:
            raise ValueError("Column 'difficulty' is required. Compute difficulty buckets before this step.")

        # Create separate subsets
        easy_df = train_df[train_df["difficulty"] == "easy"].reset_index(drop=True)
        medium_df = train_df[train_df["difficulty"] == "medium"].reset_index(drop=True)
        hard_df = train_df[train_df["difficulty"] == "hard"].reset_index(drop=True)

        # Convert back to HF Dataset
        self.curriculum = {
            "easy": Dataset.from_pandas(easy_df),
            "medium": Dataset.from_pandas(medium_df),
            "hard": Dataset.from_pandas(hard_df),
            "full": self.splits["train"],  # the original full train split
        }

        self.logger.info(
            "Curriculum datasets created:\n"
            f"  easy:   {len(easy_df)} samples\n"
            f"  medium: {len(medium_df)} samples\n"
            f"  hard:   {len(hard_df)} samples\n"
            f"  full:   {len(train_df)} samples"
        )

        return self.curriculum

    def build_curriculum(
        self,
        easy_threshold: float = 70.0,
        medium_threshold: float = 30.0,
        min_bucket_size: int = 50,
        drop_small_buckets: bool = True,
        regenerate: bool = True,
    ):
        """
        Build curriculum-learning datasets from TRAIN SPLIT only.

        Parameters
        ----------
        easy_threshold : float
            equal_percentage >= easy_threshold → EASY bucket
        medium_threshold : float
            equal_percentage between medium_threshold and easy_threshold → MEDIUM bucket
        min_bucket_size : int
            discard buckets smaller than this number (optional)
        drop_small_buckets : bool
            if True, remove buckets smaller than min_bucket_size
        regenerate : bool
            if True, overwrites existing curriculum

        Returns
        -------
        dict : {"easy": Dataset, "medium": Dataset, "hard": Dataset, "full": Dataset}
        """

        if self.splits is None or "train" not in self.splits:
            raise ValueError("You must run stratified_split() before building curriculum.")

        if not regenerate and hasattr(self, "curriculum"):
            return self.curriculum

        train_ds = self.splits["train"]

        if "equal_percentage" not in train_ds.column_names:
            raise ValueError(
                "equal_percentage not found. You must run compute_overlap_feature() before curriculum."
            )

        self.logger.info(
            f"Building curriculum datasets with thresholds: "
            f"easy >= {easy_threshold}, medium >= {medium_threshold}."
        )

        # -----------------------
        # BUILD RAW BUCKETS
        # -----------------------
        easy = train_ds.filter(lambda x: x["equal_percentage"] >= easy_threshold)
        medium = train_ds.filter(
            lambda x: (x["equal_percentage"] < easy_threshold)
                      and (x["equal_percentage"] >= medium_threshold)
        )
        hard = train_ds.filter(lambda x: x["equal_percentage"] < medium_threshold)

        buckets = {
            "easy": easy,
            "medium": medium,
            "hard": hard,
        }

        # -----------------------
        # OPTIONAL FILTERING
        # -----------------------
        if drop_small_buckets:
            for name, ds in list(buckets.items()):
                if len(ds) < min_bucket_size:
                    self.logger.warning(
                        f"Bucket '{name}' has only {len(ds)} samples < {min_bucket_size}. Dropping."
                    )
                    del buckets[name]

        # -----------------------
        # ALWAYS INCLUDE FULL
        # -----------------------
        buckets["full"] = train_ds

        # Store
        self.curriculum = buckets

        # Logging summary
        self.logger.info("Curriculum buckets created:")
        for k, ds in self.curriculum.items():
            self.logger.info(f"  {k}: {len(ds)} samples")

        return self.curriculum


    def get_curriculum_splits(self):
        """
        Safely returns curriculum splits.

        Returns
        -------
        dict: curriculum buckets
        """
        if not hasattr(self, "curriculum"):
            raise ValueError(
                "Curriculum not generated. Call build_curriculum() first."
            )

        return self.curriculum



    # ============================================================
    # ====================== SUMMARY ==============================
    # ============================================================

    def summary(self):
        """
        Muestra estadísticas básicas del dataset.
        """
        if self.raw_dataset is None:
            raise ValueError("Dataset no cargado.")

        df = pd.DataFrame(self.raw_dataset.select_columns(
            ["language", "toxic_sentence", "neutral_sentence", "equal_percentage"]
        ))

        self.logger.info("===== RESUMEN DEL DATASET =====")

        self.logger.info(f"Muestras totales: {len(df)}")
        self.logger.info("Muestras por idioma:")
        self.logger.info(str(df["language"].value_counts().to_dict()))

        self.logger.info(f"Overlap medio: {df['equal_percentage'].mean():.2f}%")
        self.logger.info("================================")

        # Add summary for splits if available
        if self.splits is not None:
            self.logger.info("===== RESUMEN DE SPLITS =====")
            for split_name, split_ds in self.splits.items():
                split_df = pd.DataFrame(split_ds.select_columns(
                    ["language", "equal_percentage"]
                ))
                self.logger.info(f"--- {split_name.upper()} ---")
                self.logger.info(f"Muestras: {len(split_df)}")
                self.logger.info("Muestras por idioma:")
                self.logger.info(str(split_df["language"].value_counts().to_dict()))
                self.logger.info(f"Overlap medio: {split_df['equal_percentage'].mean():.2f}%")
            self.logger.info("================================")


    # ============================================================
    # ================== PIPELINE COMPLETO =======================
    # ============================================================

    def full_prepare(self, stratify_columns: List[str] = ["language", "difficulty"]):
        """
        Pipeline completo:
        - Carga dataset
        - Overlap
        - Splits
        - Summary
        """
        self.load_main_dataset()
        self.compute_overlap_feature()
        self.stratified_split(stratify_by=stratify_columns)
        self.create_curriculum_datasets()
        self.summary()

        return self.splits


