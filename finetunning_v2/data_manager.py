import logging
from typing import List, Dict, Optional

from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from transformers import T5TokenizerFast

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

    def compute_overlap_feature(self) -> None:
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

        self.raw_dataset = self.raw_dataset.map(_map, batched=True)
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
    
    def get_overlap_splits(self, split_name: str = "train", threshold: float = 70.0):
        """
        Devuelve dos subsets del split indicado:
        - easy: ejemplos con overlap >= threshold
        - hard: ejemplos con overlap < threshold
        """
        if self.splits is None:
            raise ValueError("Los splits no están generados. Ejecuta make_splits() primero.")

        if split_name not in self.splits:
            raise ValueError(f"El split {split_name} no existe. Opciones: {list(self.splits.keys())}")

        ds = self.splits[split_name]

        if "equal_percentage" not in ds.column_names:
            raise ValueError("No existe la columna 'equal_percentage'. Ejecuta compute_overlap_feature().")

        self.logger.info(f"Separando split '{split_name}' con umbral {threshold}%...")

        easy = ds.filter(lambda x: x["equal_percentage"] >= threshold)
        hard = ds.filter(lambda x: x["equal_percentage"] < threshold)

        self.logger.info(f" - EASY  (>= {threshold}%): {len(easy)} ejemplos")
        self.logger.info(f" - HARD  (<  {threshold}%): {len(hard)} ejemplos")

        return easy, hard


    # ============================================================
    # ===================== TOKENIZATION ==========================
    # ============================================================

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Limpieza ligera y segura. No agresiva.
        """
        if text is None:
            return ""

        # Normalizar unicode
        import unicodedata
        text = unicodedata.normalize("NFC", text)

        # Sustituir saltos de línea/tabs por espacios
        text = text.replace("\n", " ").replace("\t", " ")

        # Quitar espacios múltiples
        text = " ".join(text.split())

        return text.strip()


    def _tokenize_batch(self, batch):
        toxic_clean = [self.clean_text(t) for t in batch["toxic_sentence"]]
        neutral_clean = [self.clean_text(n) for n in batch["neutral_sentence"]]

        inputs = [f"{self.prefix}{t}" for t in toxic_clean]
        targets = neutral_clean

        enc_inputs = self.tokenizer(
            inputs,
            truncation=True,
            max_length=256,
            padding=False
        )

        with self.tokenizer.as_target_tokenizer():
            enc_targets = self.tokenizer(
                targets,
                truncation=True,
                max_length=256,
                padding=False
            )

        enc_inputs["labels"] = enc_targets["input_ids"]
        return enc_inputs

    def tokenize_all_splits(self):
        """
        Tokenize all dataset splits using the configured tokenizer.
        This method applies tokenization to all splits in the dataset (train, validation, test)
        using batch processing for efficiency. The original columns are removed after tokenization,
        keeping only the tokenized features.
        Returns:
            DatasetDict: A dictionary containing tokenized versions of all splits, where each
                        split contains tokenized input features suitable for model training.
        Raises:
            ValueError: If splits have not been generated yet (self.splits is None).
        Note:
            - Requires that splits have been previously generated using generate_splits()
            - Uses the _tokenize_batch method for batch tokenization
            - Removes original text columns after tokenization to save memory
            - Progress is logged during the tokenization process
        """
        if self.splits is None:
            raise ValueError("Los splits no están generados.")

        self.logger.info("Tokenizando splits...")

        self.tokenized_splits = DatasetDict({
            split: self.splits[split].map(
                self._tokenize_batch,
                batched=True,
                remove_columns=self.splits[split].column_names
            )
            for split in self.splits
        })

        self.logger.info("Tokenización completada.")
        return self.tokenized_splits

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


    # ============================================================
    # ================== PIPELINE COMPLETO =======================
    # ============================================================

    def full_prepare(self):
        """
        Pipeline completo:
        - Carga dataset
        - Overlap
        - Splits
        - Tokenización
        - Summary
        """
        self.load_main_dataset()
        self.compute_overlap_feature()
        self.make_splits()
        self.tokenize_all_splits()
        self.summary()

        return self.tokenized_splits


