from typing import List
import pandas as pd
from datasets import load_dataset, concatenate_datasets, DatasetDict

import logging

LOGGER = logging.getLogger(__name__)

def load_toxic_spans(lang_list: List[str] = None):
    
    lang = ['en', 'am', 'ar', 'de', 'es', 'hi', 'ru', 'uk', 'zh'] if not lang_list else lang_list
    
    url_template = "https://raw.githubusercontent.com/Mariogarber/PLN-Project/main/dataset/multilingual_toxic_spans/{}.parquet"
    
    datasets = []
    for lang_code in lang:
        url = url_template.format(lang_code)
        try:
            # Load directly as HuggingFace dataset
            ds = load_dataset('parquet', data_files=url, split='train')
            # Add language column
            ds = ds.add_column("language", [lang_code] * len(ds))
            datasets.append(ds)
            LOGGER.info(f"Loaded {len(ds)} samples for language: {lang_code}")
        except Exception as e:
            LOGGER.warning(f"Failed to load data for {lang_code}: {e}")

    # Concatenate all language datasets
    combined_dataset = concatenate_datasets(datasets)
    LOGGER.info(f"Total forbidden words dataset size: {len(combined_dataset)} samples across {len(datasets)} languages")
    return combined_dataset

def get_forbidden_words_list(toxic_spans_dataset) -> List[str]:
    forbidden_words = toxic_spans_dataset['Negative Connotations']

    # Convert to a flat list of unique words
    forbidden_words_list = []
    for words in forbidden_words:
        if words:  # Check if not None or empty
            # Split by common separators and add to list
            if isinstance(words, str):
                # Split by comma, semicolon, or other separators
                word_parts = words.replace(';', ',').replace('|', ',').split(',')
                for word in word_parts:
                    word = word.strip().lower()
                    if word and word not in forbidden_words_list:
                        forbidden_words_list.append(word)

    forbidden_words_list = list(set(forbidden_words_list))
    return forbidden_words_list