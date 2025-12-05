from collections import defaultdict
import unicodedata
from typing import List
from datasets import load_dataset, concatenate_datasets

def normalize_token(token):
    token = unicodedata.normalize("NFKC", token)
    return token.strip().lower()

def build_lexicons_from_dataset(dataset):
    toxic_lexicons = {str(lang): [] for lang in set(dataset.data['language'])}

    for lexicon, lang in zip(dataset.data['text'], dataset.data['language']):
        normalized_lexicon = normalize_token(str(lexicon))
        toxic_lexicons[str(lang)] += [normalized_lexicon]

    return toxic_lexicons

def load_toxic_lexicon(lang_list: List[str] = None):
    
    lang = ['en', 'am', 'ar', 'de', 'es', 'hi', 'ru', 'uk', 'zh'] if not lang_list else lang_list
    
    url_template = "https://raw.githubusercontent.com/Mariogarber/PLN-Project/main/dataset/multilingual_toxic_lexicon/{}.parquet"
    
    datasets = []
    for lang_code in lang:
        url = url_template.format(lang_code)
        try:
            # Load directly as HuggingFace dataset
            ds = load_dataset('parquet', data_files=url, split='train')
            # Add language column
            ds = ds.add_column("language", [lang_code] * len(ds))
            datasets.append(ds)
            print(f"Loaded {len(ds)} samples for language: {lang_code}")
        except Exception as e:
            print(f"Failed to load data for {lang_code}: {e}")

    # Concatenate all language datasets
    combined_dataset = concatenate_datasets(datasets)
    print(f"Total forbidden words dataset size: {len(combined_dataset)} samples across {len(datasets)} languages")
    return combined_dataset