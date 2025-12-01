from torch.utils.data import Dataset
import pandas as pd
from typing import List
import pandas as pd
from bert_score import score as bertscore
import sacrebleu
from rouge_score import rouge_scorer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging
from datasets import load_dataset, concatenate_datasets, DatasetDict

TOKENIZER = T5Tokenizer.from_pretrained("google/mt5-base")
MODEL = T5ForConditionalGeneration.from_pretrained("google/mt5-base")

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

def load_data(lang_list: List[str]):
    """Load multilingual detoxification dataset directly as HuggingFace Dataset with train/test split"""
    lang = ['en', 'am', 'ar', 'de', 'es', 'hi', 'ru', 'uk', 'zh'] if not lang_list else lang_list
    
    url_template = "https://raw.githubusercontent.com/Mariogarber/PLN-Project/main/dataset/toxic_nontoxic/multilingual_paradetox_{}.parquet"
    
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
    
    # Shuffle the combined dataset
    combined_dataset = combined_dataset.shuffle(seed=42)
    
    # Split into train/test (80/20 split)
    train_test_split = combined_dataset.train_test_split(test_size=0.2, seed=42)
    
    # Create DatasetDict with train and test splits
    dataset_dict = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })
    
    LOGGER.info(f"Total dataset size: {len(combined_dataset)} samples across {len(datasets)} languages")
    LOGGER.info(f"Train split: {len(dataset_dict['train'])} samples")
    LOGGER.info(f"Test split: {len(dataset_dict['test'])} samples")
    
    return dataset_dict

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

def get_forbidden_token_ids(forbidden_words_list: List[str], tokenizer: T5Tokenizer) -> List[int]:
    """
    Get forbidden token IDs from forbidden words.
    
    CRITICAL: This function now handles tokenization context properly.
    We test multiple tokenization contexts to catch different token splits.
    """
    forbidden_token_ids = set()
    
    # Test contexts to catch different tokenization patterns
    test_contexts = [
        "",  # Standalone word
        " ",  # With leading space
        " {} ",  # With surrounding spaces
        "The {} is",  # In sentence context
        "{}!",  # With punctuation
        "{} and"  # In compound context
    ]
    
    for word in forbidden_words_list:
        # Skip empty or very short words
        if not word or len(word.strip()) < 2:
            continue
            
        word = word.strip().lower()
        
        # Test word in different contexts
        for context in test_contexts:
            if "{}" in context:
                test_text = context.format(word)
            else:
                test_text = context + word
                
            # Tokenize and find word boundaries
            token_ids = tokenizer.encode(test_text, add_special_tokens=False)
            
            # For contexts with {}, try to isolate the word tokens
            if "{}" in context:
                # Also tokenize the context without the word to subtract
                context_only = context.replace(" {} ", " ").replace("{}", "").strip()
                if context_only:
                    context_tokens = tokenizer.encode(context_only, add_special_tokens=False)
                    # Find tokens that are in test_text but not in context_only
                    word_tokens = [tid for tid in token_ids if tid not in context_tokens]
                    forbidden_token_ids.update(word_tokens)
                else:
                    forbidden_token_ids.update(token_ids)
            else:
                forbidden_token_ids.update(token_ids)
    
    # Also add direct tokenization for safety
    for word in forbidden_words_list:
        if word and len(word.strip()) >= 2:
            word = word.strip().lower()
            direct_tokens = tokenizer.encode(word, add_special_tokens=False)
            forbidden_token_ids.update(direct_tokens)
    
    result = list(forbidden_token_ids)
    print(f"🚫 Generated {len(result)} forbidden token IDs from {len(forbidden_words_list)} words")
    return result

def tokenize_function(inputs, prefix="detoxify: "):
    """Tokenize the dataset for T5 training"""
    # Create input text with task prefix
    if prefix:
        input_texts = [f"{prefix}{text}" for text in inputs['toxic_sentence']]
    else:
        input_texts = inputs['toxic_sentence']
    target_texts = inputs['neutral_sentence']
    
    # Tokenize inputs
    model_inputs = TOKENIZER(
        input_texts,
        max_length=512,
        truncation=True,
        padding=False  # We'll pad in the data collator
    )
    
    # Tokenize targets
    with TOKENIZER.as_target_tokenizer():
        labels = TOKENIZER(
            target_texts,
            max_length=512,
            truncation=True,
            padding=False
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_dataset(dataset, prefix="detoxify: "):
    """Preprocess the entire dataset with tokenization"""
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, prefix=prefix),
        batched=True,
        remove_columns=dataset.column_names
    )
    return tokenized_dataset

def split_dataset(dataset, train_size=0.8, seed=42):
    """Split dataset into train and test sets"""
    split = dataset.train_test_split(test_size=1 - train_size, seed=seed)
    return split['train'], split['test']

def train_eval_test_split(dataset, train_size=0.7, eval_size=0.15, seed=42):
    """Split dataset into train, eval, and test sets"""
    train_eval_split = dataset.train_test_split(test_size=1 - train_size, seed=seed)
    train_dataset = train_eval_split['train']
    temp_dataset = train_eval_split['test']
    
    eval_test_split = temp_dataset.train_test_split(test_size=eval_size / (1 - train_size), seed=seed)
    eval_dataset = eval_test_split['train']
    test_dataset = eval_test_split['test']
    
    return train_dataset, eval_dataset, test_dataset