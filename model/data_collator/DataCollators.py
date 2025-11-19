import torch
from transformers import AutoTokenizer
from typing import List, Dict, Any

class ToxicDataCollator:
    """
    Data collator para tokenizar y generar tensores de entrada.
    Diseñado para datasets con palabras individuales y etiquetas binarias.
    """

    def __init__(self, tokenizer_name: str, max_length: int = 32):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extraer textos y etiquetas
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]

        # Tokenizar (cada palabra = una secuencia individual)
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            is_split_into_words=False
        )

        # Alinear etiquetas con sub-tokens
        all_labels = []
        for i, label in enumerate(labels):
            word_ids = encoded.word_ids(batch_index=i)
            label_ids = []
            prev_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != prev_word_idx:
                    label_ids.append(label)
                else:
                    label_ids.append(-100)
                prev_word_idx = word_idx
            all_labels.append(label_ids)

        encoded["labels"] = torch.tensor(all_labels, dtype=torch.long)
        return encoded