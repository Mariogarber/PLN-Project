import torch
import random
from typing import List, Dict, Any

class DetoxificationDataCollator:
    """
    Data collator para tareas de detoxificación de texto.
    Maneja pares de oraciones (tóxica -> neutral) para entrenar modelos
    de detoxificación tipo seq2seq o masked language modeling.
    """
    
    def __init__(self, tokenizer, max_length=512, task_type="seq2seq"):
        """
        Args:
            tokenizer: Tokenizer de Hugging Face
            max_length: Longitud máxima de las secuencias
            task_type: "seq2seq" para generación, "mlm" para masked language modeling
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        
        # Para MLM
        self.mlm_probability = 0.15
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = len(tokenizer)

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """
        Procesa un batch de pares toxic_text -> neutral_text
        
        Args:
            batch: Lista de diccionarios con keys 'toxic_text' y 'neutral_text'
            
        Returns:
            Dict con tensores listos para el entrenamiento
        """
        if self.task_type == "seq2seq":
            return self._collate_seq2seq(batch)
        elif self.task_type == "mlm":
            return self._collate_mlm(batch)
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")
    
    def _collate_seq2seq(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """
        Collator para tareas seq2seq: toxic_text como input, neutral_text como target
        """
        toxic_texts = [item["toxic_text"] for item in batch]
        neutral_texts = [item["neutral_text"] for item in batch]
        
        # Tokenizar inputs (textos tóxicos)
        input_encodings = self.tokenizer(
            toxic_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Tokenizar targets (textos neutrales)
        target_encodings = self.tokenizer(
            neutral_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": target_encodings["input_ids"],
            "decoder_attention_mask": target_encodings["attention_mask"]
        }
    
    def _collate_mlm(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """
        Collator para MLM: usa texto neutral como input y aplica masking
        """
        # Para MLM, usamos los textos neutrales como target
        neutral_texts = [item["neutral_text"] for item in batch]
        
        # Tokenizar
        encodings = self.tokenizer(
            neutral_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Aplicar masking
        input_ids, labels = self._apply_mlm_masking(encodings["input_ids"])
        
        return {
            "input_ids": input_ids,
            "attention_mask": encodings["attention_mask"],
            "labels": labels
        }
    
    def _apply_mlm_masking(self, input_ids: torch.Tensor) -> tuple:
        """
        Aplica la política de masking de BERT/RoBERTa para MLM
        """
        labels = input_ids.clone()
        
        # Crear máscara de probabilidad
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # No enmascarar tokens especiales
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        # Seleccionar tokens a enmascarar
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Solo predecir tokens enmascarados
        
        # Aplicar estrategia de masking (80% [MASK], 10% random, 10% unchanged)
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id
        
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_tokens = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_tokens[indices_random]
        
        return input_ids, labels

class ContrastiveDetoxificationCollator:
    """
    Collator alternativo que crea ejemplos contrastivos para aprender
    la diferencia entre texto tóxico y neutral.
    """
    
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """
        Crea un batch con ejemplos tóxicos y neutrales mezclados
        """
        texts = []
        labels = []
        
        for item in batch:
            # Agregar texto tóxico (label=1)
            texts.append(item["toxic_text"])
            labels.append(1)
            
            # Agregar texto neutral (label=0)
            texts.append(item["neutral_text"])
            labels.append(0)
        
        # Tokenizar todos los textos
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long)
        }