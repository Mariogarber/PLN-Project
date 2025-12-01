from torch.utils.data import Dataset
import pandas as pd
from typing import List, Set, Union, Tuple
import pandas as pd
from bert_score import score as bertscore
import sacrebleu
from rouge_score import rouge_scorer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging
import torch
import numpy as np
from collections import defaultdict

TOKENIZER = T5Tokenizer.from_pretrained("google/mt5-base")
MODEL = T5ForConditionalGeneration.from_pretrained("google/mt5-base")


class ToxicTokenMaskGenerator:
    """
    Class to generate training masks that focus on toxic tokens.
    This enables targeted training on specific toxic words/tokens.
    """
    
    def __init__(
        self, 
        tokenizer, 
        forbidden_token_ids: List[int] = None,
        mask_strategy: str = "toxic_only"
    ):
        """
        Initialize the toxic token mask generator.
        
        Args:
            tokenizer: The tokenizer to use
            forbidden_token_ids: List of token IDs that are considered toxic
            mask_strategy: Strategy for creating masks
                - "toxic_only": Only train on positions with toxic tokens
                - "toxic_context": Train on toxic tokens and surrounding context
                - "inverse_toxic": Train on everything EXCEPT toxic tokens
        """
        self.tokenizer = tokenizer
        self.forbidden_token_ids = set(forbidden_token_ids) if forbidden_token_ids else set()
        self.mask_strategy = mask_strategy
        
        print(f"🎭 ToxicTokenMaskGenerator initialized")
        print(f"🚫 Tracking {len(self.forbidden_token_ids)} forbidden token IDs")
        print(f"📋 Strategy: {mask_strategy}")
    
    def create_toxic_token_mask(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        context_window: int = 2
    ) -> torch.Tensor:
        """
        Create a training mask focusing on toxic tokens.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Label token IDs [batch_size, seq_len] 
            context_window: Number of tokens around toxic tokens to include
            
        Returns:
            mask: Boolean tensor [batch_size, seq_len] where True = train on this position
        """
        batch_size, seq_len = labels.shape
        device = labels.device
        
        if self.mask_strategy == "toxic_only":
            # Only train on positions with toxic tokens
            mask = self._get_toxic_positions_mask(labels)
            
        elif self.mask_strategy == "toxic_context":
            # Train on toxic tokens and surrounding context
            toxic_mask = self._get_toxic_positions_mask(labels)
            context_mask = self._expand_mask_with_context(toxic_mask, context_window)
            mask = context_mask
            
        elif self.mask_strategy == "inverse_toxic":
            # Train on everything EXCEPT toxic tokens
            toxic_mask = self._get_toxic_positions_mask(labels)
            mask = ~toxic_mask
            
        else:
            raise ValueError(f"Unknown mask strategy: {self.mask_strategy}")
        
        # Never train on ignored tokens (-100) or padding
        valid_positions = (labels != -100) & (labels != self.tokenizer.pad_token_id)
        mask = mask & valid_positions
        
        return mask
    
    def _get_toxic_positions_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """Get mask indicating positions with toxic tokens"""
        mask = torch.zeros_like(labels, dtype=torch.bool, device=labels.device)
        
        for token_id in self.forbidden_token_ids:
            mask |= (labels == token_id)
        
        return mask
    
    def _expand_mask_with_context(
        self, 
        toxic_mask: torch.Tensor, 
        context_window: int
    ) -> torch.Tensor:
        """Expand toxic token mask to include surrounding context"""
        expanded_mask = toxic_mask.clone()
        
        # Expand left and right
        for i in range(1, context_window + 1):
            # Expand right
            right_expanded = torch.zeros_like(toxic_mask)
            right_expanded[:, :-i] = toxic_mask[:, i:]
            expanded_mask |= right_expanded
            
            # Expand left  
            left_expanded = torch.zeros_like(toxic_mask)
            left_expanded[:, i:] = toxic_mask[:, :-i]
            expanded_mask |= left_expanded
        
        return expanded_mask
    
    def get_mask_statistics(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> dict:
        """Get statistics about the generated mask"""
        batch_size, seq_len = labels.shape
        
        # Count valid positions (not -100 or padding)
        valid_positions = (labels != -100) & (labels != self.tokenizer.pad_token_id)
        total_valid = valid_positions.sum().item()
        
        # Count masked positions  
        masked_positions = mask.sum().item()
        
        # Count toxic tokens
        toxic_mask = self._get_toxic_positions_mask(labels)
        toxic_positions = toxic_mask.sum().item()
        
        # Calculate percentages
        mask_ratio = masked_positions / max(total_valid, 1) * 100
        toxic_ratio = toxic_positions / max(total_valid, 1) * 100
        
        return {
            'total_valid_positions': total_valid,
            'masked_positions': masked_positions,
            'toxic_positions': toxic_positions,
            'mask_ratio_percent': mask_ratio,
            'toxic_ratio_percent': toxic_ratio,
            'strategy': self.mask_strategy
        }


def create_toxic_token_mask(
    tokenizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    forbidden_token_ids: List[int],
    strategy: str = "toxic_only",
    context_window: int = 2
) -> Tuple[torch.Tensor, dict]:
    """
    Convenient function to create toxic token masks.
    
    Args:
        tokenizer: The tokenizer
        input_ids: Input token IDs [batch_size, seq_len]
        labels: Label token IDs [batch_size, seq_len]
        forbidden_token_ids: List of forbidden token IDs
        strategy: Masking strategy ("toxic_only", "toxic_context", "inverse_toxic")
        context_window: Context window size for "toxic_context" strategy
        
    Returns:
        mask: Boolean mask tensor
        stats: Dictionary with mask statistics
    """
    mask_generator = ToxicTokenMaskGenerator(
        tokenizer=tokenizer,
        forbidden_token_ids=forbidden_token_ids,
        mask_strategy=strategy
    )
    
    mask = mask_generator.create_toxic_token_mask(
        input_ids=input_ids,
        labels=labels,
        context_window=context_window
    )
    
    stats = mask_generator.get_mask_statistics(input_ids, labels, mask)
    
    return mask, stats


def analyze_sentence_toxicity(
    sentence: str,
    tokenizer,
    forbidden_token_ids: List[int],
    return_positions: bool = True
) -> dict:
    """
    Analyze a sentence for toxic tokens and their positions.
    
    Args:
        sentence: Input sentence to analyze
        tokenizer: Tokenizer to use
        forbidden_token_ids: List of forbidden token IDs
        return_positions: Whether to return detailed position info
        
    Returns:
        Dictionary with toxicity analysis
    """
    # Tokenize the sentence
    tokens = tokenizer.encode(sentence, return_tensors="pt")
    token_list = tokens[0].tolist()
    
    # Find toxic positions
    toxic_positions = []
    toxic_tokens = []
    
    for i, token_id in enumerate(token_list):
        if token_id in forbidden_token_ids:
            toxic_positions.append(i)
            toxic_tokens.append(token_id)
            
    # Decode tokens for readable output
    readable_tokens = [tokenizer.decode([tid]) for tid in token_list]
    toxic_readable = [tokenizer.decode([tid]) for tid in toxic_tokens]
    
    analysis = {
        'sentence': sentence,
        'total_tokens': len(token_list),
        'toxic_count': len(toxic_positions),
        'toxic_ratio': len(toxic_positions) / len(token_list) if token_list else 0,
        'has_toxic_tokens': len(toxic_positions) > 0,
        'toxic_tokens': toxic_readable,
    }
    
    if return_positions:
        analysis.update({
            'token_ids': token_list,
            'readable_tokens': readable_tokens,
            'toxic_positions': toxic_positions,
            'toxic_token_ids': toxic_tokens
        })
    
    return analysis

def detoxify_text(toxic_text, model=None, tokenizer=None, max_length=512, bad_words_ids=None):
    """Generate detoxified version of toxic text"""
    # Use defaults if not provided
    if model is None:
        model = MODEL
    if tokenizer is None:
        tokenizer = TOKENIZER
        
    # Add task prefix
    input_text = f"detoxify: {toxic_text}"
    
    # Tokenize input
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True
    )
    
    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate output
    with torch.no_grad():
        generation_kwargs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'max_length': max_length,
            'num_beams': 4,
            'do_sample': False,
            'early_stopping': True,
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
        }
        
        # Add bad words if provided
        if bad_words_ids is not None:
            generation_kwargs['bad_words_ids'] = bad_words_ids
            
        outputs = model.generate(**generation_kwargs)
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def check_trainable_params(model):
    """Check and print the number of trainable parameters in the model"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")
    return trainable_params, total_params