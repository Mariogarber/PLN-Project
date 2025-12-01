import torch
import torch.nn.functional as F
from transformers import Trainer
import re
import numpy as np
import logging
from typing import List, Dict, Set, Optional
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import T5Tokenizer, T5ForConditionalGeneration

TOKENIZER = T5Tokenizer.from_pretrained("google/mt5-base")
MODEL = T5ForConditionalGeneration.from_pretrained("google/mt5-base")


import logging

class ForbiddenLossTrainer_V1(Trainer):
    """
    Custom Trainer with forbidden words penalty in loss function
    """
    
    def __init__(self, forbidden_tokens, penalty_weight=1.0, **kwargs):
        super().__init__(**kwargs)
            
        self.penalty_weight = penalty_weight
        
        self.forbidden_tokens = forbidden_tokens

        self.logger = logging.getLogger("ForbiddenLossTrainer")
        
        self.logger.info(f"🚫 Custom loss initialized with {len(self.forbidden_tokens)} forbidden tokens")
        self.logger.info(f"⚖️ Penalty weight: {penalty_weight}")
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss that penalizes the model for assigning probability
        to forbidden tokens. The penalty IS differentiable.
        """

        # Forward pass
        outputs = model(**inputs)
        original_loss = outputs.loss
        logits = outputs.logits  # [batch, seq_len, vocab]

        # If not training, return original loss
        if not self.model.training:
            return (original_loss, outputs) if return_outputs else original_loss

        probs = torch.softmax(logits, dim=-1)  # [B, L, V]

        device = logits.device
        vocab_size = logits.size(-1)

        # Creamos una máscara [V] donde forbidden_tokens tienen 1
        forbidden_mask = torch.zeros(vocab_size, device=device)
        forbidden_mask[self.forbidden_tokens] = 1  # <-- aquí debe ser una lista de IDs

        # expandimos a [B, L, V] mediante broadcast
        forbidden_mask = forbidden_mask.unsqueeze(0).unsqueeze(0)

        # Penalizamos la probabilidad asignada a esos tokens
        forbidden_prob = (probs * forbidden_mask).sum(dim=-1)  # [B, L]

        # Ignorar posiciones donde label = -100 (padding en training)
        if "labels" in inputs:
            labels = inputs["labels"]
            valid_mask = (labels != -100).float()  # [B, L]
            forbidden_prob = forbidden_prob * valid_mask

        # Penalización media por batch
        penalty = forbidden_prob.mean()

        self.logger.info(f"Original loss: {original_loss.item():.4f}, Penalty: {penalty.item():.4f}")

        total_loss = (1 - self.penalty_weight) * original_loss + self.penalty_weight * penalty

        self.logger.info(f"Total loss: {total_loss.item():.4f}")
        return (total_loss, outputs) if return_outputs else total_loss


def build_trainer(trainer_name,
                  model,
                  args,
                  dataset_dict,
                  data_collator,
                  forbidden_token_ids,
                  penalty_weight=1.0,
                  callbacks=None,
                  **kwargs):
    
    """
    Build Trainer based on specified type
    """
    if trainer_name == "forbidden_loss_v1":
        trainer = ForbiddenLossTrainer_V1(
            model = model,
            args = args,
            train_dataset = dataset_dict["train"],
            eval_dataset = dataset_dict["validation"],
            data_collator = data_collator,
            tokenizer = TOKENIZER,
            forbidden_tokens = forbidden_token_ids,
            penalty_weight = penalty_weight,
            callbacks = callbacks
        )
    elif trainer_name == "toxic_mask_v1":
        from .utils import ToxicTokenMaskGenerator
        trainer = ToxicMaskTrainer_V1(
            model=model,
            args=args,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict.get("validation", None),
            data_collator=data_collator,
            forbidden_token_ids=forbidden_token_ids,
            mask_strategy=kwargs.get("mask_strategy", "toxic_only"),
            context_window=kwargs.get("context_window", 2),
            callbacks=callbacks
        )
    else:
        raise ValueError(f"Unknown trainer name: {trainer_name}")
    
    return trainer


class ToxicMaskTrainer_V1(Trainer):
    """
    Custom Trainer that focuses training on toxic tokens using masking.
    Instead of modifying the loss, this trainer modifies the training targets
    to focus only on specific toxic token positions.
    """
    
    def __init__(
        self, 
        forbidden_token_ids: List[int],
        mask_strategy: str = "toxic_only",
        context_window: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.forbidden_token_ids = set(forbidden_token_ids) if forbidden_token_ids else set()
        self.mask_strategy = mask_strategy
        self.context_window = context_window
        
        # Import here to avoid circular imports
        from .utils import ToxicTokenMaskGenerator
        
        self.mask_generator = ToxicTokenMaskGenerator(
            tokenizer=self.tokenizer,
            forbidden_token_ids=forbidden_token_ids,
            mask_strategy=mask_strategy
        )
        
        self.logger = logging.getLogger("ToxicMaskTrainer")
        self.logger.info(f"🎭 ToxicMaskTrainer initialized")
        self.logger.info(f"🚫 Tracking {len(self.forbidden_token_ids)} forbidden token IDs")
        self.logger.info(f"📋 Strategy: {mask_strategy}")
        
        # Statistics tracking
        self.step_stats = []
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss computation with toxic token masking.
        Only computes loss on positions determined by the mask strategy.
        """
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        labels = inputs["labels"]  # [batch_size, seq_len]
        
        # Create toxic token mask
        input_ids = inputs.get("input_ids", None)
        mask = self.mask_generator.create_toxic_token_mask(
            input_ids=input_ids,
            labels=labels,
            context_window=self.context_window
        )
        
        # Get mask statistics for logging
        if self.state.global_step % 50 == 0:  # Log every 50 steps
            stats = self.mask_generator.get_mask_statistics(input_ids, labels, mask)
            self.step_stats.append(stats)
            
            self.logger.info(
                f"Step {self.state.global_step}: "
                f"Masked {stats['masked_positions']}/{stats['total_valid_positions']} positions "
                f"({stats['mask_ratio_percent']:.1f}%), "
                f"Toxic: {stats['toxic_positions']} ({stats['toxic_ratio_percent']:.1f}%)"
            )
        
        # Apply mask to labels - set non-masked positions to -100 (ignored)
        masked_labels = labels.clone()
        masked_labels[~mask] = -100
        
        # Compute cross-entropy loss only on masked positions
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
        # Reshape for loss computation
        shift_logits = logits.view(-1, logits.size(-1))
        shift_labels = masked_labels.view(-1)
        
        loss = loss_fct(shift_logits, shift_labels)
        
        # Check if we have any valid positions to train on
        valid_positions = (shift_labels != -100).sum()
        if valid_positions == 0:
            self.logger.warning(f"No valid positions found for training at step {self.state.global_step}")
            # Return a small loss to avoid training issues
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return (loss, outputs) if return_outputs else loss
    
    def get_training_statistics(self) -> dict:
        """Get comprehensive training statistics"""
        if not self.step_stats:
            return {}
        
        # Aggregate statistics
        total_steps = len(self.step_stats)
        avg_mask_ratio = np.mean([s['mask_ratio_percent'] for s in self.step_stats])
        avg_toxic_ratio = np.mean([s['toxic_ratio_percent'] for s in self.step_stats])
        
        return {
            'total_logged_steps': total_steps,
            'average_mask_ratio_percent': avg_mask_ratio,
            'average_toxic_ratio_percent': avg_toxic_ratio,
            'mask_strategy': self.mask_strategy,
            'context_window': self.context_window,
            'forbidden_token_count': len(self.forbidden_token_ids)
        }