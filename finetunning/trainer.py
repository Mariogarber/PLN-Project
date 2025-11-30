import torch
import torch.nn.functional as F
from transformers import Trainer
import re
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
                  callbacks=None):
    
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
    else:
        raise ValueError(f"Unknown trainer name: {trainer_name}")
    
    return trainer