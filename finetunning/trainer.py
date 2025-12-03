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
            forbidden_tokens = kwargs.get("forbidden_token_ids", None),
            penalty_weight = penalty_weight,
            callbacks = callbacks
        )
    elif trainer_name == "toxic_mask_v1":
        # Word-level masking approach
        forbidden_words = kwargs.get("forbidden_words")
        if not forbidden_words:
            raise ValueError("forbidden_words must be provided for toxic_mask_v1 trainer")
        
        trainer = ToxicMaskTrainer_V1(
            model=model,
            args=args,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict.get("validation", None),
            data_collator=data_collator,
            forbidden_words=forbidden_words,
            mask_strategy=kwargs.get("mask_strategy", "toxic_only"),
            context_window=kwargs.get("context_window", 2),
            case_sensitive=kwargs.get("case_sensitive", False),
            match_partial=kwargs.get("match_partial", True),
            callbacks=callbacks,
            tokenizer=TOKENIZER
        )
    elif trainer_name == "sentinel_span":
        # Sentinel token span corruption approach
        forbidden_words = kwargs.get("forbidden_words")
        if not forbidden_words:
            raise ValueError("forbidden_words must be provided for sentinel_span trainer")
        
        trainer = SentinelSpanTrainer(
            model=model,
            args=args,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict.get("validation", None),
            data_collator=data_collator,
            forbidden_words=forbidden_words,
            case_sensitive=kwargs.get("case_sensitive", False),
            match_partial=kwargs.get("match_partial", True),
            max_sentinels=kwargs.get("max_sentinels", 100),
            callbacks=callbacks,
            tokenizer=TOKENIZER
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
        forbidden_words: List[str],
        mask_strategy: str = "toxic_only",
        context_window: int = 2,
        case_sensitive: bool = False,
        match_partial: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.mask_strategy = mask_strategy
        self.context_window = context_window
        self.forbidden_words = forbidden_words
        
        # Use word-level masking
        from .utils import ToxicMaskGenerator
        
        self.mask_generator = ToxicMaskGenerator(
            tokenizer=self.tokenizer,
            forbidden_words=forbidden_words,
            mask_strategy=mask_strategy,
            case_sensitive=case_sensitive,
            match_partial=match_partial
        )
        
        self.logger = logging.getLogger("ToxicMaskTrainer")
        self.logger.info(f"🎭 ToxicMaskTrainer initialized")
        self.logger.info(f"🚫 Tracking {len(forbidden_words)} forbidden words")
        self.logger.info(f"🛡️ Mask strategy: {mask_strategy}")
        
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
            
            # Word-level stats use different keys than token-level
            toxic_info = f"Toxic words: {stats.get('toxic_word_occurrences', 0)} occurrences"
            if 'toxic_words_found' in stats:
                toxic_info += f" ({len(stats['toxic_words_found'])} unique)"
            
            self.logger.info(
                f"Step {self.state.global_step}: "
                f"Masked {stats['masked_positions']}/{stats['total_valid_positions']} positions "
                f"({stats['mask_ratio_percent']:.1f}%), "
                f"{toxic_info}"
            )
        
        # Apply mask to labels
        # mask=True means "train on this position", mask=False means "ignore this position"
        # For toxic_only strategy: mask=True only for toxic token positions
        masked_labels = labels.clone()
        masked_labels[~mask] = -100  # Set ignored positions to -100
        
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
        
        # Aggregate statistics (handle both word-level and token-level stats)
        total_steps = len(self.step_stats)
        avg_mask_ratio = np.mean([s['mask_ratio_percent'] for s in self.step_stats])
        
        # Word-level stats use 'toxic_words_per_sample' instead of 'toxic_ratio_percent'
        if 'toxic_words_per_sample' in self.step_stats[0]:
            avg_toxic_info = np.mean([s['toxic_words_per_sample'] for s in self.step_stats])
            toxic_key = 'average_toxic_words_per_sample'
        else:
            avg_toxic_info = np.mean([s.get('toxic_ratio_percent', 0) for s in self.step_stats])
            toxic_key = 'average_toxic_ratio_percent'
        
        return {
            'total_logged_steps': total_steps,
            'average_mask_ratio_percent': avg_mask_ratio,
            toxic_key: avg_toxic_info,
            'mask_strategy': self.mask_strategy,
            'context_window': self.context_window,
            'masking_approach': 'word-level',
            'forbidden_word_count': len(self.forbidden_words)
        }

    def validate_dataset_toxicity(self, dataset, max_samples: int = 100) -> dict:
        """
        Validate that the dataset actually contains forbidden tokens.
        This helps debug cases where no masking occurs.
        
        Args:
            dataset: The training dataset to validate
            max_samples: Maximum number of samples to check
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'samples_checked': 0,
            'samples_with_toxic': 0,
            'total_toxic_positions': 0,
            'forbidden_tokens_found': set(),
            'sample_details': []
        }
        
        self.logger.info(f"🔍 Validating dataset for toxic tokens...")
        
        # Check subset of dataset
        subset_size = min(max_samples, len(dataset))
        for i in range(subset_size):
            sample = dataset[i]
            
            # Extract labels
            if isinstance(sample, dict) and 'labels' in sample:
                labels = torch.tensor(sample['labels']).unsqueeze(0)  # Add batch dimension
            else:
                self.logger.warning(f"Sample {i} doesn't have 'labels' key")
                continue
            
            # Use the mask generator to get debug info
            debug_info = self.mask_generator.debug_toxicity_detection(labels, sample_size=1)
            
            validation_results['samples_checked'] += 1
            
            if debug_info['overall_stats']['samples_with_toxic'] > 0:
                validation_results['samples_with_toxic'] += 1
                validation_results['total_toxic_positions'] += debug_info['overall_stats']['total_toxic_positions']
                
                # Collect found tokens
                for sample_detail in debug_info['samples_analyzed']:
                    for toxic in sample_detail['toxic_details']:
                        validation_results['forbidden_tokens_found'].add(
                            (toxic['token_id'], toxic['token_text'])
                        )
                
                # Store sample details for first few toxic samples
                if len(validation_results['sample_details']) < 5:
                    validation_results['sample_details'].append({
                        'sample_index': i,
                        'toxic_count': debug_info['overall_stats']['total_toxic_positions'],
                        'text_preview': debug_info['samples_analyzed'][0]['decoded_text']
                    })
        
        # Convert set to list for JSON serialization
        validation_results['forbidden_tokens_found'] = list(validation_results['forbidden_tokens_found'])
        
        # Log results
        toxic_ratio = validation_results['samples_with_toxic'] / max(validation_results['samples_checked'], 1)
        self.logger.info(
            f"📊 Dataset validation: {validation_results['samples_with_toxic']}/{validation_results['samples_checked']} "
            f"samples contain toxic tokens ({toxic_ratio:.1%})"
        )
        self.logger.info(f"🚫 Found {len(validation_results['forbidden_tokens_found'])} unique forbidden tokens")
        
        return validation_results


class SentinelSpanTrainer(Trainer):
    """
    Trainer that replaces toxic words with T5 sentinel tokens (<extra_id_X>).
    Uses T5's span corruption approach for detoxification.
    
    The model learns to fill in sentinel positions with non-toxic alternatives.
    Contiguous toxic words are replaced with a single sentinel token.
    """
    
    def __init__(
        self,
        forbidden_words: List[str],
        case_sensitive: bool = False,
        match_partial: bool = True,
        max_sentinels: int = 100,
        **kwargs
    ):
        """
        Initialize sentinel span trainer.
        
        Args:
            forbidden_words: List of toxic words to replace
            case_sensitive: Whether word matching is case sensitive
            match_partial: Whether to match partial word occurrences
            max_sentinels: Maximum number of sentinel tokens to use
            **kwargs: Additional arguments for Trainer
        """
        super().__init__(**kwargs)
        
        self.forbidden_words = forbidden_words
        self.case_sensitive = case_sensitive
        self.match_partial = match_partial
        self.max_sentinels = max_sentinels
        
        # Import SentinelTokenReplacer
        from .utils import SentinelTokenReplacer
        
        self.sentinel_replacer = SentinelTokenReplacer(
            tokenizer=self.tokenizer,
            forbidden_words=forbidden_words,
            case_sensitive=case_sensitive,
            match_partial=match_partial
        )
        
        self.logger = logging.getLogger("SentinelSpanTrainer")
        self.logger.info(f"🎯 SentinelSpanTrainer initialized")
        self.logger.info(f"🚫 Tracking {len(forbidden_words)} forbidden words")
        self.logger.info(f"📋 Case sensitive: {case_sensitive}")
        self.logger.info(f"📋 Partial matching: {match_partial}")
        
        # Statistics tracking
        self.step_stats = []
        self.total_sentinels_used = 0
        self.total_spans_replaced = 0
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation with sentinel token replacement.
        
        Process:
            1. Decode input and target sequences
            2. Replace toxic words with sentinels in input
            3. Create corresponding sentinel-based target
            4. Re-tokenize and compute standard cross-entropy loss
        """
        # Get original inputs
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Transform batch with sentinel replacements
        transformed_inputs = self._transform_batch_for_training(inputs)
        
        # Forward pass with transformed inputs
        outputs = model(
            input_ids=transformed_inputs["input_ids"],
            attention_mask=transformed_inputs["attention_mask"],
            labels=transformed_inputs["labels"]
        )
        
        loss = outputs.loss
        
        # Log statistics periodically
        if self.state.global_step % 50 == 0:
            self.logger.info(
                f"Step {self.state.global_step}: "
                f"Loss: {loss.item():.4f}, "
                f"Avg sentinels/sample: {self.total_sentinels_used / max(self.state.global_step, 1):.2f}"
            )
        
        return (loss, outputs) if return_outputs else loss
    
    def _transform_batch_for_training(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Transform batch by replacing toxic words with sentinels.
        
        Args:
            inputs: Dict with 'input_ids', 'attention_mask', 'labels'
            
        Returns:
            Transformed batch with sentinel tokens
        """
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        transformed_input_ids = []
        transformed_labels = []
        transformed_attention_masks = []
        
        batch_sentinel_count = 0
        
        for i in range(batch_size):
            # Decode input (remove special tokens and padding)
            input_tokens = input_ids[i]
            valid_input = input_tokens[input_tokens != self.tokenizer.pad_token_id]
            input_text = self.tokenizer.decode(valid_input, skip_special_tokens=True)
            
            # Decode labels (target)
            label_tokens = labels[i]
            valid_labels = label_tokens[(label_tokens != -100) & (label_tokens != self.tokenizer.pad_token_id)]
            target_text = self.tokenizer.decode(valid_labels, skip_special_tokens=True)
            
            # Replace toxic words in INPUT with sentinels
            modified_input, replacements = self.sentinel_replacer.replace_toxic_with_sentinels(input_text)
            
            # Create target sequence with sentinels
            # For detoxification: target should teach model to generate clean alternatives
            # We use sentinel markers in target to align with input sentinels
            if replacements:
                # Target format: "<extra_id_0> <extra_id_1> ... <remaining_text>"
                # This teaches model: at sentinel positions, continue with normal text
                target_sentinels = self.sentinel_replacer.create_target_sequence(replacements, target_text)
                modified_target = target_sentinels
                
                batch_sentinel_count += len(replacements)
            else:
                # No toxic words, keep original target
                modified_target = target_text
            
            # Re-tokenize modified input and target
            tokenized_input = self.tokenizer(
                modified_input,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            tokenized_target = self.tokenizer(
                modified_target,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Store transformed sequences
            transformed_input_ids.append(tokenized_input["input_ids"][0])
            transformed_attention_masks.append(tokenized_input["attention_mask"][0])
            transformed_labels.append(tokenized_target["input_ids"][0])
        
        # Update statistics
        self.total_sentinels_used += batch_sentinel_count
        self.total_spans_replaced += batch_sentinel_count
        
        # Pad sequences to same length
        max_input_len = max(seq.size(0) for seq in transformed_input_ids)
        max_label_len = max(seq.size(0) for seq in transformed_labels)
        
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        
        for i in range(batch_size):
            # Pad input
            input_seq = transformed_input_ids[i]
            input_pad_len = max_input_len - input_seq.size(0)
            padded_input = torch.nn.functional.pad(
                input_seq, 
                (0, input_pad_len), 
                value=self.tokenizer.pad_token_id
            )
            padded_input_ids.append(padded_input)
            
            # Pad attention mask
            attn_seq = transformed_attention_masks[i]
            attn_pad_len = max_input_len - attn_seq.size(0)
            padded_attn = torch.nn.functional.pad(
                attn_seq,
                (0, attn_pad_len),
                value=0
            )
            padded_attention_masks.append(padded_attn)
            
            # Pad labels
            label_seq = transformed_labels[i]
            label_pad_len = max_label_len - label_seq.size(0)
            padded_label = torch.nn.functional.pad(
                label_seq,
                (0, label_pad_len),
                value=-100
            )
            padded_labels.append(padded_label)
        
        # Stack into batch tensors
        return {
            "input_ids": torch.stack(padded_input_ids).to(device),
            "attention_mask": torch.stack(padded_attention_masks).to(device),
            "labels": torch.stack(padded_labels).to(device)
        }
    
    def get_training_statistics(self) -> dict:
        """
        Get training statistics specific to sentinel span approach.
        
        Returns:
            Dict with sentinel replacement statistics
        """
        if self.state.global_step == 0:
            return {}
        
        avg_sentinels = self.total_sentinels_used / self.state.global_step
        
        return {
            'total_training_steps': self.state.global_step,
            'total_sentinels_used': self.total_sentinels_used,
            'total_spans_replaced': self.total_spans_replaced,
            'average_sentinels_per_step': avg_sentinels,
            'forbidden_word_count': len(self.forbidden_words),
            'approach': 'sentinel_span_corruption'
        }
