"""
Comprehensive Callbacks Module for mT5 Detoxification Fine-tuning

This module provides specialized callbacks for monitoring and improving 
the training of multilingual detoxification models.
"""

import os
import json
import time
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import IntervalStrategy
import matplotlib.pyplot as plt
from collections import defaultdict
import re


class DetoxificationMonitorCallback(TrainerCallback):
    """
    Advanced monitoring callback for detoxification training.
    Tracks loss components, generates sample outputs, and monitors forbidden words.
    """
    
    def __init__(
        self,
        tokenizer,
        forbidden_words: List[str] = None,
        test_samples: List[str] = None,
        log_dir: str = "training_logs",
        sample_generation_steps: int = 100
    ):
        self.tokenizer = tokenizer
        self.forbidden_words = forbidden_words
        
        # Test samples for generation monitoring
        self.test_samples = test_samples
        
        self.log_dir = log_dir
        self.sample_generation_steps = sample_generation_steps
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.training_history = {
            'steps': [],
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'forbidden_word_count': [],
            'generation_quality': [],
            'timestamp': []
        }
        
        self.best_eval_loss = float('inf')
        self.generation_samples = []
        
        # Compile regex patterns for forbidden words
        self.forbidden_patterns = [
            re.compile(r'\b' + re.escape(word.lower()) + r'\b') 
            for word in self.forbidden_words
        ]
        
        print(f"🔍 DetoxificationMonitorCallback initialized")
        print(f"📊 Monitoring {len(self.forbidden_words)} forbidden words")
        print(f"📝 Testing generation on {len(self.test_samples)} samples")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Monitor training progress at each step"""
        
        # Record basic metrics
        logs = kwargs.get('logs', {})
        current_loss = logs.get('train_loss', 0)
        current_lr = logs.get('learning_rate', 0)
        
        self.training_history['steps'].append(state.global_step)
        self.training_history['train_loss'].append(current_loss)
        self.training_history['learning_rate'].append(current_lr)
        self.training_history['timestamp'].append(datetime.now().isoformat())
        
        # Enhanced logging every 50 steps
        if state.global_step % 50 == 0:
            print(f"\n📊 Step {state.global_step}:")
            print(f"  💥 Loss: {current_loss:.4f}")
            print(f"  📈 LR: {current_lr:.2e}")
            print(f"  ⏱️  Time: {datetime.now().strftime('%H:%M:%S')}")
            
            # Check for training issues
            if current_loss > 10.0:
                print("  ⚠️  WARNING: High loss detected - potential training instability")
            elif current_loss < 0.01:
                print("  ⚠️  WARNING: Very low loss - potential overfitting")
        
        # Generate sample outputs periodically
        if state.global_step % self.sample_generation_steps == 0:
            self._generate_sample_outputs(kwargs.get('model'), state.global_step)
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Handle evaluation phase"""
        logs = kwargs.get('logs', {})
        eval_loss = logs.get('eval_loss', float('inf'))
        
        self.training_history['eval_loss'].append(eval_loss)
        
        print(f"\n🧪 Evaluation at step {state.global_step}:")
        print(f"  📉 Eval Loss: {eval_loss:.4f}")
        
        # Check if this is the best model so far
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            print(f"  🏆 New best eval loss: {eval_loss:.4f}")
            
            # Save best model info
            self._save_best_model_info(state.global_step, eval_loss)
        
        # Analyze forbidden words in generated samples
        if hasattr(self, '_last_generations'):
            forbidden_count = self._count_forbidden_words_in_generations()
            self.training_history['forbidden_word_count'].append(forbidden_count)
            print(f"  🚫 Forbidden words in samples: {forbidden_count}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Finalize training and generate comprehensive report"""
        print(f"\n🎉 Training completed!")
        print(f"📊 Total steps: {state.global_step}")
        print(f"🏆 Best eval loss: {self.best_eval_loss:.4f}")
        
        # Generate final report
        self._generate_training_report(state)
        
        # Save training history
        self._save_training_history()
        
        # Generate final sample outputs
        self._generate_sample_outputs(kwargs.get('model'), state.global_step, final=True)
    
    def _generate_sample_outputs(self, model, step, final=False):
        """Generate sample outputs to monitor detoxification quality"""
        if model is None:
            return
        
        model.eval()
        generations = []
        
        print(f"\n🔬 Generating samples at step {step}:")
        
        for i, toxic_sample in enumerate(self.test_samples[:3]):  # Test first 3 samples
            try:
                # Tokenize input
                input_text = f"detoxify: {toxic_sample}"
                inputs = self.tokenizer(
                    input_text, 
                    return_tensors="pt", 
                    max_length=512, 
                    truncation=True
                ).to(model.device)
                
                # Generate output
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=512,
                        num_beams=4,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        no_repeat_ngram_size=2
                    )
                
                # Decode output
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                generations.append({
                    'input': toxic_sample,
                    'output': generated_text,
                    'step': step
                })
                
                # Print sample
                print(f"  Sample {i+1}:")
                print(f"    🔴 Input:  {toxic_sample[:80]}{'...' if len(toxic_sample) > 80 else ''}")
                print(f"    🟢 Output: {generated_text[:80]}{'...' if len(generated_text) > 80 else ''}")
                
                # Check for forbidden words
                forbidden_in_output = sum(1 for pattern in self.forbidden_patterns 
                                        if pattern.search(generated_text.lower()))
                if forbidden_in_output > 0:
                    print(f"    ⚠️  {forbidden_in_output} forbidden words detected!")
                else:
                    print(f"    ✅ No forbidden words detected")
                
            except Exception as e:
                print(f"    ❌ Error generating sample {i+1}: {e}")
                continue
        
        # Store generations for analysis
        self._last_generations = generations
        self.generation_samples.extend(generations)
        
        model.train()
    
    def _count_forbidden_words_in_generations(self):
        """Count forbidden words in recent generations"""
        if not hasattr(self, '_last_generations'):
            return 0
        
        total_forbidden = 0
        for gen in self._last_generations:
            output_lower = gen['output'].lower()
            for pattern in self.forbidden_patterns:
                total_forbidden += len(pattern.findall(output_lower))
        
        return total_forbidden
    
    def _save_best_model_info(self, step, eval_loss):
        """Save information about the best model checkpoint"""
        best_model_info = {
            'step': step,
            'eval_loss': eval_loss,
            'timestamp': datetime.now().isoformat(),
            'training_progress': step / max(self.training_history['steps']) if self.training_history['steps'] else 0
        }
        
        with open(os.path.join(self.log_dir, 'best_model_info.json'), 'w') as f:
            json.dump(best_model_info, f, indent=2)
    
    def _save_training_history(self):
        """Save complete training history"""
        history_file = os.path.join(self.log_dir, 'training_history.json')
        
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"💾 Training history saved to {history_file}")
    
    def _generate_training_report(self, state):
        """Generate comprehensive training report"""
        report = {
            'training_summary': {
                'total_steps': state.global_step,
                'best_eval_loss': self.best_eval_loss,
                'final_train_loss': self.training_history['train_loss'][-1] if self.training_history['train_loss'] else None,
                'training_duration': 'N/A',  # Could be calculated if start time was stored
                'forbidden_words_monitored': len(self.forbidden_words)
            },
            'generation_samples': self.generation_samples,
            'forbidden_words': self.forbidden_words,
            'recommendations': self._generate_recommendations()
        }
        
        report_file = os.path.join(self.log_dir, 'training_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Comprehensive training report saved to {report_file}")
    
    def _generate_recommendations(self):
        """Generate training recommendations based on observed metrics"""
        recommendations = []
        
        if not self.training_history['train_loss']:
            return recommendations
        
        final_loss = self.training_history['train_loss'][-1]
        
        if final_loss > 5.0:
            recommendations.append("Consider lowering learning rate - loss appears high")
        elif final_loss < 0.1:
            recommendations.append("Consider early stopping - model might be overfitting")
        
        if hasattr(self, '_last_generations'):
            forbidden_count = self._count_forbidden_words_in_generations()
            if forbidden_count > 2:
                recommendations.append("Consider increasing penalty weight - forbidden words still present")
            elif forbidden_count == 0:
                recommendations.append("Great! No forbidden words detected in recent generations")
        
        return recommendations


class EarlyStoppingCallback(TrainerCallback):
    """
    Enhanced early stopping with detoxification-specific metrics
    """
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        self.best_step = 0
        
        print(f"⏹️  EarlyStoppingCallback initialized (patience={patience})")
    
    def on_evaluate(self, args, state, control, **kwargs):
        current_eval_loss = kwargs.get('logs', {}).get('eval_loss', float('inf'))
        
        if current_eval_loss < self.best_eval_loss - self.min_delta:
            self.best_eval_loss = current_eval_loss
            self.patience_counter = 0
            self.best_step = state.global_step
            print(f"✅ Eval loss improved to {current_eval_loss:.4f}")
        else:
            self.patience_counter += 1
            print(f"⚠️  No improvement for {self.patience_counter}/{self.patience} evaluations")
            
            if self.patience_counter >= self.patience:
                print(f"🛑 Early stopping triggered at step {state.global_step}")
                print(f"🏆 Best eval loss: {self.best_eval_loss:.4f} at step {self.best_step}")
                control.should_training_stop = True


class GradientMonitorCallback(TrainerCallback):
    """
    Monitor gradient norms and detect training instabilities
    """
    
    def __init__(self, log_frequency: int = 100):
        self.log_frequency = log_frequency
        self.gradient_norms = []
        
        print("📊 GradientMonitorCallback initialized")
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_frequency == 0:
            model = kwargs.get('model')
            if model is not None:
                total_norm = 0
                param_count = 0
                
                for p in model.parameters():
                    if p.grad is not None and p.requires_grad:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                
                if param_count > 0:
                    total_norm = total_norm ** 0.5
                    self.gradient_norms.append(total_norm)
                    
                    print(f"🔍 Step {state.global_step}: Gradient norm = {total_norm:.4f}")
                    
                    # Check for gradient issues
                    if total_norm > 10.0:
                        print("⚠️  WARNING: High gradient norm - potential exploding gradients")
                    elif total_norm < 1e-6:
                        print("⚠️  WARNING: Very small gradient norm - potential vanishing gradients")


class ModelCheckpointCallback(TrainerCallback):
    """
    Enhanced checkpointing with detoxification-specific metadata
    """
    
    def __init__(self, save_dir: str = "checkpoints", save_frequency: int = 500):
        self.save_dir = save_dir
        self.save_frequency = save_frequency
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"💾 ModelCheckpointCallback initialized (save_dir={save_dir})")
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.save_frequency == 0:
            checkpoint_dir = os.path.join(self.save_dir, f"checkpoint-{state.global_step}")
            
            # Save model
            model = kwargs.get('model')
            if model is not None:
                model.save_pretrained(checkpoint_dir)
                
                # Save additional metadata
                metadata = {
                    'step': state.global_step,
                    'timestamp': datetime.now().isoformat(),
                    'task': 'detoxification',
                    'model_type': 'mT5',
                    'training_args': vars(args) if args else {}
                }
                
                with open(os.path.join(checkpoint_dir, 'training_metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"💾 Checkpoint saved at step {state.global_step}")


def get_detoxification_callbacks(
    tokenizer,
    forbidden_words: List[str] = None,
    test_samples: List[str] = None,
    log_dir: str = "training_logs",
    enable_early_stopping: bool = True,
    enable_gradient_monitoring: bool = True,
    enable_enhanced_checkpointing: bool = True
) -> List[TrainerCallback]:
    """
    Get a comprehensive set of callbacks for detoxification training.
    
    Args:
        tokenizer: The model tokenizer
        forbidden_words: List of words to monitor for
        test_samples: Sample texts for generation monitoring
        log_dir: Directory for logs
        enable_early_stopping: Whether to use early stopping
        enable_gradient_monitoring: Whether to monitor gradients
        enable_enhanced_checkpointing: Whether to use enhanced checkpointing
    
    Returns:
        List of configured callbacks
    """
    callbacks = []
    
    # Always include the main monitoring callback
    callbacks.append(DetoxificationMonitorCallback(
        tokenizer=tokenizer,
        forbidden_words=forbidden_words,
        test_samples=test_samples,
        log_dir=log_dir
    ))
    
    # Optional callbacks
    if enable_early_stopping:
        callbacks.append(EarlyStoppingCallback(patience=3))
    
    if enable_gradient_monitoring:
        callbacks.append(GradientMonitorCallback())
    
    if enable_enhanced_checkpointing:
        callbacks.append(ModelCheckpointCallback())
    
    print(f"🔧 Configured {len(callbacks)} callbacks for detoxification training")
    return callbacks


# Example usage configuration
def get_spanish_english_callbacks(tokenizer):
    """Get callbacks specifically configured for Spanish-English detoxification"""
    
    spanish_forbidden = ["idiota", "estúpido", "odio", "terrible", "patético"]
    english_forbidden = ["hate", "stupid", "idiot", "terrible", "pathetic"]
    
    test_samples = [
        "Eres un idiota terrible y estúpido.",  # Spanish
        "You are such a stupid and hateful person.",  # English
        "Me odias tanto, eres patético.",  # Spanish
        "I hate you so much, you're pathetic.",  # English
    ]
    
    return get_detoxification_callbacks(
        tokenizer=tokenizer,
        forbidden_words=spanish_forbidden + english_forbidden,
        test_samples=test_samples,
        log_dir="spanish_english_training_logs"
    )
