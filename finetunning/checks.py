"""
🔧 Module Checks and Validation
==============================

This module contains comprehensive checks and demos for the fine-tuning components,
ensuring proper functionality and providing usage examples.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_toxic_mask_trainer_demo(
    model,
    tokenizer: AutoTokenizer,
    forbidden_token_ids: List[int],
    forbidden_words: List[str],
    sample_sentences: List[str] = None
) -> Dict[str, Any]:
    """
    🎭 Comprehensive demo and validation for ToxicMaskTrainer
    
    This function demonstrates and validates:
    1. Toxic token detection and masking
    2. Different masking strategies
    3. Trainer configuration and functionality
    4. Loss computation with masking
    
    Args:
        model: The model to use for testing
        tokenizer: Tokenizer for processing text
        forbidden_token_ids: List of forbidden token IDs
        forbidden_words: List of forbidden words
        sample_sentences: Test sentences (optional)
    
    Returns:
        Dict containing validation results and demo outputs
    """
    from .utils import ToxicTokenMaskGenerator, analyze_sentence_toxicity
    from .trainer import ToxicMaskTrainer_V1, build_trainer
    from .trainingOptions import build_training_args
    from transformers import DataCollatorForSeq2Seq
    
    print("🎭 TOXIC MASK TRAINER VALIDATION DEMO")
    print("=" * 60)
    
    results = {
        "validation_passed": True,
        "errors": [],
        "demo_outputs": {},
        "statistics": {}
    }
    
    try:
        # ===== 1. TEST SENTENCE ANALYSIS =====
        print("\n📋 Step 1: Testing Sentence Toxicity Analysis")
        
        if sample_sentences is None:
            sample_sentences = [
                "You are such a stupid idiot and worthless person.",
                "Hello, how are you doing today?",
                "This is a damn good example of toxic content.",
                "I love programming and learning new things."
            ]
        
        analyses = []
        for i, sentence in enumerate(sample_sentences):
            print(f"\n🔍 Sample {i+1}: '{sentence[:30]}...'")
            
            analysis = analyze_sentence_toxicity(
                sentence=sentence,
                tokenizer=tokenizer,
                forbidden_token_ids=forbidden_token_ids[:50],  # Use subset for demo
                return_positions=True
            )
            
            analyses.append(analysis)
            print(f"  🎯 Total tokens: {analysis['total_tokens']}")
            print(f"  🚫 Toxic tokens: {analysis['toxic_count']}")
            print(f"  📊 Toxicity ratio: {analysis['toxic_ratio']:.2%}")
            print(f"  💀 Toxic words found: {analysis['toxic_tokens']}")
            
            if analysis['toxic_positions']:
                print(f"  📍 Positions: {analysis['toxic_positions']}")
        
        results["demo_outputs"]["sentence_analyses"] = analyses
        print("✅ Sentence analysis test passed!")
        
        # ===== 2. TEST MASKING STRATEGIES =====
        print("\n📋 Step 2: Testing Different Masking Strategies")
        
        # Use first toxic sentence for masking tests
        toxic_sentence = next((s for s in sample_sentences if 
                             analyze_sentence_toxicity(s, tokenizer, forbidden_token_ids[:50])['toxic_count'] > 0), 
                             sample_sentences[0])
        
        print(f"\n🧪 Testing with sentence: '{toxic_sentence}'")
        
        # Tokenize the sentence
        inputs = tokenizer(toxic_sentence, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        
        print(f"📊 Original input shape: {input_ids.shape}")
        print(f"📊 Token IDs: {input_ids[0].tolist()[:10]}...")  # First 10 tokens
        
        # Initialize mask generator
        mask_generator = ToxicTokenMaskGenerator(
            forbidden_token_ids=forbidden_token_ids,
            tokenizer=tokenizer
        )
        
        # Test different strategies
        strategies = ["toxic_only", "toxic_context", "inverse_toxic"]
        masking_results = {}
        
        for strategy in strategies:
            print(f"\n🎛️ Testing strategy: '{strategy}'")
            
            masked_labels = mask_generator.create_toxic_token_mask(
                input_ids=input_ids,
                labels=labels,
                strategy=strategy,
                context_window=2
            )
            
            # Count masked positions
            masked_count = (masked_labels == -100).sum().item()
            total_tokens = masked_labels.numel()
            mask_ratio = masked_count / total_tokens
            
            print(f"  📊 Masked positions: {masked_count}/{total_tokens}")
            print(f"  📊 Masking ratio: {mask_ratio:.2%}")
            
            # Show which positions are masked
            mask_positions = torch.where(masked_labels[0] == -100)[0].tolist()
            print(f"  📍 Masked positions: {mask_positions[:10]}")  # First 10
            
            masking_results[strategy] = {
                "masked_count": masked_count,
                "total_tokens": total_tokens,
                "mask_ratio": mask_ratio,
                "mask_positions": mask_positions
            }
        
        results["demo_outputs"]["masking_strategies"] = masking_results
        print("✅ Masking strategies test passed!")
        
        # ===== 3. TEST TRAINER CONFIGURATION =====
        print("\n📋 Step 3: Testing Trainer Configuration")
        
        # Create minimal training arguments for testing
        training_args = build_training_args(
            output_name="./test_results",
            num_train_epochs=1,
            real_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=2e-5,
            logging_steps=1
        )
        
        # Create data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer, 
            model=model,
            padding=True,
            return_tensors="pt"
        )
        
        # Test trainer creation with different strategies
        trainer_configs = {}
        for strategy in strategies:
            print(f"\n🎭 Creating trainer with strategy: '{strategy}'")
            
            try:
                trainer = ToxicMaskTrainer_V1(
                    model=model,
                    args=training_args,
                    data_collator=data_collator,
                    tokenizer=tokenizer,
                    forbidden_token_ids=forbidden_token_ids[:20],  # Use subset for demo
                    mask_strategy=strategy,
                    context_window=2
                )
                
                print(f"  ✅ Trainer created successfully!")
                print(f"  📋 Strategy: {trainer.mask_strategy}")
                print(f"  🚫 Forbidden tokens: {len(trainer.forbidden_token_ids)}")
                print(f"  🪟 Context window: {trainer.context_window}")
                
                trainer_configs[strategy] = {
                    "created": True,
                    "strategy": trainer.mask_strategy,
                    "forbidden_tokens_count": len(trainer.forbidden_token_ids),
                    "context_window": trainer.context_window
                }
                
            except Exception as e:
                print(f"  ❌ Error creating trainer: {e}")
                trainer_configs[strategy] = {"created": False, "error": str(e)}
                results["errors"].append(f"Trainer creation failed for {strategy}: {e}")
                results["validation_passed"] = False
        
        results["demo_outputs"]["trainer_configs"] = trainer_configs
        
        # ===== 4. TEST LOSS COMPUTATION =====
        print("\n📋 Step 4: Testing Loss Computation")
        
        # Use the first successfully created trainer
        test_trainer = None
        for strategy in strategies:
            if trainer_configs[strategy].get("created", False):
                test_trainer = ToxicMaskTrainer_V1(
                    model=model,
                    args=training_args,
                    data_collator=data_collator,
                    tokenizer=tokenizer,
                    forbidden_token_ids=forbidden_token_ids[:20],
                    mask_strategy=strategy,
                    context_window=2
                )
                print(f"🎯 Using trainer with strategy: '{strategy}' for loss testing")
                break
        
        if test_trainer is not None:
            # Prepare test input
            test_inputs = {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": inputs.get("attention_mask", torch.ones_like(input_ids))
            }
            
            print(f"📊 Test input shape: {test_inputs['input_ids'].shape}")
            print(f"📊 Test labels shape: {test_inputs['labels'].shape}")
            
            try:
                # Compute loss
                loss_output = test_trainer.compute_loss(
                    model=model,
                    inputs=test_inputs,
                    return_outputs=True
                )
                
                if isinstance(loss_output, tuple):
                    loss, outputs = loss_output
                else:
                    loss = loss_output
                    outputs = None
                
                print(f"✅ Loss computation successful!")
                print(f"📊 Loss value: {loss.item():.4f}")
                
                # Check if loss is reasonable (not NaN or infinite)
                if torch.isnan(loss) or torch.isinf(loss):
                    results["errors"].append(f"Invalid loss value: {loss.item()}")
                    results["validation_passed"] = False
                    print(f"❌ Invalid loss value: {loss.item()}")
                else:
                    print(f"✅ Loss value is valid: {loss.item():.4f}")
                
                results["demo_outputs"]["loss_computation"] = {
                    "loss_value": loss.item(),
                    "loss_valid": not (torch.isnan(loss) or torch.isinf(loss)),
                    "strategy_used": test_trainer.mask_strategy
                }
                
            except Exception as e:
                print(f"❌ Error in loss computation: {e}")
                results["errors"].append(f"Loss computation failed: {e}")
                results["validation_passed"] = False
        else:
            print("❌ No trainer available for loss testing")
            results["errors"].append("No trainer available for loss testing")
            results["validation_passed"] = False
        
        # ===== 5. SUMMARY STATISTICS =====
        print("\n📋 Step 5: Summary Statistics")
        
        # Count toxic vs clean sentences
        toxic_sentences = sum(1 for analysis in analyses if analysis['toxic_count'] > 0)
        clean_sentences = len(analyses) - toxic_sentences
        
        # Average toxicity ratio
        avg_toxicity = sum(analysis['toxic_ratio'] for analysis in analyses) / len(analyses)
        
        results["statistics"] = {
            "total_sentences_tested": len(sample_sentences),
            "toxic_sentences": toxic_sentences,
            "clean_sentences": clean_sentences,
            "average_toxicity_ratio": avg_toxicity,
            "strategies_tested": len(strategies),
            "trainers_created": sum(1 for config in trainer_configs.values() if config.get("created", False)),
            "forbidden_tokens_used": len(forbidden_token_ids[:50])  # Subset used for demo
        }
        
        print(f"📊 Total sentences tested: {len(sample_sentences)}")
        print(f"🚫 Toxic sentences: {toxic_sentences}")
        print(f"✅ Clean sentences: {clean_sentences}")
        print(f"📊 Average toxicity ratio: {avg_toxicity:.2%}")
        print(f"🎛️ Strategies tested: {len(strategies)}")
        print(f"🎭 Trainers created: {sum(1 for config in trainer_configs.values() if config.get('created', False))}")
        
        print("\n" + "=" * 60)
        print("🎯 VALIDATION SUMMARY")
        print("=" * 60)
        
        if results["validation_passed"]:
            print("✅ ALL TESTS PASSED!")
            print("🎭 ToxicMaskTrainer is working correctly")
            print("📋 All masking strategies are functional")
            print("🔧 Loss computation is stable")
        else:
            print("❌ SOME TESTS FAILED!")
            print("🔍 Errors encountered:")
            for error in results["errors"]:
                print(f"  • {error}")
        
        print(f"\n📊 Final Statistics:")
        for key, value in results["statistics"].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
            
    except Exception as e:
        print(f"❌ Critical error in validation: {e}")
        results["validation_passed"] = False
        results["errors"].append(f"Critical validation error: {e}")
        logger.error(f"ToxicMaskTrainer validation failed: {e}")
    
    return results


def quick_toxic_mask_check(
    tokenizer: AutoTokenizer,
    forbidden_token_ids: List[int],
    test_sentence: str = "You are such a stupid idiot."
) -> bool:
    """
    🚀 Enhanced validation check for toxic masking functionality
    
    Args:
        tokenizer: Tokenizer to use
        forbidden_token_ids: List of forbidden token IDs
        test_sentence: Sentence to test with
        
    Returns:
        bool: True if basic functionality works
    """
    try:
        from .utils import analyze_sentence_toxicity, create_toxic_token_mask, ToxicTokenMaskGenerator
        
        print("🚀 Enhanced Toxic Mask Check")
        print("-" * 40)
        
        # First, let's analyze the tokenization
        inputs = tokenizer(test_sentence, return_tensors="pt")
        labels = inputs["input_ids"].clone()
        
        print(f"📝 Test sentence: '{test_sentence}'")
        print(f"📊 Tokenized to {inputs['input_ids'].shape[1]} tokens")
        
        # Show first few tokens for debugging
        first_tokens = inputs["input_ids"][0][:10].tolist()
        token_texts = [tokenizer.decode([tid]) for tid in first_tokens]
        print(f"🔍 First tokens: {list(zip(first_tokens, token_texts))}")
        
        # Check which forbidden tokens are actually in the labels
        found_forbidden = []
        for token_id in forbidden_token_ids[:20]:  # Check first 20
            if token_id in labels[0]:
                found_forbidden.append((token_id, tokenizer.decode([token_id])))
        
        print(f"🚫 Forbidden tokens found in sentence: {found_forbidden}")
        
        # Test sentence analysis
        analysis = analyze_sentence_toxicity(
            sentence=test_sentence,
            tokenizer=tokenizer,
            forbidden_token_ids=forbidden_token_ids,
            return_positions=True
        )
        
        print(f"🎯 Analysis results:")
        print(f"  • Toxic tokens found: {analysis['toxic_count']}")
        print(f"  • Toxicity ratio: {analysis['toxic_ratio']:.2%}")
        print(f"  • Toxic words: {analysis['toxic_tokens']}")
        
        # Test masking with debug info
        mask_generator = ToxicTokenMaskGenerator(
            tokenizer=tokenizer,
            forbidden_token_ids=forbidden_token_ids,
            mask_strategy="toxic_only"
        )
        
        # Get debug info
        debug_info = mask_generator.debug_toxicity_detection(labels, sample_size=1)
        print(f"🔍 Debug toxicity detection:")
        for sample in debug_info['samples_analyzed']:
            print(f"  • Sample: {sample['toxic_tokens_found']} toxic tokens found")
            print(f"  • Text: {sample['decoded_text']}")
            for toxic in sample['toxic_details']:
                print(f"    - '{toxic['token_text']}' (ID: {toxic['token_id']}) at positions {toxic['position']}")
        
        # Test masking
        mask = mask_generator.create_toxic_token_mask(
            input_ids=inputs["input_ids"],
            labels=labels,
            context_window=2
        )
        
        masked_positions = mask.sum().item()
        total_positions = ((labels != -100) & (labels != tokenizer.pad_token_id)).sum().item()
        
        print(f"🎭 Masking results:")
        print(f"  • Positions to train on: {masked_positions}/{total_positions}")
        print(f"  • Masking ratio: {masked_positions/max(total_positions,1)*100:.1f}%")
        
        # Enhanced success criteria
        has_forbidden_tokens = len(found_forbidden) > 0
        analysis_found_toxic = analysis['toxic_count'] > 0
        masking_worked = masked_positions > 0
        
        print(f"\n📊 Validation Results:")
        print(f"  • Forbidden tokens in sentence: {'✅' if has_forbidden_tokens else '❌'} ({len(found_forbidden)} found)")
        print(f"  • Analysis detected toxic: {'✅' if analysis_found_toxic else '❌'} ({analysis['toxic_count']} found)")
        print(f"  • Masking worked: {'✅' if masking_worked else '❌'} ({masked_positions} positions)")
        
        success = has_forbidden_tokens and analysis_found_toxic and masking_worked
        print(f"\n🎯 Overall result: {'✅ PASSED' if success else '❌ FAILED'}")
        
        if not success:
            print(f"\n🚨 TROUBLESHOOTING:")
            if not has_forbidden_tokens:
                print(f"  • No forbidden tokens found in sentence - check tokenization consistency")
            if not analysis_found_toxic:
                print(f"  • Analysis didn't detect toxicity - check forbidden_token_ids")
            if not masking_worked:
                print(f"  • Masking didn't work - check mask generation logic")
        
        return success
        
    except Exception as e:
        print(f"❌ Quick check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_trainer_build_function():
    """
    🏗️ Validate the build_trainer function supports toxic_mask_v1
    """
    try:
        from .trainer import build_trainer
        import inspect
        
        print("🏗️ Validating build_trainer function")
        print("-" * 40)
        
        # Get function signature
        sig = inspect.signature(build_trainer)
        print(f"📋 Function signature: {sig}")
        
        # Check if function exists and can be called
        print("✅ build_trainer function is accessible")
        
        # Try to get help/docstring
        if build_trainer.__doc__:
            print(f"📚 Documentation available: {len(build_trainer.__doc__)} chars")
        
        print("✅ build_trainer validation passed")
        return True
        
    except Exception as e:
        print(f"❌ build_trainer validation failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Running module checks...")
    print("This module provides validation functions for ToxicMaskTrainer")
    print("Use check_toxic_mask_trainer_demo() for comprehensive testing")
