# Base Model

Selected Base model is `google/mt5-base`. Transformer model Seq2Seq type, include encoder and decoder. When trained, the model recieve a first word that represents the 
task he should do, and the rest of the input. It encode the input and use a decoder to inference the most probable token next.

# QLoRA Config based

## "Q" Config
Model loaded using 4 bits. However, they were `nf4` bit-type, so good representation. When compute, they cast to `bfloat16`, to add stability on training.

## LoRA Config

### First Approach

- Rank of matrix: 8
- Lora Alpha: 16
- No bias
- Dropout: 0.05
- Target Modules = Q and V

### Second Approach

- Rank of matrix: 8
- Lora Alpha: 16
- No bias
- Dropout: 0.05
- Target Modules = Q, V, K, O

# Custom Loss

Because we need to detoxify a dataset, we can use several loss function to finetune our model.

## Original Loss Function

Just use the original loss function that Mt5 implemented to try to reconstruct the sentences without any toxic word. However, some limitations of this is that the model 
does not care about generate or not toxic words, because it do not penalize them

## Toxic Penalization

We add a new component to the loss function, this component requires a list of toxic tokens, and compute a penalization if that tokens had higher probabilities on logits layer. We need to work with logits to ensure that the gradiant of this component does not just disapear. We define a gamma parameter to set the importance of the toxic penalization related with the original loss.

# Limitations

## Identity Mapping
Sometimes input are so similar to outputs, and if we do not compute the toxic penalization or the loss correctly, and if we do not define the task, the model just become a identity mapping, because is the easiest and the lowest-loss solution. 

## Not Defined task
MT5 tends to copy the input when no task is defined, soy we need to add the task before any input sequence. In our case, our task is just `detoxify: `.

# Tries

## First Try

The first try use the original function and the first lora config. I do not compute the metrics because the model was just copying the input. I do not add the task prefix to the inputs.

## Second Try

I add a toxic optimization of 0.15. Also I add the task before the input. Used the first lora config. The result is again the model copying the input IDK why.

# Toxic Mask Trainer

After experiencing identity mapping issues with penalty-based approaches, I developed a novel **Toxic Mask Trainer** that uses a fundamentally different strategy: **selective token masking**.

## Core Concept

Instead of penalizing toxic tokens in the loss function, the Toxic Mask Trainer **masks non-relevant positions** during training, forcing the model to focus learning **only on specific token positions**. This approach addresses the identity mapping problem by ensuring the model cannot simply copy input tokens—it must actively learn to replace toxic content.

## Masking Strategies

The trainer implements three distinct masking strategies:

### 1. "toxic_only" Strategy
- **Focus**: Train ONLY on toxic token positions
- **Method**: All non-toxic token positions are masked (-100 in labels)
- **Advantage**: Maximum focus on problematic content
- **Use case**: When you want aggressive detoxification with minimal changes to clean content

### 2. "toxic_context" Strategy  
- **Focus**: Train on toxic tokens + surrounding context window
- **Method**: Masks everything except toxic tokens and N neighboring tokens
- **Advantage**: Preserves semantic coherence around toxic content
- **Use case**: When replacement words need grammatical/semantic consistency

### 3. "inverse_toxic" Strategy
- **Focus**: Train on everything EXCEPT toxic tokens
- **Method**: Only toxic token positions are masked
- **Advantage**: Model learns general language patterns while avoiding toxic reinforcement
- **Use case**: For improving overall fluency while preventing toxic token generation

## Technical Implementation

```python
# Example trainer creation
trainer_toxic_mask = build_trainer(
    trainer_name="toxic_mask_v1",
    model=model_qlora,
    args=training_args,
    dataset_dict={"train": train, "validation": eval},
    data_collator=data_collator,
    callbacks=callbacks,
    forbidden_token_ids=forbidden_token_ids,
    mask_strategy="toxic_only",  # or "toxic_context", "inverse_toxic"
    context_window=2  # for "toxic_context" strategy
)
```

## Key Advantages

1. **🎯 Targeted Learning**: Model focuses computational resources only where needed
2. **⚡ Efficiency**: Reduced training time by ignoring irrelevant positions  
3. **🔄 Prevents Identity Mapping**: Cannot copy input due to strategic masking
4. **🎛️ Flexibility**: Multiple strategies for different detoxification needs
5. **📊 Measurable Impact**: Clear masking statistics for training monitoring

## Validation System

Implemented comprehensive validation with `finetunning/checks.py`:

- **Sentence Analysis**: Tests toxicity detection on multiple languages
- **Strategy Testing**: Validates all masking approaches with real data
- **Trainer Configuration**: Ensures proper setup for each strategy
- **Loss Computation**: Verifies training stability with masked labels
- **Statistics Tracking**: Detailed metrics on masking effectiveness

## Expected Benefits Over Penalty Approach

1. **No Identity Mapping**: Masking prevents direct copying behavior
2. **Focused Learning**: Resources concentrated on meaningful changes
3. **Better Convergence**: Clearer training signal without conflicting objectives
4. **Language Agnostic**: Works across Spanish/English without language-specific tuning
5. **Interpretable**: Easy to understand which tokens the model is learning to modify

This approach represents a shift from **"penalize bad behavior"** to **"focus on learning good replacements"**, which should be more effective for the detoxification task.