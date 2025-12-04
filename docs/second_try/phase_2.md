# Phase 2 Training Plan: Limitations of Baseline and Improved Strategy

## 1. Summary of the Baseline Training

The baseline model (**mT5 + LoRA**) was trained on ~3k toxic → neutral sentence pairs across 9 languages.  
Results in validation showed:

- Validation loss ≈ **1.49**
- Perplexity ≈ **4.4**

However, evaluation on the test set revealed:

- ~72% of predictions were **identical to the toxic inputs**
- Only 1 prediction matched the gold neutral target exactly
- Detoxification was mostly **absent or incorrect**

This indicates that although the model optimized its loss, it **failed to learn the detoxification transformation**.

---

## 2. Problems Identified in the Baseline

### **2.1. Shortcut Learning: Copying the Input**
Most toxic and neutral pairs differ only by 1–2 words.  
CrossEntropy encourages the model to learn the shortcut:

> Copy everything and risk only a small penalty on the toxic token.

This behavior is common in seq2seq tasks with high lexical overlap.

---

### **2.2. Teacher Forcing Hides Generative Errors**
During training, the model always sees the correct previous token.  
Thus:

- It appears accurate in loss space  
- But in inference (auto-regressive mode), it copies or fails to detox

---

### **2.3. Dataset Size and Noise**
With only ~3k examples:

- Limited supervision
- Hard, medium, and noisy examples are mixed together
- Model cannot infer consistent detoxification patterns

---

### **2.4. No Explicit Incentive to Remove Toxicity**
CrossEntropy does **not** enforce:

- Removing insults  
- Reducing toxic patterns  
- Avoiding toxic vocabulary during decoding  

So the model chooses the energetically cheapest strategy: **copy the input**.

---

### **2.5. Weak Semantic Alignment Enforcement**
The model is not trained to preserve meaning or structure beyond matching tokens.

---

## 3. Objectives of the Second Training Phase

### **🎯 1. Prevent the Copy-Input Shortcut**
We must structurally prevent the model from copying the toxic input unchanged.

---

### **🎯 2. Increase Pressure to Detoxify**
Add mechanisms that penalize toxic outputs or boost non-toxic alternatives.

---

### **🎯 3. Preserve Meaning**
Ensure that the neutralization does not distort the semantic content.

---

### **🎯 4. Improve Signal Supervision**
Reduce dataset noise and reorganize the training curriculum.

---

## 4. Plan to Address the Problems

### **4.1. Curriculum Learning**
Train in stages:

1. **Easy** examples → high lexical overlap  
2. **Medium** → moderate rewriting  
3. **Hard** → heavy rephrasing or strong toxicity  

Benefit: forces model to learn detox gradually and meaningfully.

---

### **4.2. Dataset Cleaning**
Remove:

- Inputs already neutral  
- Outputs inconsistent with neutralization  
- Pairs with excessive rewriting or noise  

This increases supervision clarity.

---

### **4.3. Decoding Controls**
Use `LogitsProcessor` to:

- Penalize toxic tokens  
- Downweight offensive vocabulary  
- Increase generation of safer alternatives  

This prevents toxic copying during inference.

---

### **4.4. Better Evaluation**
Use:

- **BERTScore** → semantic similarity  
- **BLEU** → structural similarity  
- **Toxicity classifier** → measure residual toxicity  

This aligns evaluation with the real goal: **non-toxic semantic preservation**.

---

### **4.5. Optional: Additional Loss Functions**
If needed, introduce:

- **Toxicity loss** (classifier-based)  
- **Contrastive loss** to discourage excessive similarity between input/output  

This is a more advanced, research-level technique.

---

## 5. Expected Benefits After Phase 2

### **⭐ 1. Reduction of Identity Outputs**
Goal: reduce from 72% identical outputs → **below 25%, ideally <10%**.

---

### **⭐ 2. More Consistent Detoxification**
We expect:

- Reliable removal of insults  
- Preservation of entities  
- Meaning retention  

---

### **⭐ 3. Quantitative Improvements**
- Higher BERTScore  
- Lower toxicity score  
- BLEU similar or slightly lower (due to detoxification changes)

---

### **⭐ 4. More Natural and Safe Outputs**
The model should generate:

- Fluent sentences  
- Non-toxic paraphrases  
- Minimal rewriting beyond necessary detoxification  

---

### **⭐ 5. Strong Baseline Comparison**
The baseline will act as a reference point to demonstrate:

- How curriculum learning improves detoxification  
- How decoding constraints reduce toxicity  
- How data cleaning improves consistency  

---

## 6. Next Steps

To begin Phase 2:

1. Implement curriculum splits  
2. Clean noise and inconsistent examples  
3. Train in stages (easy → medium → hard)  
4. Add decoding penalties against toxic tokens  
5. Evaluate with semantic + toxicity metrics  
6. Compare with baseline

# CurriculumSeq2SeqTrainer: Design and Training Strategy

## 1. Motivation for a New Trainer

The baseline model trained in Phase 1 demonstrated a critical issue:  
although the validation loss decreased steadily, the model overwhelmingly learned a **shortcut behavior**—copying the input rather than performing meaningful detoxification.

This occurs due to:

- High lexical overlap between toxic and neutral sentences.
- CrossEntropy loss rewarding token-level similarity.
- Teacher forcing masking generative errors.
- Small dataset with heterogeneous difficulty levels.
- Lack of explicit incentives to remove toxic content.

Therefore, the Phase 2 system requires a **new trainer**, not to replace the HuggingFace trainer internals, but to *control the learning process* in a structured, curriculum-based sequence.

---

## 2. Purpose of the CurriculumSeq2SeqTrainer

The goal of the new trainer is to:

1. **Guide the model through gradually increasing difficulty levels**  
   (easy → medium → hard → full dataset).

2. **Avoid shortcut learning by isolating detoxification behavior**  
   through staged exposure.

3. **Consolidate generalization after learning detoxification rules.**

4. **Enable phase-specific metrics, logging, and model checkpoints.**

5. **Support optional decoding-based toxicity control**  
   (e.g., LogitsProcessor to reduce toxic token probabilities).

The trainer builds on top of HuggingFace's `Seq2SeqTrainer`, extending it with multi-phase orchestration, tracking, and curriculum scheduling.

---

## 3. What This Trainer Is *Not*

- It is **not** a re-implementation of the training loop.  
- It does **not** require a custom loss function at this stage.  
- It does **not** modify the architecture of mT5 or LoRA adapters.  

The internal optimization remains based on **standard CrossEntropy**, which is stable and sufficient when paired with curriculum learning.

The objective is to control *how* the model learns, not reinvent *what* it learns.

---

## 4. Architecture Overview

The `CurriculumSeq2SeqTrainer` will act as a high-level controller:

```
CurriculumSeq2SeqTrainer
 ├── stage: easy
 ├── stage: medium
 ├── stage: hard
 ├── stage: full consolidation
 ├── evaluation on validation/test sets
 ├── metric aggregation per stage
 └── optional decoding constraints (toxicity-aware generation)
```

It wraps around a standard `Seq2SeqTrainer` configured with:

- the same model instance,
- the same tokenizer,
- the same LoRA adapters,
- the same compute_metrics function,
- the same data collator.

---

## 5. Curriculum Schedule Design

A schedule defines the training sequence, example:

```python
[
  {"stage": "easy",   "epochs": 2, "lr": 2e-4},
  {"stage": "medium", "epochs": 2, "lr": 2e-4},
  {"stage": "hard",   "epochs": 1, "lr": 1e-4},
  {"stage": "full",   "epochs": 1, "lr": 1e-4}
]
```

Each stage:

1. Loads a different subset of the curriculum.
2. Optionally adjusts learning rate.
3. Trains for a specific number of epochs.
4. Logs metrics and saves checkpoints tagged by stage.

---

## 6. Metric Tracking Requirements

For each stage, the trainer must collect:

- **Training loss**
- **Validation loss**
- **Perplexity**
- **Semantic preservation (BERTScore)**
- **Structural similarity (BLEU / chrF)**
- **Copy-rate**  
  (percentage of outputs identical to inputs)
- **Optional: toxicity residual score**

This enables direct comparison between:

- baseline (single-stage),
- curriculum-trained model (multi-stage).

---

## 7. Loss Function Requirements

The **standard sequence-to-sequence CrossEntropy** is sufficient for Phase 2.

However, the trainer must be designed to optionally plug in future advanced loss components, such as:

1. **Toxicity-aware loss**  
   Penalizing outputs classified as toxic.

2. **Contrastive loss**  
   Reducing similarity between toxic input and output when toxicity is present.

3. **Semantic consistency loss**  
   Using embeddings to preserve meaning.

These are *not necessary now*, but the trainer must be flexible enough to accommodate them in Phase 3 of the project.

---

## 8. Decoding-Control Integration

During evaluation and inference, the trainer must be able to activate:

### **Toxicity-aware LogitsProcessor**

- Reduces logits of toxic tokens.
- Helps prevent copying insults.
- Works without modifying the loss.

This is essential to align the generation behavior with the detoxification objective.

---

## 9. Training Flow Summary

The high-level flow:

```
1. Load model, tokenizer, curriculum datasets.
2. Initialize CurriculumSeq2SeqTrainer.
3. For each stage:
       - set train_dataset
       - set learning rate or scheduler
       - run HF Trainer.train()
       - evaluate and store metrics
       - save checkpoint specific to stage
4. After last stage:
       - full evaluation on test set
       - generate qualitative samples
       - produce comparison report
```

---

## 10. Expected Improvements

With this trainer, we expect:

- **Reduced copy-rate** (from ~72% → <25% ideally).
- **Better detoxification consistency.**
- **Better semantic preservation.**
- **More meaningful generation behavior.**
- **Clear per-stage metric diagnostics** enabling scientific evaluation.
- **Much stronger model than baseline**, while remaining interpretable.

---

## 11. Conclusion

The new trainer does not alter the underlying loss or architecture.  
Instead, it provides a **structured learning environment** required for detoxification to emerge naturally:

- staged exposure,
- focused specialization,
- meaningful generalization,
- interpretability of progress.

This “CurriculumSeq2SeqTrainer” becomes the core of Phase 2 and the foundation for Phase 3 enhancements.

