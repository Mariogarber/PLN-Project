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
