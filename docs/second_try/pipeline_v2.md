# Revised Strategy for mT5 Detoxification Fine-Tuning After Dataset Analysis

## 1. Introduction
The initial plan for this project was to perform a multilingual fine-tuning of **mT5** for detoxification: given a toxic sentence, the model should output a semantically faithful, neutralised version of the input.

However, after running a full exploratory analysis of the dataset—including lexical, semantic and structural statistics—new issues emerged that directly affect the stability and generalisation of the fine-tuned model.  

This document summarises:

- The **problems uncovered** during the analysis.  
- The **risks** they pose for model training.  
- The **updated approach** for preparing the dataset and training mT5.  
- The decision to **keep the original baseline fine-tuning path** for comparison.

---

## 2. Problems Identified in the Dataset

### 2.1. Severe Length Imbalance Across Languages
Different languages exhibit drastically different average sentence lengths:

- *Chinese ≈ 1–2 tokens*  
- *German ≈ 15–17 tokens*  
- Others between *9–12 tokens*

#### Why this is a problem
mT5 can easily *exploit spurious correlations*:

- “short sentence → likely Chinese → neutral”
- “long sentence → likely German → toxic”

This leads to:
- **Label leakage**  
- **Poor cross-lingual generalisation**  
- **Bad behaviour on unseen languages**  

---

### 2.2. Semantic Preservation is Not Consistent
Neutral sentences often diverge significantly from toxic ones based on *equal_percentage* and embedding similarity.

Some examples behave like:
- **Free paraphrasing**
- **Simplification**
- **Content alteration**

instead of genuine detoxification.

#### Implications
- The model does not learn *controlled style transfer*  
- It may overwrite content unnecessarily  
- It can hallucinate new information  

---

### 2.3. Presence of Hard Cases and Low-Quality Example Buckets
There are:
- Low-overlap examples  
- Semantically incoherent pairs  
- Very short or nearly-identical pairs  

#### Implications
Noise introduces:
- Poor convergence  
- Over-rewriting  
- Unstable mapping patterns  

---

### 2.4. Quality Inconsistency Between Languages
Some languages show worse semantic alignment (am, de, hi), while others behave consistently.

#### Implications
- mT5 becomes **excellent** in some languages  
- And **poor or unstable** in others  

---

## 3. Consequences for Fine-Tuning the Model

If trained directly (without corrections):

- The model may rewrite entire sentences  
- It may use superficial cues (length, token patterns)  
- It may hallucinate on difficult examples  
- It may fail in inconsistent languages  

The outcome would be:
- Strong performance in some languages  
- Weak behaviour in others  
- Poor semantic preservation overall  

---

## 4. Updated Strategy for a More Robust Fine-Tuning

### 4.1. Per-Language Preprocessing
For each language:
- Compute difficulty  
- Filter internal noise  
- Avoid structural leakage  

✔ Improves cross-lingual consistency  

---

### 4.2. Filtering by Overlap and Embedding Similarity
Remove examples where:

```
equal_percentage < 0.30
embedding_similarity < 0.40
```

✔ Reduces hallucination  
✔ Improves semantic fidelity  

---

### 4.3. Curriculum Learning Based on Difficulty
Split dataset into:

- **Easy** (high overlap/similarity)  
- **Medium**  
- **Hard**  

Training schedule:

```
Stage 1 → Easy
Stage 2 → Easy + Medium
Stage 3 → Full dataset
```

✔ Stabilises early learning  
✔ Reduces noise propagation  

---

### 4.4. Length-Balanced Batching
Group samples into buckets by length to avoid length becoming a shortcut.

✔ Prevents structural leakage  

---

### 4.5. Optional Semantic Loss
Use:

```
Loss = CrossEntropy + λ * (1 - cosine_similarity)
```

✔ Preserves meaning  
✔ Reduces unnecessary rewriting  

---

## 5. Keeping the Baseline Fine-Tuning Approach

We will also maintain the **original simple mT5 fine-tuning**, using:

- Raw dataset  
- No filtering  
- No curriculum  
- No semantic constraints  
- No per-language adjustments  

### Why keep it?
To enable:
- **Ablation studies**  
- **Quantitative comparison**  
- **Clear justification of improvements**  

---

## 6. Expected Benefits of the New Strategy

| Issue | Baseline Effect | New Strategy Mitigation |
|-------|-----------------|--------------------------|
| Length imbalance | Leakage, shortcuts | Length bucketing |
| Low semantic overlap | Hallucination | Filtering + semantic loss |
| Noise/outliers | Instability | Cleaning + buckets |
| Cross-lingual inconsistency | Uneven performance | Per-language curriculum |
| Hard cases | Collapse early | Gradual learning |

---

## 7. Final Remarks

The revised strategy evolves the model pipeline from naïve multilingual training into a **research-grade detoxification workflow**, backed by empirical dataset understanding.

Keeping the baseline model enables:
- rigorous evaluation  
- stronger scientific reasoning  
- and clearer demonstration of the impact of dataset quality.

This two-track approach ensures both **improved performance** and **credible experimental reporting**.

