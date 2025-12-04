# Phase 3 — Detoxification-Aware Learning & Controlled Generation  
### Advanced Training Pipeline for Multilingual Detoxification Using mT5

---

## 1. Overview

Phase 3 addresses the fundamental weaknesses identified after Phase 2:

- The model no longer copies excessively, but **still fails to detoxify reliably**.  
- Many toxic words remain in the predictions.  
- `<extra_id_0>` appears frequently, indicating decoding instability.  
- Semantic drift is too low → model stays too close to the toxic input.  
- Residual toxicity measurement requires improvement.  
- The model lacks **explicit incentives** to remove harmful content.

Phase 3 introduces a complete redesign of the training and inference pipeline to explicitly enforce detoxification behavior while preserving semantic meaning.

---

## 2. Core Objectives

Phase 3 aims to:

1. **Prevent the model from generating toxic tokens** (multilingual).
2. **Introduce losses that reward detoxification explicitly**, not implicitly.
3. **Force semantic separation between toxic input and neutral output**.
4. **Stabilize decoding and remove `<extra_id_*>` artifacts.**
5. **Improve evaluation using toxicity-focused metrics.**
6. **Enable cross-lingual detoxification through controlled generation.**

---

## 3. Pipeline Structure

### Phase 3 consists of three major components:

---

## A. Controlled Decoding  
### (Hard constraint — Prevent toxic output)

The generative model must be prevented from choosing toxic or unwanted tokens during decoding.

### Key mechanism: **Toxicity-Aware LogitsProcessor**

- Penalizes token logits belonging to language-specific toxic lexicons.  
- Applies penalties of **–5 to –15 logits** (tunable).  
- Removes `<extra_id_*>` tokens from the generation space.  
- Works across languages: English, Spanish, Russian, Arabic, etc.  
- Directly reduces toxic outputs, even without retraining.

This component alone produces a major qualitative improvement and is a foundational element for real detox systems (similar to GeDi or DExperts).

---

## B. Detox-Aware Training  
### (Soft constraint — Encourage meaningful detoxification)

Standard CrossEntropy (CE) loss does not penalize toxic output.  
Phase 3 introduces *additional objectives* to guide the model.

### 1. **Contrastive Loss (Input → Prediction → Gold Neutral)**

We want:

- `prediction` to be **close** to `gold_neutral`
- `prediction` to be **far** from the `toxic_input`

Contrastive objective:

```
loss = CE_loss(model_output, gold)
     + α * max(0, margin + sim(pred, toxic) - sim(pred, gold))
```

This forces the model to stop mimicking the toxic input and instead align with the detoxified target.

### 2. **Residual Toxicity Penalty**

Penalize predicted sentences containing toxic lexicon items:

```
loss += β * toxicity_score(prediction)
```

Optional extension:

```
loss += β * sum(logits[token] for token in toxic_tokens)
```

This directly teaches the model that producing insults has a cost.

### 3. **Optional: RLHF-Light (Reward-Shaping)**

Define a reward:

```
reward = - toxicity(pred) + semantic_similarity(pred, gold)
```

Train using a small policy-gradient step (not full PPO).  
This helps refine edge cases where CE + contrastive loss are insufficient.

---

## C. Dataset Reinforcement & Cleaning

### 1. **Remove sentinel tokens `<extra_id_*>` from all training targets**
Their presence in outputs indicates misalignment.

### 2. **Improve multilingual toxic lexicons**
The lexicon must:

- include variants of insults,
- handle morphology,
- support Russian, Arabic, Chinese, etc.

### 3. **Create a Hard-Rewrite subset**
A small curated set of examples requiring:

- structural changes,
- idiomatic rewriting,
- stronger paraphrasing.

These help the model break out of “minimal edits”.

---

## 4. Enhanced Evaluation Framework

Phase 3 must use **evaluation metrics that measure detoxification directly**, not only similarity.

### 1. Core metrics:

- **Detoxification Success Rate**
- **Toxicity Delta → tox(input) – tox(pred)**
- **Residual Toxicity** (improved lexicon + classifier)
- **Semantic Preservation → SBERT similarity**
- **Rewriting Distance → semantic drift**
- **Copy Rate** (aim: <10%)

### 2. Multilingual evaluation:

- by language,
- by difficulty level,
- per toxic category.

### 3. Visual tools:

- Heatmaps per language,
- Semantic drift distribution,
- Length ratio histograms,
- Semantic vs toxicity scatter.

---

## 5. Expected Outcomes

After Phase 3, the model should:

### ✔ Stop emitting `<extra_id_0>` and other invalid tokens  
### ✔ Remove harmful content consistently  
### ✔ Maintain semantic meaning  
### ✔ Maintain fluency across languages  
### ✔ Reduce copy rate below 10%  
### ✔ Produce outputs that resemble human-style detoxification  
### ✔ Show measurable toxicity reduction across benchmarks  
### ✔ Perform better on medium/hard samples

This pipeline transforms the model from “slightly modifies toxic text” → **“actively neutralizes toxic text consistently”**.

---

## 6. Implementation Roadmap

### Week 1:
- Implement LogitsProcessor  
- Expand lexicon  
- Add decoding constraints

### Week 2:
- Implement contrastive loss  
- Add toxicity penalty  
- Integrate into trainer

### Week 3:
- Re-train model with new objectives  
- Validate on controlled examples  
- Tune α, β, margin

### Week 4:
- Full evaluation  
- Outputs comparison with Phase 1 and Phase 2  
- Prepare final report & analysis  

---

## 7. Conclusion

Phase 3 converts your system into a **scientifically grounded detoxification pipeline**, combining:

- controlled generation,
- explicit detoxification incentives,
- semantic constraints,
- multilingual stability.

This is the phase in which your model will *actually learn to detoxify reliably*, leaving behind shortcut behaviors and producing consistent, high-quality neutral rewrites.

---
