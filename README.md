# PLN-Project
Repositorio del proyecto desarrollado para la tarea [PAN 2025 de Multilingual Text Detoxification](https://pan.webis.de/clef25/pan25-web/text-detoxification.html). Su objetivo es generar versiones no tóxicas de textos ofensivos en varios idiomas manteniendo el contenido original. Incluye experimentos con modelos de lenguaje, evaluación y análisis de resultados.

## Purpose

This project focuses on **text detoxification**, transforming toxic or harmful texts into neutral versions that preserve their original meaning but remove any form of *toxicity*.  

There are several possible approaches to achieve this goal:

- **Masking and filling**: identifying toxic words or expressions and replacing them with neutral alternatives.  
- **Text generation**: generating a new text based on the toxic one, maintaining the same semantic content while modifying specific words and expressions.  
- **Hybrid approach** (explained later in section [Hybrid Approach](#hybrid-approach-wip)).

Each approach has its own strengths and limitations. The following sections explore these ideas in detail.

---

## Masking and Filling Toxicity

This approach will likely involve two or more models.  
First, a **classifier** will label each token as either *toxic* or *non-toxic*.  
This can be done using two of the provided datasets:  
`datasets/multilingual_toxic_lexicon` and `datasets/multilingual_toxic_spans`.  
These datasets link tokens and expressions directly with their toxicity labels.  

Once toxic tokens are detected, they will be **masked**, and a **filling model** will generate new words to replace them appropriately.

**Advantages**:
- Only the classifier needs to be trained; the filling model can often work without fine-tuning.  
- Uses two out of the four available datasets.  

**Disadvantages**:
- Limited flexibility - only modifies the tokens labeled as *toxic*.  
- If additional words need to change for grammatical or semantic coherence, this method cannot handle it effectively.  

---

## Rewriter Model

In this approach, a **text generation model** will be fine-tuned (using techniques such as **LoRA** or **QLoRA**) with the dataset `datasets/toxic_nontoxic`.  
During fine-tuning, toxic phrases will be used as inputs, and the model will learn to minimize the difference between the output embeddings and those of the corresponding *non-toxic* phrases.  

This allows the model to generate non-toxic text **conditioned on** a toxic input.  

**Advantages**:
- Highly flexible - works at the **semantic** level rather than purely lexical.  

**Disadvantages**:
- Fine-tuning can be computationally expensive, especially with large models.  
- Uses only one dataset.  

## Hybrid Approach (WIP)