import torch
import numpy as np
import pandas as pd
import nltk
import json
import os
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
# Importante: usar la configuración correcta de bitsandbytes
from transformers import BitsAndBytesConfig 
from sentence_transformers import SentenceTransformer, util

# --- CONFIGURACIÓN ---
MODEL_ID = "google/mt5-base"
MAX_LENGTH = 256
BATCH_SIZE = 2          # Con 8GB VRAM, mantén esto en 2 o 4 máximo.
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
LOSS_WEIGHT_VALUE = 3.0
SEMANTIC_THRESHOLD = 0.55
OUTPUT_DIR = "./mt5_qlora_weighted"

# Detectar dispositivo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Hardware detectado: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# --- DESCARGAS NLTK ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- CARGA DE DATOS ---
print("Cargando datos...")
if not os.path.exists("train.json") or not os.path.exists("keywords.csv"):
    raise FileNotFoundError("¡Faltan archivos! Asegúrate de tener 'train.json', 'keywords.csv' y 'toxic_expressions.csv' en la misma carpeta.")

dataset = load_dataset("json", data_files={"train": "train.json"})
train_data = dataset["train"]

keywords_df = pd.read_csv("keywords.csv")
keyword_list = keywords_df["keyword"].tolist()

expr_df = pd.read_csv("toxic_expressions.csv")
expression_list = expr_df["toxic_expression"].tolist()

# Modelo de embeddings para la similaridad (se queda en CPU o GPU secundaria para ahorrar VRAM)
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2").to("cpu")

# --- MODELO Y TOKENIZER ---
print("Cargando modelo MT5 en 8-bits...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto" # Esto distribuirá automáticamente a tu 3060 Ti
)

# Configurar LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --- FUNCIONES DE PROCESADO ---
def compute_loss_weights(input_text, keywords, toxic_expressions, weight=LOSS_WEIGHT_VALUE, max_length=MAX_LENGTH):
    tokens = tokenizer(
        input_text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = tokens["offset_mapping"]
    weights = np.ones(len(offsets), dtype=np.float32)

    # 1. Coincidencia exacta
    lowered = input_text.lower()
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in lowered:
            start_search = 0
            while True:
                idx = lowered.find(kw_lower, start_search)
                if idx == -1: break
                end_idx = idx + len(kw_lower)
                # Marcar tokens que caen en este rango
                for i, (s, e) in enumerate(offsets):
                    if s >= idx and e <= end_idx:
                        weights[i] = weight
                start_search = end_idx

    # 2. Coincidencia semántica (Simplificada para velocidad)
    # Nota: Si esto va muy lento, coméntalo para probar primero que el pipeline funciona
    words = nltk.word_tokenize(input_text)
    if len(words) > 3:
        chunks = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
    else:
        chunks = [input_text]
    
    if chunks:
        chunk_embs = embedder.encode(chunks, convert_to_tensor=True)
        # Codificar expresiones tóxicas (idealmente esto se hace fuera del bucle una sola vez)
        expr_embs = embedder.encode(toxic_expressions, convert_to_tensor=True)
        
        # Calcular similitud
        cosine_scores = util.cos_sim(chunk_embs, expr_embs)
        
        # Si algún chunk supera el umbral
        matches = torch.where(cosine_scores > SEMANTIC_THRESHOLD)
        # matches[0] son índices de chunks, matches[1] son índices de expresiones
        
        for chunk_idx in matches[0].unique().tolist():
            chunk_text = chunks[chunk_idx]
            start = input_text.find(chunk_text)
            if start != -1:
                end = start + len(chunk_text)
                for i, (s, e) in enumerate(offsets):
                    if s >= start and e <= end:
                        weights[i] = weight

    return weights

def preprocess(batch):
    in_text = batch["input"]
    weights = compute_loss_weights(in_text, keyword_list, expression_list)
    
    model_inputs = tokenizer(
        in_text, max_length=MAX_LENGTH, padding="max_length", truncation=True
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["output"], max_length=MAX_LENGTH, padding="max_length", truncation=True
        )["input_ids"]
        
    model_inputs["labels"] = labels
    model_inputs["loss_weights"] = weights.tolist()
    return model_inputs

print("Procesando dataset (esto tomará un tiempo por los embeddings)...")
# Usamos .map con carga simple para evitar problemas de multiprocessing en Windows
train_tokenized = train_data.map(preprocess)

# --- ENTRENAMIENTO ---
class WeightedDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        batch = super().__call__(features)
        batch["loss_weights"] = torch.tensor(
            [f["loss_weights"] for f in features], dtype=torch.float32
        )
        return batch

collator = WeightedDataCollator(tokenizer, model=model)

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        loss_weights = inputs["loss_weights"].to(model.device)
        
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels
        )
        logits = outputs.logits
        
        # Shift logits y labels para calcular loss del siguiente token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Alinear pesos: Debemos quitar el primer peso para que coincida con shift_labels
        # loss_weights shape: [batch, 256], shift_labels shape: [batch, 255]
        shift_weights = loss_weights[..., 1:].contiguous()

        ce_loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        
        # Calcular loss por token
        token_losses = ce_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # Aplicar pesos (reshape necesario para multiplicar correctamente)
        token_losses = token_losses.view(shift_labels.size())
        weighted_loss = token_losses * shift_weights
        
        final_loss = weighted_loss.mean()
        
        return (final_loss, outputs) if return_outputs else final_loss

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    logging_steps=10,
    save_total_limit=2,
    fp16=True, # Usar FP16 acelera mucho en RTX 3000 series
    optim="paged_adamw_8bit", # Ahorra memoria VRAM
    remove_unused_columns=False 
)

trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    data_collator=collator,
)

print("Iniciando entrenamiento...")
trainer.train()

# Guardar
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"¡Entrenamiento finalizado! Modelo guardado en {OUTPUT_DIR}")