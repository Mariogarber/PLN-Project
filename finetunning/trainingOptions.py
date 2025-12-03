from transformers import TrainerCallback, TrainingArguments, DataCollatorForSeq2Seq
import numpy as np
from transformers import Trainer

DEFAULT_TRAINING_ARGS = TrainingArguments(
    output_dir="./mt5-detoxify-qlora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=10,
    learning_rate=1e-4,
    bf16=True,
    logging_steps=200,
    save_steps=500,
    eval_strategy="steps",
    eval_steps=200,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
)

def build_training_args(output_name,
                        real_batch_size,
                        gradient_accumulation_steps,
                        num_train_epochs,
                        learning_rate,
                        logging_steps=200
                        ):
    """
    Build TrainingArguments with custom parameters for finetuning mT5 using QLoRA.

    Parameters
    ----------
    output_name : str
        Nombre del directorio donde se guardarán los checkpoints y resultados.
        Suele ser el nombre del experimento o del modelo final.

    real_batch_size : int
        Tamaño del batch por dispositivo.
        
        Recomendaciones:
        - En QLoRA y modelos T5/mT5 suele ser pequeño por restricciones de memoria.
        - Valores típicos: 1-4.
        - Lo ideal es compensarlo con gradient_accumulation_steps para obtener
          un "effective batch size" estable de 32-64.

    gradient_accumulation_steps : int
        Número de pasos de acumulación antes de aplicar un gradiente.
        Esto permite simular batchs grandes aunque la GPU sea limitada.

        Recomendaciones:
        - Si real_batch_size es bajo, aumentar este valor.
        - Effective batch = real_batch_size * gradient_accumulation_steps.
          Para mT5, apuntar a un effective batch de 32-64.
        - Ejemplos:
            real_batch_size = 2 → grad_accum = 16  → effective batch = 32
            real_batch_size = 2 → grad_accum = 32  → effective batch = 64

    num_train_epochs : int
        Número de épocas de entrenamiento.

        Recomendaciones:
        - En tareas de detoxificación y style transfer, demasiadas épocas causan
          overfitting y "identity collapse" (el modelo aprende a copiar el input).
        - Valores recomendados: 2-4.
        - No usar más de 5 a menos que el dataset sea grande (100k+ ejemplos).

    learning_rate : float
        Tasa de aprendizaje para finetuning con QLoRA.

        Recomendaciones:
        - T5/mT5 son muy sensibles a tasas altas.
        - Valores típicos y estables:
            2e-5  → estable y recomendado
            5e-5  → más agresivo, puede funcionar si el dataset es grande
            1e-4  → demasiado alto; suele inducir inestabilidad o colapso
        - En QLoRA siempre preferir un LR bajo.

    Notas adicionales
    -----------------
    - bf16=True mejora la estabilidad y velocidad en GPUs modernas (A100, 4090, etc.).
    - gradient_checkpointing=True reduce memoria pero hace el entrenamiento más lento.
    - remove_unused_columns=False es necesario para modelos seq2seq como T5/mT5.
    - eval_strategy="steps" permite detectar temprano si el modelo empieza a copiar.
    - max_grad_norm=1.0 controla la magnitud de los gradientes y evita inestabilidad.
    
    Returns
    -------
    TrainingArguments
        Objeto de HuggingFace con todos los parámetros configurados.
    """

    training_args = TrainingArguments(
        output_dir=f"./{output_name}",
        per_device_train_batch_size=real_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=logging_steps,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=200,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )
    return training_args