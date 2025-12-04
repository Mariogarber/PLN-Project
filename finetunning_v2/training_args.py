from transformers import Seq2SeqTrainingArguments

STANDAR_ARGS = Seq2SeqTrainingArguments(
    output_dir="./mt5_detox_baseline",
    overwrite_output_dir=True,

    # Optimización
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=10,

    # Logging
    logging_dir="./logs",
    logging_steps=5,
    log_level="info",

    # Checkpoints
    save_total_limit=4,
    save_strategy="epoch",
    evaluation_strategy="epoch",

    # Decoding params for evaluation
    predict_with_generate=True,
    generation_max_length=128,

    # Disable fp16 because you're training on CPU
    fp16=False,

    # Metrics
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)
