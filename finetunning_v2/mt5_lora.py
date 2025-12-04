from transformers import MT5ForConditionalGeneration, MT5Config
from peft import LoraConfig, get_peft_model, TaskType


def build_mt5_lora(
    base_model_name: str = "google/mt5-base",
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules = None,
    device: str = "cpu",
):
    """
    Build an mT5 model with LoRA adapters for detoxification.

    Parameters
    ----------
    base_model_name:
        HuggingFace model id, e.g. "google/mt5-base".

    r, lora_alpha, lora_dropout:
        LoRA hyperparameters.

    target_modules:
        Which modules to apply LoRA to. If None, a good default
        for T5/mT5 attention layers is used.

    device:
        "cpu" or "cuda". In tu caso, "cpu".

    Returns
    -------
    model: PEFT-wrapped MT5ForConditionalGeneration
    """

    # Load base mT5
    model = MT5ForConditionalGeneration.from_pretrained(base_model_name)

    # Good default for T5-style attention:
    # - "q", "k", "v", "o" (attention projections)
    if target_modules is None:
        target_modules = ["q", "k", "v", "o"]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.to(device)

    # Optional: print trainable params
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"Total parameters:     {total/1e6:.2f}M")
    print(f"Trainable parameters: {trainable/1e6:.2f}M "
          f"({100*trainable/total:.2f}%)")

    return model
