from transformers import TrainerCallback
import torch
import random
import psutil


class PrintExamplesCallback(TrainerCallback):
    """
    Every N steps, generate detoxified examples to visually monitor progress.
    Useful for debugging seq2seq behaviour.
    """

    def __init__(self, tokenizer, every_steps=100):
        self.tokenizer = tokenizer
        self.every_steps = every_steps
        self.trainer = None  # se rellenará en on_init_end

    def on_init_end(self, args, state, control, **kwargs):
        # Aquí sí viene el trainer en kwargs
        self.trainer = kwargs.get("trainer", None)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return control
        if state.global_step % self.every_steps != 0:
            return control

        # Cogemos un ejemplo aleatorio del train_dataset
        ds = self.trainer.train_dataset
        idx = random.randint(0, len(ds) - 1)
        sample = ds[idx]

        input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(self.trainer.model.device)

        # Texto tóxico (input)
        toxic_text = self.tokenizer.decode(sample["input_ids"], skip_special_tokens=True)

        # Generamos la frase detoxificada
        with torch.no_grad():
            pred = self.trainer.model.generate(
                input_ids,
                max_length=128,
                num_beams=4,
            )
        neutral_text = self.tokenizer.decode(pred[0], skip_special_tokens=True)

        print("\n--- SAMPLE DETOXIFICATION ---")
        print("Toxic:   ", toxic_text)
        print("Neutral: ", neutral_text)
        print("--------------------------------\n")

        return control


class PerplexityCallback(TrainerCallback):
    """Computes perplexity at the end of each evaluation."""
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if "eval_loss" in metrics:
            ppl = torch.exp(torch.tensor(metrics["eval_loss"]))
            print(f"Perplexity: {ppl:.4f}")
        return control


class MemoryCallback(TrainerCallback):
    """Monitor RAM usage every epoch."""
    def on_epoch_end(self, args, state, control, **kwargs):
        mem = psutil.virtual_memory().percent
        print(f"RAM usage: {mem:.2f}%")
        return control


class GradNormCallback(TrainerCallback):
    """Monitor gradient norms every N steps."""
    def __init__(self, every_steps=100):
        self.every_steps = every_steps
        self.trainer = None

    def on_init_end(self, args, state, control, **kwargs):
        self.trainer = kwargs.get("trainer", None)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return control
        if state.global_step % self.every_steps != 0:
            return control

        total_norm = 0.0
        for p in self.trainer.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5
        print(f"Gradient norm: {total_norm:.4f}")
        return control


class LearningRateCallback(TrainerCallback):
    """Print the LR at the end of each epoch."""
    def __init__(self):
        self.trainer = None

    def on_init_end(self, args, state, control, **kwargs):
        self.trainer = kwargs.get("trainer", None)
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return control
        # _get_learning_rate es interno, pero vale para debugging
        lr = self.trainer._get_learning_rate()
        print(f"Learning Rate: {lr}")
        return control
