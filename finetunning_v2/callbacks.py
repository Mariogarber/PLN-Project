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

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_steps != 0:
            return

        trainer = kwargs["trainer"]
        batch = kwargs["train_dataloader"].dataset[random.randint(0, len(kwargs["train_dataloader"].dataset) - 1)]

        # Extract toxic
        text = trainer.tokenizer.decode(batch["input_ids"], skip_special_tokens=True)

        # Generate detox
        pred = trainer.model.generate(
            batch["input_ids"].unsqueeze(0),
            max_length=128,
            num_beams=4
        )
        det = trainer.tokenizer.decode(pred[0], skip_special_tokens=True)

        print("\n--- SAMPLE DETOXIFICATION ---")
        print("Toxic:   ", text)
        print("Neutral: ", det)
        print("--------------------------------\n")


class PerplexityCallback(TrainerCallback):
    """Computes perplexity at the end of each epoch."""
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if "eval_loss" in metrics:
            ppl = torch.exp(torch.tensor(metrics["eval_loss"]))
            print(f"Perplexity: {ppl:.4f}")


class MemoryCallback(TrainerCallback):
    """Monitor RAM usage every epoch."""
    def on_epoch_end(self, args, state, control, **kwargs):
        mem = psutil.virtual_memory().percent
        print(f"RAM usage: {mem:.2f}%")

        
class GradNormCallback(TrainerCallback):
    """Monitor gradient norms every N steps."""
    def __init__(self, every_steps=100):
        self.every_steps = every_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_steps != 0:
            return

        trainer = kwargs["trainer"]
        total_norm = 0.0
        for p in trainer.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5
        print(f"Gradient norm: {total_norm:.4f}")


class LearningRateCallback(TrainerCallback):
    """Print the LR at the end of each epoch."""
    def on_epoch_end(self, args, state, control, **kwargs):
        lr = kwargs["trainer"]._get_learning_rate()
        print(f"Learning Rate: {lr}")
