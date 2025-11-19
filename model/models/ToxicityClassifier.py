import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm


class ToxicityClassifier(nn.Module):
    """
    XLM-RoBERTa + head de clasificación binaria personalizada.
    """
    def __init__(self, model_name="xlm-roberta-base", hidden_dropout=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size  # 768 para base

        # 🔥 Head personalizada
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size, 1)  # salida binaria
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]

        logits = self.classifier(sequence_output)  # [batch, seq_len, 1]
        logits = logits.squeeze(-1)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            active_loss = labels != -100
            active_logits = logits[active_loss]
            active_labels = labels[active_loss].float()
            loss = loss_fct(active_logits, active_labels)
            return loss, logits
        else:
            probs = self.sigmoid(logits)
            return probs

    def train_model(self, train_dataset, test_dataset, epochs=3, lr=2e-5, device='cuda' if torch.cuda.is_available() else 'cpu'):

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        total_steps = len(train_dataset) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for batch in tqdm(train_dataset, desc=f"Epoch {epoch+1}/{epochs}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                loss, _ = self(input_ids, attention_mask, labels=labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataset)
            print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")

            # Evaluación rápida
            self.eval()
            with torch.no_grad():
                total_eval_loss = 0
                for batch in test_dataset:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    loss, _ = self(input_ids, attention_mask, labels=labels)
                    total_eval_loss += loss.item()
            print(f"Epoch {epoch+1} - Eval Loss: {total_eval_loss / len(test_dataset):.4f}")

        torch.save(self.state_dict(), "toxic_classifier_custom.pt")
        print("✅ Modelo guardado: toxic_classifier_custom.pt")
        return self