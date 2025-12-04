from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any
import torch
from transformers import PreTrainedTokenizerBase

@dataclass
class DataCollatorT5:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = "longest"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract labels before padding
        labels = [example["labels"] for example in batch]

        # Pad encoder inputs
        batch = self.tokenizer.pad(
            batch,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Pad labels manually (because tokenizer.pad does not touch labels)
        max_label_len = max(len(lbl) for lbl in labels)

        padded_labels = []
        for lbl in labels:
            remainder = max_label_len - len(lbl)
            padded = lbl + [-100] * remainder  # T5 masking
            padded_labels.append(padded)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch
