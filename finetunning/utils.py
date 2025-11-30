from torch.utils.data import Dataset
import pandas as pd
from typing import List
import pandas as pd
from bert_score import score as bertscore
import sacrebleu
from rouge_score import rouge_scorer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging

TOKENIZER = T5Tokenizer.from_pretrained("google/mt5-base")
MODEL = T5ForConditionalGeneration.from_pretrained("google/mt5-base")

def detoxify_text(toxic_text, max_length=512):
    """Generate detoxified version of toxic text"""
    # Add task prefix
    input_text = f"{toxic_text}"
    
    # Tokenize input
    inputs = TOKENIZER(
        input_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True
    ).to(model_device)
    
    # Generate output
    with torch.no_grad():
        outputs = model_lora.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
            pad_token_id=TOKENIZER.pad_token_id,
            eos_token_id=TOKENIZER.eos_token_id,
            bad_words_ids=bad_words
        )
    
    # Decode output
    generated_text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def check_trainable_params(model):
    """Check and print the number of trainable parameters in the model"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")
    return trainable_params, total_params