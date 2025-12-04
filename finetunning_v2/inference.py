from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

def batch_inference(model, tokenizer, test_dataset, collator):
    loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collator
    )

    preds = []
    golds = []
    toxics = []

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=4
        )

        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # FIX GOLD LABELS
        labels = batch["labels"].clone()
        labels[labels == -100] = tokenizer.pad_token_id
        decoded_golds = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_toxics = tokenizer.batch_decode(input_ids, skip_special_tokens=True)


        preds.extend(decoded_preds)
        golds.extend(decoded_golds)
        toxics.extend(decoded_toxics)

    df = pd.DataFrame({
        "toxic": toxics,
        "gold_neutral": golds,
        "predicted_neutral": preds
    })

    df.to_csv("test_results_batch.csv", index=False)
    print("Saved batch predictions to test_results_batch.csv")
