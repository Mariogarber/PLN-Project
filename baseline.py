import logging

import torch

from model.dataset.Datasets import ToxicNonToxicDataset

import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
from bert_score import score as bertscore
import sacrebleu
from rouge_score import rouge_scorer

SYSTEM_PROMPT = """Eres un modelo de lenguaje avanzado entrenado para detoxificar texto.
Tu tarea es transformar oraciones tóxicas en versiones neutrales, manteniendo el significado original
tanto como sea posible. Evita cualquier lenguaje ofensivo o inapropiado en tus respuestas. Detoxifica este texto:
"""

TOKENIZER = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
MODEL = AutoModelForMaskedLM.from_pretrained("FacebookAI/xlm-roberta-base")

LOGGER = logging.getLogger("baseline_logger")

def load_data() -> ToxicNonToxicDataset:
    lang = ['en', 'am', 'ar', 'de', 'es', 'hi', 'ru', 'uk', 'zh']

    url = "https://raw.githubusercontent.com/Mariogarber/PLN-Project/main/dataset/toxic_nontoxic/multilingual_paradetox_idioma.parquet"

    urls = [url.replace("idioma", l) for l in lang]

    dataset = ToxicNonToxicDataset(parquet_paths=urls)
    LOGGER.info(f"Loaded dataset with {len(dataset)} samples.")
    return dataset


if __name__ == "__main__":
    logging.basicConfig(filename='baseline_evaluation.log', level=logging.INFO)
    #store the logs in a file
    dataset = load_data()
    result_df = pd.DataFrame(columns=['toxic_text', 'detoxified_text', 'non_toxic_text', 'rouge1', 'rougeL', 'bleu', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1'])
    for toxic_text, non_toxic_text, lang in dataset:
        prompt = SYSTEM_PROMPT + toxic_text
        inputs = TOKENIZER(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = MODEL(**inputs)
        LOGGER.info(f"Toxic: {toxic_text}\nDetoxified: {outputs}\nCorrect Non-Toxic: {non_toxic_text}\n")

        # ROUGE Evaluation 
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(non_toxic_text, str(outputs))
        LOGGER.info(f"ROUGE Scores: {rouge_scores}")

        # BLEU Evaluation
        bleu = sacrebleu.corpus_bleu([str(outputs)], [[non_toxic_text]])
        LOGGER.info(f"BLEU Score: {bleu.score}")

        # BERTScore Evaluation
        P, R, F1 = bertscore([str(outputs)], [non_toxic_text], lang=lang, rescale_with_baseline=True)
        LOGGER.info(f"BERTScore - Precision: {P}, Recall: {R}, F1: {F1}")

        new_row = pd.DataFrame([{
            'toxic_text': toxic_text,
            'detoxified_text': str(outputs),
            'non_toxic_text': non_toxic_text,
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure,
            'bleu': bleu.score,
            'bertscore_precision': P.item(),
            'bertscore_recall': R.item(),
            'bertscore_f1': F1.item()
        }])
        result_df = pd.concat([result_df, new_row], ignore_index=True)
        
    
    result_df.to_csv('baseline_evaluation_results.csv', index=False)
    LOGGER.info("Evaluation results saved to baseline_evaluation_results.csv")