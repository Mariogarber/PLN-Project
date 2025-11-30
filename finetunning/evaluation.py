import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')
from utils import detoxify_text
import pandas as pd

class DetoxificationEvaluator:
    def __init__(self, bert_model='all-MiniLM-L6-v2'):
        """
        Initialize the evaluator with ROUGE, BLEU, and BERT similarity metrics.
        
        Args:
            bert_model: The sentence transformer model to use for BERT similarity
        """
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bert_model = SentenceTransformer(bert_model)
        self.smoothing = SmoothingFunction()
    
    def compute_rouge(self, reference, prediction):
        """Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)"""
        scores = self.rouge_scorer.score(reference, prediction)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def compute_bleu(self, reference, prediction):
        """Compute BLEU score with smoothing"""
        ref_tokens = reference.split()
        pred_tokens = prediction.split()
        
        # Use smoothing to handle cases with no n-gram overlap
        bleu = sentence_bleu(
            [ref_tokens], 
            pred_tokens, 
            smoothing_function=self.smoothing.method1
        )
        return bleu
    
    def compute_bert_similarity(self, text1, text2):
        """Compute BERT-based semantic similarity using sentence embeddings"""
        embeddings = self.bert_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity
    
    def compute_input_output_similarity(self, input_text, output_text):
        """
        Compute similarity ratio between input and output.
        Uses both character-level and semantic similarity.
        """
        # Character-level similarity (like difflib)
        char_similarity = SequenceMatcher(None, input_text, output_text).ratio()
        
        # Semantic similarity using BERT
        semantic_similarity = self.compute_bert_similarity(input_text, output_text)
        
        return {
            'character_similarity': char_similarity,
            'semantic_similarity': semantic_similarity,
            'average_similarity': (char_similarity + semantic_similarity) / 2
        }
    
    def evaluate_single(self, input_text, reference_text, prediction_text):
        """
        Evaluate a single prediction against reference and input.
        
        Args:
            input_text: The original toxic input
            reference_text: The ground truth detoxified text
            prediction_text: The model's predicted detoxified text
        
        Returns:
            Dictionary with all evaluation metrics
        """
        rouge_scores = self.compute_rouge(reference_text, prediction_text)
        bleu_score = self.compute_bleu(reference_text, prediction_text)
        bert_similarity = self.compute_bert_similarity(reference_text, prediction_text)
        input_output_sim = self.compute_input_output_similarity(input_text, prediction_text)
        
        return {
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL'],
            'bleu': bleu_score,
            'bert_similarity': bert_similarity,
            'input_output_char_similarity': input_output_sim['character_similarity'],
            'input_output_semantic_similarity': input_output_sim['semantic_similarity'],
            'input_output_avg_similarity': input_output_sim['average_similarity']
        }
    
    def evaluate_batch(self, inputs, references, predictions):
        """
        Evaluate a batch of predictions.
        
        Args:
            inputs: List of original toxic inputs
            references: List of ground truth detoxified texts
            predictions: List of model predictions
        
        Returns:
            Dictionary with averaged metrics and individual scores
        """
        results = []
        
        for inp, ref, pred in zip(inputs, references, predictions):
            result = self.evaluate_single(inp, ref, pred)
            results.append(result)
        
        # Compute average metrics
        avg_metrics = {
            metric: np.mean([r[metric] for r in results])
            for metric in results[0].keys()
        }
        
        return {
            'average_metrics': avg_metrics,
            'individual_scores': results
        }
    
    def evaluate_dataset(self, test_dataset, model, max_length=512):
        """
        Evaluate the model on a test dataset.
        
        Args:
            test_dataset: Dataset with 'toxic_sentence' and 'neutral_sentence' fields
            model: The detoxification model to evaluate
            max_length: Maximum sequence length for generation
        Returns:
            Dictionary with evaluation results
        """
        results = pd.DataFrame()
        for i, row in test_dataset.iterrows():
            input_text = row['toxic_sentence']
            reference_text = row['neutral_sentence']
            prediction_text = detoxify_text(input_text, max_length=max_length)
            
            eval_result = self.evaluate_single(input_text, reference_text, prediction_text)
            results = pd.concat([results, pd.DataFrame([eval_result])], ignore_index=True)

        average_metrics = results.mean().to_dict()
        return {
            'average_metrics': average_metrics,
            'scores': results
        }


    def print_summary(self, evaluation_results):
        """Print a formatted summary of evaluation results"""
        metrics = evaluation_results['average_metrics']
        
        print("=" * 60)
        print("DETOXIFICATION EVALUATION SUMMARY")
        print("=" * 60)
        print("\nContent Preservation Metrics (vs Reference):")
        print(f"  ROUGE-1:        {metrics['rouge1']:.4f}")
        print(f"  ROUGE-2:        {metrics['rouge2']:.4f}")
        print(f"  ROUGE-L:        {metrics['rougeL']:.4f}")
        print(f"  BLEU:           {metrics['bleu']:.4f}")
        print(f"  BERT Similarity: {metrics['bert_similarity']:.4f}")
        
        print("\nInput-Output Similarity (Content Retention):")
        print(f"  Character-level: {metrics['input_output_char_similarity']:.4f}")
        print(f"  Semantic:        {metrics['input_output_semantic_similarity']:.4f}")
        print(f"  Average:         {metrics['input_output_avg_similarity']:.4f}")
        print("=" * 60)
