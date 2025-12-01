import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')
from .utils import detoxify_text
import pandas as pd
import torch
from typing import List, Dict, Union, Optional
from transformers import AutoTokenizer
from datasets import Dataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


class DetoxificationInference:
    """
    🔬 Comprehensive inference class for detoxification models
    
    This class handles inference on test datasets with proper preprocessing,
    batch processing, and comprehensive result analysis.
    """
    
    def __init__(
        self, 
        model, 
        tokenizer: AutoTokenizer,
        prefix: str = "detoxify and rewrite: ",
        device: str = "auto",
        max_length: int = 512,
        max_new_tokens: int = 128
    ):
        """
        Initialize the inference handler
        
        Args:
            model: The fine-tuned model for inference
            tokenizer: Tokenizer for text processing
            prefix: Task prefix to add to input sentences
            device: Device to run inference on ("auto", "cuda", "cpu")
            max_length: Maximum input sequence length
            max_new_tokens: Maximum new tokens to generate
        """
        self.model = model
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"🎯 DetoxificationInference initialized")
        logger.info(f"📋 Prefix: '{self.prefix}'")
        logger.info(f"💻 Device: {self.device}")
        logger.info(f"📏 Max length: {self.max_length}")
        logger.info(f"🔢 Max new tokens: {self.max_new_tokens}")
    
    def preprocess_input(self, text: str) -> str:
        """
        Preprocess input text by adding task prefix
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text with prefix
        """
        if not text.startswith(self.prefix):
            return self.prefix + text
        return text
    
    def preprocess_dataset(self, dataset: Union[Dataset, List[Dict], pd.DataFrame]) -> List[Dict]:
        """
        Preprocess dataset for inference
        
        Args:
            dataset: Test dataset (HuggingFace Dataset, list of dicts, or DataFrame)
            
        Returns:
            List of preprocessed samples
        """
        processed_samples = []
        
        # Handle different input types
        if isinstance(dataset, pd.DataFrame):
            dataset_items = dataset.to_dict('records')
        elif hasattr(dataset, '__iter__') and not isinstance(dataset, str):
            # Handle HuggingFace Dataset or list
            dataset_items = list(dataset)
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")
        
        for item in dataset_items:
            # Handle different field names
            toxic_text = (
                item.get('toxic_sentence') or 
                item.get('toxic_text') or 
                item.get('input') or 
                item.get('text', '')
            )
            
            neutral_text = (
                item.get('neutral_sentence') or 
                item.get('neutral_text') or 
                item.get('target') or 
                item.get('reference', '')
            )
            
            language = item.get('language', 'unknown')
            
            # Preprocess the toxic text
            preprocessed_input = self.preprocess_input(toxic_text)
            
            processed_samples.append({
                'original_toxic': toxic_text,
                'preprocessed_input': preprocessed_input,
                'reference_neutral': neutral_text,
                'language': language,
                'sample_id': len(processed_samples)
            })
        
        logger.info(f"📊 Preprocessed {len(processed_samples)} samples")
        return processed_samples
    
    def generate_single(self, input_text: str) -> Dict[str, str]:
        """
        Generate detoxified text for a single input
        
        Args:
            input_text: Preprocessed input text
            
        Returns:
            Dictionary with generation results
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate with the model
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_beams=1  # Greedy for consistency
                )
            
            # Decode the output
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Remove the input prefix from generated text if present
            if generated_text.startswith(input_text):
                generated_text = generated_text[len(input_text):].strip()
            elif generated_text.startswith(self.prefix):
                # Remove prefix if model included it in output
                generated_text = generated_text[len(self.prefix):].strip()
            
            return {
                'success': True,
                'generated_text': generated_text,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return {
                'success': False,
                'generated_text': f"[Generation Error: {str(e)}]",
                'error': str(e)
            }
    
    def run_inference(
        self, 
        dataset: Union[Dataset, List[Dict], pd.DataFrame],
        batch_size: int = 8,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Run inference on entire dataset
        
        Args:
            dataset: Test dataset
            batch_size: Batch size for processing (currently processes one by one)
            show_progress: Whether to show progress
            
        Returns:
            List of inference results
        """
        logger.info("🚀 Starting inference process...")
        
        # Preprocess dataset
        processed_samples = self.preprocess_dataset(dataset)
        
        results = []
        total_samples = len(processed_samples)
        
        for i, sample in enumerate(processed_samples):
            if show_progress and (i % 10 == 0 or i == total_samples - 1):
                print(f"🔬 Processing sample {i+1}/{total_samples} ({((i+1)/total_samples)*100:.1f}%)")
            
            # Generate detoxified text
            generation_result = self.generate_single(sample['preprocessed_input'])
            
            # Combine all information
            result = {
                **sample,  # Include all original sample data
                **generation_result,  # Include generation results
                'input_length': len(sample['preprocessed_input']),
                'output_length': len(generation_result['generated_text']) if generation_result['success'] else 0
            }
            
            results.append(result)
        
        logger.info(f"✅ Inference completed on {total_samples} samples")
        
        # Summary statistics
        successful_generations = sum(1 for r in results if r['success'])
        failure_rate = (total_samples - successful_generations) / total_samples * 100
        
        logger.info(f"📊 Success rate: {successful_generations}/{total_samples} ({100-failure_rate:.1f}%)")
        if failure_rate > 0:
            logger.warning(f"⚠️ Failure rate: {failure_rate:.1f}%")
        
        return results
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """
        Analyze inference results and provide statistics
        
        Args:
            results: List of inference results
            
        Returns:
            Analysis dictionary with statistics
        """
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            return {"error": "No successful generations to analyze"}
        
        analysis = {
            'total_samples': len(results),
            'successful_samples': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'average_input_length': np.mean([r['input_length'] for r in successful_results]),
            'average_output_length': np.mean([r['output_length'] for r in successful_results]),
            'languages': {},
            'generation_errors': []
        }
        
        # Language distribution
        for result in results:
            lang = result.get('language', 'unknown')
            analysis['languages'][lang] = analysis['languages'].get(lang, 0) + 1
        
        # Collect errors
        for result in results:
            if not result['success'] and result.get('error'):
                analysis['generation_errors'].append(result['error'])
        
        return analysis
    
    def save_results(
        self, 
        results: List[Dict], 
        output_path: str,
        include_analysis: bool = True
    ):
        """
        Save inference results to file
        
        Args:
            results: Inference results
            output_path: Path to save results (CSV format)
            include_analysis: Whether to include analysis summary
        """
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        
        # Save main results
        df_results.to_csv(output_path, index=False)
        logger.info(f"💾 Results saved to: {output_path}")
        
        if include_analysis:
            analysis = self.analyze_results(results)
            analysis_path = output_path.replace('.csv', '_analysis.txt')
            
            with open(analysis_path, 'w', encoding='utf-8') as f:
                f.write("🔬 INFERENCE ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total samples: {analysis['total_samples']}\n")
                f.write(f"Successful generations: {analysis['successful_samples']}\n")
                f.write(f"Success rate: {analysis['success_rate']:.2%}\n")
                f.write(f"Average input length: {analysis['average_input_length']:.1f} chars\n")
                f.write(f"Average output length: {analysis['average_output_length']:.1f} chars\n")
                f.write(f"\nLanguage distribution:\n")
                for lang, count in analysis['languages'].items():
                    f.write(f"  {lang}: {count} samples\n")
                
                if analysis['generation_errors']:
                    f.write(f"\nGeneration errors ({len(analysis['generation_errors'])}):\n")
                    for error in analysis['generation_errors'][:5]:  # Show first 5
                        f.write(f"  • {error}\n")
            
            logger.info(f"📊 Analysis saved to: {analysis_path}")
    
    def print_sample_results(self, results: List[Dict], n_samples: int = 3):
        """
        Print sample results for inspection
        
        Args:
            results: Inference results
            n_samples: Number of samples to display
        """
        print("\n" + "🔍" * 20 + " SAMPLE INFERENCE RESULTS " + "🔍" * 20)
        
        successful_results = [r for r in results if r['success']]
        sample_results = successful_results[:n_samples]
        
        for i, result in enumerate(sample_results, 1):
            print(f"\n📝 Sample {i}:")
            print(f"🔴 Original toxic:     {result['original_toxic']}")
            print(f"🤖 Model prediction:   {result['generated_text']}")
            print(f"✅ Reference neutral:  {result['reference_neutral']}")
            print(f"🌍 Language:           {result['language']}")
            print(f"📏 Input length:       {result['input_length']} chars")
            print(f"📏 Output length:      {result['output_length']} chars")
            print("-" * 80)
        
        print(f"\n📊 Showing {len(sample_results)} of {len(results)} total results")
        print("🔍" * 70)
