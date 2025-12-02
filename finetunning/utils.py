from torch.utils.data import Dataset
import pandas as pd
from typing import List, Set, Union, Tuple
import pandas as pd
from bert_score import score as bertscore
import sacrebleu
from rouge_score import rouge_scorer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging
import torch
import numpy as np
from collections import defaultdict

TOKENIZER = T5Tokenizer.from_pretrained("google/mt5-base")
MODEL = T5ForConditionalGeneration.from_pretrained("google/mt5-base")


class ToxicMaskGenerator:
    """
    Word-level toxic mask generator that matches actual words in text.
    This approach is more robust and language-agnostic than token-based matching.
    """
    
    def __init__(
        self, 
        tokenizer, 
        forbidden_words: List[str] = None,
        mask_strategy: str = "toxic_only",
        case_sensitive: bool = False,
        match_partial: bool = True
    ):
        """
        Initialize the toxic mask generator.
        
        Args:
            tokenizer: The tokenizer to use
            forbidden_words: List of forbidden words/phrases
            mask_strategy: Strategy for creating masks
                - "toxic_only": Only train on positions with toxic words
                - "toxic_context": Train on toxic words + surrounding context
                - "inverse_toxic": Train on everything EXCEPT toxic words
            case_sensitive: Whether word matching is case sensitive
            match_partial: Whether to match partial word occurrences
        """
        self.tokenizer = tokenizer
        self.forbidden_words = [word.strip() for word in (forbidden_words or []) if word.strip()]
        self.mask_strategy = mask_strategy
        self.case_sensitive = case_sensitive
        self.match_partial = match_partial
        
        # Prepare forbidden words for matching
        if not self.case_sensitive:
            self.forbidden_words = [word.lower() for word in self.forbidden_words]
        
        print(f"🎭 ToxicMaskGenerator initialized")
        print(f"🚫 Tracking {len(self.forbidden_words)} forbidden words")
        print(f"📋 Strategy: {mask_strategy}")
        print(f"📋 Case sensitive: {case_sensitive}")
        print(f"📋 Partial matching: {match_partial}")
    
    def create_toxic_token_mask(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        context_window: int = 2
    ) -> torch.Tensor:
        """
        Create a training mask focusing on toxic words using word-level matching.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Label token IDs [batch_size, seq_len] 
            context_window: Number of tokens around toxic words to include
            
        Returns:
            mask: Boolean tensor [batch_size, seq_len] where True = train on this position
        """
        return self.create_word_level_mask(input_ids, labels, context_window)

    
    def create_word_level_mask(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        context_window: int = 2
    ) -> torch.Tensor:
        """
        Create a training mask focusing on toxic words using word-level matching.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Label token IDs [batch_size, seq_len] 
            context_window: Number of tokens around toxic words to include
            
        Returns:
            mask: Boolean tensor [batch_size, seq_len] where True = train on this position
        """
        batch_size, seq_len = labels.shape
        device = labels.device
        
        # Initialize mask
        mask = torch.zeros_like(labels, dtype=torch.bool, device=device)
        
        # Process each sample in the batch
        for batch_idx in range(batch_size):
            sample_labels = labels[batch_idx]
            
            # Decode the target text to find toxic words
            valid_positions = (sample_labels != -100) & (sample_labels != self.tokenizer.pad_token_id)
            valid_tokens = sample_labels[valid_positions]
            
            try:
                decoded_text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
                if not self.case_sensitive:
                    decoded_text = decoded_text.lower()
                
                # Find toxic word positions in the decoded text
                toxic_word_positions = self._find_toxic_words_in_text(decoded_text)
                
                if toxic_word_positions:
                    # Map text positions back to token positions
                    token_mask = self._map_text_positions_to_tokens(
                        valid_tokens, decoded_text, toxic_word_positions, batch_idx
                    )
                    
                    # Apply masking strategy
                    if self.mask_strategy == "toxic_only":
                        sample_mask = torch.zeros_like(sample_labels, dtype=torch.bool, device=device)
                        sample_mask[valid_positions] = token_mask
                    elif self.mask_strategy == "toxic_context":
                        base_mask = torch.zeros_like(sample_labels, dtype=torch.bool, device=device)
                        base_mask[valid_positions] = token_mask
                        sample_mask = self._expand_mask_with_context_word_level(
                            base_mask, context_window
                        )
                    elif self.mask_strategy == "inverse_toxic":
                        sample_mask = valid_positions.clone()
                        sample_mask[valid_positions] = ~token_mask
                    else:
                        raise ValueError(f"Unknown mask strategy: {self.mask_strategy}")
                    
                    mask[batch_idx] = sample_mask
                    
            except Exception as e:
                print(f"Warning: Failed to process sample {batch_idx}: {e}")
                continue
        
        # Ensure we never train on invalid positions
        valid_positions = (labels != -100) & (labels != self.tokenizer.pad_token_id)
        mask = mask & valid_positions
        
        return mask
    
    def _find_toxic_words_in_text(self, text: str) -> List[tuple]:
        """
        Find toxic words in decoded text and return their positions.
        
        Args:
            text: Decoded text to search
            
        Returns:
            List of (start_pos, end_pos, word) tuples
        """
        import re
        toxic_positions = []
        
        for word in self.forbidden_words:
            if self.match_partial:
                # Find all occurrences of the word (including as substring)
                pattern = re.escape(word)
                matches = re.finditer(pattern, text, re.IGNORECASE if not self.case_sensitive else 0)
            else:
                # Find only whole word matches
                pattern = r'\b' + re.escape(word) + r'\b'
                matches = re.finditer(pattern, text, re.IGNORECASE if not self.case_sensitive else 0)
            
            for match in matches:
                toxic_positions.append((match.start(), match.end(), word))
        
        return sorted(toxic_positions, key=lambda x: x[0])  # Sort by start position
    
    def _map_text_positions_to_tokens(
        self, 
        valid_tokens: torch.Tensor, 
        decoded_text: str, 
        toxic_positions: List[tuple],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Map character positions in decoded text back to token positions.
        This is approximate but works well in practice.
        
        Args:
            valid_tokens: Valid token IDs for this sample
            decoded_text: The decoded text
            toxic_positions: List of (start, end, word) tuples
            batch_idx: Batch index for this sample
            
        Returns:
            Boolean mask indicating toxic token positions
        """
        token_mask = torch.zeros_like(valid_tokens, dtype=torch.bool)
        
        if not toxic_positions:
            return token_mask
        
        # Create character-to-token mapping by decoding each token
        char_to_token_map = []
        current_char = 0
        
        for token_idx, token_id in enumerate(valid_tokens):
            try:
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                if not self.case_sensitive:
                    token_text = token_text.lower()
                
                token_length = len(token_text)
                
                # Map character positions to this token
                for _ in range(token_length):
                    if current_char < len(decoded_text):
                        char_to_token_map.append(token_idx)
                        current_char += 1
                        
            except Exception:
                # Skip problematic tokens
                continue
        
        # Mark tokens that overlap with toxic positions
        for start_pos, end_pos, word in toxic_positions:
            for char_pos in range(start_pos, min(end_pos, len(char_to_token_map))):
                if char_pos < len(char_to_token_map):
                    token_idx = char_to_token_map[char_pos]
                    if token_idx < len(token_mask):
                        token_mask[token_idx] = True
        
        return token_mask
    
    def _expand_mask_with_context_word_level(
        self, 
        toxic_mask: torch.Tensor, 
        context_window: int
    ) -> torch.Tensor:
        """Expand toxic word mask to include surrounding context"""
        expanded_mask = toxic_mask.clone()
        
        # Expand left and right
        for i in range(1, context_window + 1):
            # Expand right
            if i < len(toxic_mask):
                right_expanded = torch.zeros_like(toxic_mask)
                right_expanded[:-i] = toxic_mask[i:]
                expanded_mask |= right_expanded
            
            # Expand left  
            if i < len(toxic_mask):
                left_expanded = torch.zeros_like(toxic_mask)
                left_expanded[i:] = toxic_mask[:-i]
                expanded_mask |= left_expanded
        
        return expanded_mask

    # Add compatibility method
    def create_toxic_token_mask(self, input_ids, labels, context_window=2):
        """Compatibility method that calls word-level masking"""
        return self.create_word_level_mask(input_ids, labels, context_window)



    def debug_toxicity_detection(
        self,
        labels: torch.Tensor,
        sample_size: int = 5
    ) -> dict:
        """
        Debug helper to check if toxic tokens are being detected properly.
        
        Args:
            labels: Label tensor to analyze
            sample_size: Number of samples to analyze in detail
            
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            'total_samples': labels.shape[0],
            'forbidden_token_ids': list(self.forbidden_token_ids)[:20],  # First 20 for display
            'samples_analyzed': [],
            'overall_stats': {
                'total_toxic_positions': 0,
                'samples_with_toxic': 0
            }
        }
        
        # Analyze each sample in batch
        for i in range(min(sample_size, labels.shape[0])):
            sample_labels = labels[i]
            valid_positions = (sample_labels != -100) & (sample_labels != self.tokenizer.pad_token_id)
            valid_tokens = sample_labels[valid_positions]
            
            # Check for toxic tokens in this sample
            toxic_found = []
            for token_id in valid_tokens:
                if token_id.item() in self.forbidden_token_ids:
                    toxic_found.append({
                        'token_id': token_id.item(),
                        'token_text': self.tokenizer.decode([token_id]),
                        'position': torch.where(sample_labels == token_id)[0].tolist()
                    })
            
            # Decode full text for context
            try:
                full_text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
            except:
                full_text = "[Unable to decode]"
            
            sample_info = {
                'sample_index': i,
                'total_valid_tokens': len(valid_tokens),
                'toxic_tokens_found': len(toxic_found),
                'toxic_details': toxic_found,
                'decoded_text': full_text[:100] + "..." if len(full_text) > 100 else full_text
            }
            
            debug_info['samples_analyzed'].append(sample_info)
            debug_info['overall_stats']['total_toxic_positions'] += len(toxic_found)
            if len(toxic_found) > 0:
                debug_info['overall_stats']['samples_with_toxic'] += 1
        
        return debug_info

# Add word-level statistics method to WordLevelToxicMaskGenerator
def get_word_mask_statistics(self, input_ids: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> dict:
    """Get statistics about the word-level generated mask"""
    batch_size, seq_len = labels.shape
    
    # Count valid positions (not -100 or padding)
    valid_positions = (labels != -100) & (labels != self.tokenizer.pad_token_id)
    total_valid = valid_positions.sum().item()
    
    # Count masked positions  
    masked_positions = mask.sum().item()
    
    # Count toxic word occurrences in decoded text
    toxic_word_count = 0
    toxic_words_found = set()
    
    for batch_idx in range(batch_size):
        sample_labels = labels[batch_idx]
        valid_tokens = sample_labels[(sample_labels != -100) & (sample_labels != self.tokenizer.pad_token_id)]
        
        try:
            decoded_text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
            if not self.case_sensitive:
                decoded_text = decoded_text.lower()
            
            toxic_positions = self._find_toxic_words_in_text(decoded_text)
            toxic_word_count += len(toxic_positions)
            toxic_words_found.update([word for _, _, word in toxic_positions])
            
        except Exception:
            continue
    
    # Calculate percentages
    mask_ratio = masked_positions / max(total_valid, 1) * 100
    toxic_ratio = toxic_word_count / max(batch_size, 1) * 100
    
    return {
        'total_valid_positions': total_valid,
        'masked_positions': masked_positions,
        'toxic_word_occurrences': toxic_word_count,
        'unique_toxic_words_found': len(toxic_words_found),
        'toxic_words_found': list(toxic_words_found),
        'mask_ratio_percent': mask_ratio,
        'toxic_words_per_sample': toxic_ratio,
        'strategy': self.mask_strategy
    }

# Bind the method to the class
ToxicMaskGenerator.get_mask_statistics = get_word_mask_statistics

def debug_word_toxicity_detection(self, labels: torch.Tensor, sample_size: int = 5) -> dict:
    """Debug helper for word-level toxic detection"""
    debug_info = {
        'total_samples': labels.shape[0],
        'forbidden_words': self.forbidden_words[:20],  # First 20 for display
        'samples_analyzed': [],
        'overall_stats': {
            'total_toxic_words': 0,
            'samples_with_toxic': 0
        }
    }
    
    # Analyze each sample in batch
    for i in range(min(sample_size, labels.shape[0])):
        sample_labels = labels[i]
        valid_positions = (sample_labels != -100) & (sample_labels != self.tokenizer.pad_token_id)
        valid_tokens = sample_labels[valid_positions]
        
        # Check for toxic words in decoded text
        toxic_found = []
        try:
            decoded_text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
            if not self.case_sensitive:
                decoded_text = decoded_text.lower()
            
            toxic_positions = self._find_toxic_words_in_text(decoded_text)
            toxic_found = [{
                'word': word,
                'start_pos': start,
                'end_pos': end,
                'matched_text': decoded_text[start:end]
            } for start, end, word in toxic_positions]
            
        except Exception as e:
            decoded_text = f"[Decode error: {e}]"
        
        sample_info = {
            'sample_index': i,
            'total_valid_tokens': len(valid_tokens),
            'toxic_words_found': len(toxic_found),
            'toxic_details': toxic_found,
            'decoded_text': decoded_text[:100] + "..." if len(decoded_text) > 100 else decoded_text
        }
        
        debug_info['samples_analyzed'].append(sample_info)
        debug_info['overall_stats']['total_toxic_words'] += len(toxic_found)
        if len(toxic_found) > 0:
            debug_info['overall_stats']['samples_with_toxic'] += 1
    
    return debug_info

# Bind the debug method to the class
ToxicMaskGenerator.debug_toxicity_detection = debug_word_toxicity_detection


def create_toxic_token_mask(
    tokenizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    forbidden_words: List[str],
    strategy: str = "toxic_only",
    context_window: int = 2,
    case_sensitive: bool = False,
    match_partial: bool = True
) -> Tuple[torch.Tensor, dict]:
    """
    Convenient function to create toxic word masks.
    
    Args:
        tokenizer: The tokenizer
        input_ids: Input token IDs [batch_size, seq_len]
        labels: Label token IDs [batch_size, seq_len]
        forbidden_words: List of forbidden words
        strategy: Masking strategy ("toxic_only", "toxic_context", "inverse_toxic")
        context_window: Context window size for "toxic_context" strategy
        case_sensitive: Whether word matching is case sensitive
        match_partial: Whether to match partial word occurrences
        
    Returns:
        mask: Boolean mask tensor
        stats: Dictionary with mask statistics
    """
    mask_generator = ToxicMaskGenerator(
        tokenizer=tokenizer,
        forbidden_words=forbidden_words,
        mask_strategy=strategy,
        case_sensitive=case_sensitive,
        match_partial=match_partial
    )
    
    mask = mask_generator.create_toxic_token_mask(
        input_ids=input_ids,
        labels=labels,
        context_window=context_window
    )
    
    stats = mask_generator.get_mask_statistics(input_ids, labels, mask)
    
    return mask, stats


def analyze_sentence_toxicity(
    sentence: str,
    tokenizer,
    forbidden_words: List[str],
    return_positions: bool = True,
    case_sensitive: bool = False,
    match_partial: bool = True
) -> dict:
    """
    Analyze a sentence for toxic words and their positions.
    
    Args:
        sentence: Input sentence to analyze
        tokenizer: Tokenizer to use
        forbidden_words: List of forbidden words
        return_positions: Whether to return detailed position info
        case_sensitive: Whether word matching is case sensitive
        match_partial: Whether to match partial word occurrences
        
    Returns:
        Dictionary with toxicity analysis
    """
    import re
    
    # Create a temporary mask generator for analysis
    temp_generator = ToxicMaskGenerator(
        tokenizer=tokenizer,
        forbidden_words=forbidden_words,
        mask_strategy="toxic_only",
        case_sensitive=case_sensitive,
        match_partial=match_partial
    )
    
    # Find toxic words in the text
    search_text = sentence if case_sensitive else sentence.lower()
    toxic_word_positions = temp_generator._find_toxic_words_in_text(search_text)
    
    # Tokenize the sentence for token-level info
    tokens = tokenizer.encode(sentence, return_tensors="pt")
    token_list = tokens[0].tolist()
    readable_tokens = [tokenizer.decode([tid]) for tid in token_list]
    
    # Extract found toxic words
    toxic_words_found = list(set([word for _, _, word in toxic_word_positions]))
    
    analysis = {
        'sentence': sentence,
        'total_tokens': len(token_list),
        'toxic_count': len(toxic_word_positions),
        'toxic_ratio': len(toxic_word_positions) / len(token_list) if token_list else 0,
        'has_toxic_tokens': len(toxic_word_positions) > 0,
        'toxic_tokens': toxic_words_found,
        'toxic_word_occurrences': len(toxic_word_positions)
    }
    
    if return_positions:
        analysis.update({
            'token_ids': token_list,
            'readable_tokens': readable_tokens,
            'toxic_word_positions': toxic_word_positions  # (start, end, word) tuples
        })
    
    return analysis

def detoxify_text(toxic_text, model=None, tokenizer=None, max_length=512, bad_words_ids=None):
    """Generate detoxified version of toxic text"""
    # Use defaults if not provided
    if model is None:
        model = MODEL
    if tokenizer is None:
        tokenizer = TOKENIZER
        
    # Add task prefix
    input_text = f"detoxify: {toxic_text}"
    
    # Tokenize input
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True
    )
    
    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate output
    with torch.no_grad():
        generation_kwargs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'max_length': max_length,
            'num_beams': 4,
            'do_sample': False,
            'early_stopping': True,
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
        }
        
        # Add bad words if provided
        if bad_words_ids is not None:
            generation_kwargs['bad_words_ids'] = bad_words_ids
            
        outputs = model.generate(**generation_kwargs)
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def check_trainable_params(model):
    """Check and print the number of trainable parameters in the model"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")
    return trainable_params, total_params


class SentinelTokenReplacer:
    """
    Replaces toxic words with T5 sentinel tokens (<extra_id_X>).
    Contiguous toxic words are replaced with a single sentinel token.
    """
    
    def __init__(
        self,
        tokenizer,
        forbidden_words: List[str],
        case_sensitive: bool = False,
        match_partial: bool = True
    ):
        """
        Initialize the sentinel token replacer.
        
        Args:
            tokenizer: T5 tokenizer
            forbidden_words: List of forbidden words/phrases
            case_sensitive: Whether word matching is case sensitive
            match_partial: Whether to match partial word occurrences
        """
        self.tokenizer = tokenizer
        self.forbidden_words = [word.strip() for word in (forbidden_words or []) if word.strip()]
        self.case_sensitive = case_sensitive
        self.match_partial = match_partial
        
        # Prepare forbidden words for matching
        if not self.case_sensitive:
            self.forbidden_words = [word.lower() for word in self.forbidden_words]
        
        print(f"🎯 SentinelTokenReplacer initialized")
        print(f"🚫 Tracking {len(self.forbidden_words)} forbidden words")
        print(f"📋 Case sensitive: {case_sensitive}")
        print(f"📋 Partial matching: {match_partial}")
    
    def _find_toxic_words_in_text(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Find toxic word positions in text.
        
        Args:
            text: Input text
            
        Returns:
            List of (start_pos, end_pos, word) tuples sorted by position
        """
        import re
        
        search_text = text if self.case_sensitive else text.lower()
        toxic_positions = []
        
        for word in self.forbidden_words:
            # Create regex pattern for word matching
            if self.match_partial:
                # Match word as part of larger words
                pattern = re.escape(word)
            else:
                # Match only whole words
                pattern = r'\b' + re.escape(word) + r'\b'
            
            # Find all matches
            for match in re.finditer(pattern, search_text):
                toxic_positions.append((match.start(), match.end(), word))
        
        return sorted(toxic_positions, key=lambda x: x[0])  # Sort by start position
    
    def _merge_contiguous_spans(
        self, 
        toxic_positions: List[Tuple[int, int, str]], 
        text: str
    ) -> List[Tuple[int, int, List[str]]]:
        """
        Merge contiguous toxic word spans into single spans.
        Words separated only by whitespace/punctuation are considered contiguous.
        
        Args:
            toxic_positions: List of (start, end, word) tuples
            text: Original text
            
        Returns:
            List of (start, end, [words]) tuples for merged spans
            
        Example:
            Input: "fucking stupid" -> [(0, 7, "fucking"), (8, 14, "stupid")]
            Output: [(0, 14, ["fucking", "stupid"])]
        """
        if not toxic_positions:
            return []
        
        merged_spans = []
        current_start, current_end, current_words = toxic_positions[0][0], toxic_positions[0][1], [toxic_positions[0][2]]
        
        for i in range(1, len(toxic_positions)):
            start, end, word = toxic_positions[i]
            
            # Check the text between current span and next toxic word
            between_text = text[current_end:start]
            
            # If only whitespace or punctuation between them, merge
            if between_text.strip() == '' or all(c in ' \t\n,.!?;:\'"' for c in between_text):
                # Extend current span
                current_end = end
                current_words.append(word)
            else:
                # Save current span and start new one
                merged_spans.append((current_start, current_end, current_words))
                current_start, current_end, current_words = start, end, [word]
        
        # Add the last span
        merged_spans.append((current_start, current_end, current_words))
        
        return merged_spans
    
    def replace_toxic_with_sentinels(
        self, 
        text: str
    ) -> Tuple[str, List[Tuple[List[str], int]]]:
        """
        Replace toxic words in text with sentinel tokens.
        Contiguous toxic words are replaced with a single sentinel.
        
        Args:
            text: Input text
            
        Returns:
            - modified_text: Text with toxic words replaced by sentinels
            - replacements: List of ([original_words], sentinel_id) tuples
            
        Example:
            Input: "Hey you fucking stupid boy, what are you fucking doing"
            Output: ("Hey you <extra_id_0> boy, what are you <extra_id_1> doing",
                     [(["fucking", "stupid"], 0), (["fucking"], 1)])
        """
        # Find all toxic word positions
        toxic_positions = self._find_toxic_words_in_text(text)
        
        if not toxic_positions:
            return text, []
        
        # Merge contiguous toxic spans
        merged_spans = self._merge_contiguous_spans(toxic_positions, text)
        
        # Build modified text with sentinels
        modified_text = ""
        last_pos = 0
        replacements = []
        
        for sentinel_id, (start, end, words) in enumerate(merged_spans):
            # Add text before toxic span
            modified_text += text[last_pos:start]
            
            # Add sentinel token
            sentinel = f"<extra_id_{sentinel_id}>"
            modified_text += sentinel
            
            # Record replacement
            replacements.append((words, sentinel_id))
            
            # Update position
            last_pos = end
        
        # Add remaining text
        modified_text += text[last_pos:]
        
        return modified_text, replacements
    
    def create_target_sequence(
        self,
        replacements: List[Tuple[List[str], int]]
    ) -> str:
        """
        Create T5-style target sequence with sentinel tokens and their replacements.
        
        Args:
            replacements: List of ([original_words], sentinel_id) tuples
            
        Returns:
            Target sequence in format: "<extra_id_0> replacement <extra_id_1> replacement"
            
        Example:
            Input: [(["fucking", "stupid"], 0), (["fucking"], 1)]
            Output: "<extra_id_0> <extra_id_1>"
            
        Note: In span corruption, target typically just contains sentinels marking spans,
              the model learns to generate appropriate replacements.
        """
        if not replacements:
            return ""
        
        target_parts = []
        for words, sentinel_id in replacements:
            # Add sentinel marker
            target_parts.append(f"<extra_id_{sentinel_id}>")
        
        return " ".join(target_parts)
    
    def create_target_with_alternatives(
        self,
        replacements: List[Tuple[List[str], int]],
        alternatives: List[str]
    ) -> str:
        """
        Create target sequence with specific alternative words.
        Use this when you have specific non-toxic alternatives to teach.
        
        Args:
            replacements: List of ([original_words], sentinel_id) tuples
            alternatives: List of alternative phrases (same length as replacements)
            
        Returns:
            Target sequence with alternatives
            
        Example:
            replacements: [(["fucking", "stupid"], 0), (["fucking"], 1)]
            alternatives: ["nice", "really"]
            Output: "<extra_id_0> nice <extra_id_1> really"
        """
        if len(replacements) != len(alternatives):
            raise ValueError(f"Mismatch: {len(replacements)} replacements but {len(alternatives)} alternatives")
        
        target_parts = []
        for i, (words, sentinel_id) in enumerate(replacements):
            target_parts.append(f"<extra_id_{sentinel_id}>")
            target_parts.append(alternatives[i])
        
        return " ".join(target_parts)