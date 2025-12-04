import re
import torch
from transformers import LogitsProcessor

class ToxicityLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        toxic_lexicons,
        language=None,
        penalty=12.0,
        block_extra_ids=True,
    ):
        self.tokenizer = tokenizer
        self.toxic_lexicons = toxic_lexicons
        self.language = language
        self.penalty = penalty
        self.block_extra_ids = block_extra_ids

        vocab = tokenizer.get_vocab()

        # 1. Map toxic words to token IDs
        self.single_token_ids = {}
        for lang, words in toxic_lexicons.items():
            ids = []
            for w in words:
                toks = tokenizer.encode(w, add_special_tokens=False)
                if len(toks) == 1:
                    ids.append(toks[0])
            self.single_token_ids[lang] = list(set(ids))

        # 2. Block <extra_id_X> tokens
        if block_extra_ids:
            pattern = re.compile(r"<extra_id_\d+>")
            self.extra_ids = [tid for tok, tid in vocab.items() if pattern.fullmatch(tok)]
        else:
            self.extra_ids = []

    def set_language(self, lang):
        self.language = lang

    def __call__(self, input_ids, scores):
        lang = self.language
        if lang is None or lang not in self.single_token_ids:
            return scores

        toxic_ids = self.single_token_ids[lang]

        # Penalizar insultos
        if toxic_ids:
            scores[:, toxic_ids] -= self.penalty

        # Eliminar sentinel tokens
        if self.extra_ids:
            scores[:, self.extra_ids] = -1e9

        return scores
