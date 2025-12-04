from collections import defaultdict
import unicodedata

def normalize_token(token):
    token = unicodedata.normalize("NFKC", token)
    return token.strip().lower()

def build_lexicons_from_dataset(dataset):
    lexicons = defaultdict(set)  # usamos set para evitar duplicados

    for toks, langs in zip(dataset["text"], dataset["language"]):
        for tok, lang in zip(toks, langs):
            tok = normalize_token(tok)
            lang = lang.strip()

            if tok and lang:
                lexicons[lang].add(tok)

    # Convertimos sets → listas y ordenamos
    lexicons = {lang: sorted(list(words)) for lang, words in lexicons.items()}

    return lexicons
