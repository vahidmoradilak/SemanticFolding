import re
import json
from collections import Counter

from hazm import Normalizer, word_tokenize
import spacy

# load models
nlp_en = spacy.load("en_core_web_sm")
normalizer = Normalizer()


# ---------------------------
# Language Detection
# ---------------------------
def detect_language(word):
    if re.search(r'[\u0600-\u06FF]', word):
        return "fa"   # فارسی/عربی
    elif re.search(r'[a-zA-Z]', word):
        return "en"
    else:
        return "other"


# ---------------------------
# N-gram generator
# ---------------------------
def generate_ngrams(tokens, max_n=3):
    ngrams = []
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngrams.append(" ".join(tokens[i:i+n]))
    return ngrams


# ---------------------------
# Persian Processing (Hazm)
# ---------------------------
def process_persian(words):
    text = " ".join(words)
    text = normalizer.normalize(text)
    tokens = word_tokenize(text)
    return tokens


# ---------------------------
# English Processing (spaCy)
# ---------------------------
def process_english(words):
    doc = nlp_en(" ".join(words))
    tokens = [token.text for token in doc if not token.is_punct]
    return tokens


# ---------------------------
# Main Extraction Function
# ---------------------------
def extract_phrases_multilingual(text, max_ngram=3):
    words = text.split()

    fa_words = []
    en_words = []

    for w in words:
        lang = detect_language(w)
        if lang == "fa":
            fa_words.append(w)
        elif lang == "en":
            en_words.append(w)

    # process separately
    fa_tokens = process_persian(fa_words) if fa_words else []
    en_tokens = process_english(en_words) if en_words else []

    # generate phrases
    fa_phrases = generate_ngrams(fa_tokens, max_ngram)
    en_phrases = generate_ngrams(en_tokens, max_ngram)

    return fa_phrases + en_phrases


# ---------------------------
# Corpus Processing
# ---------------------------
def process_corpus(file_path, output_path, min_freq=2, max_ngram=3):
    counter = Counter()

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # اگر corpus مثل فایل تو index داره
            if "," in line:
                _, text = line.split(",", 1)
            else:
                text = line

            phrases = extract_phrases_multilingual(text, max_ngram)
            counter.update(phrases)

    # filter by frequency
    filtered = {k: v for k, v in counter.items() if v >= min_freq}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(filtered)} phrases to {output_path}")


# ---------------------------
# Test
# ---------------------------
if __name__ == "__main__":
    text = "هوش مصنوعی در ایران AI is growing سریع است"
    textA = "هوش مصنوعی در ایران AI is growing بسرعة سریع است"
    tokens = extract_phrases_multilingual(textA, 1)
    print(tokens)