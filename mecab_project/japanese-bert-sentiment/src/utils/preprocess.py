from typing import List
import re
import unicodedata

def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)  # Unicode normalization
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()  # Remove leading and trailing spaces
    return text

def tokenize(text: str) -> List[str]:
    # Simple tokenization by splitting on spaces
    return text.split()

def preprocess_text(text: str) -> List[str]:
    normalized_text = normalize_text(text)
    tokens = tokenize(normalized_text)
    return tokens