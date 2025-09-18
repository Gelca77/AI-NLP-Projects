# src/utils.py
#helper functions

import re

# Simple keyword lists (I will expand as I get more smishing texts)
URGENCY_WORDS = [
    "urgent","immediately","now","act now","last chance","final notice",
    "verify now","respond now","limited time","24 hours","48 hours","asap","today"
]

FEAR_WORDS = [
    "suspended","locked","compromised","warning","failed","problem",
    "overdue","deactivated","penalty","fraud","unauthorized","threat","cancelled"
]

def clean_text(text: str) -> str:
    """
    Lowercase and remove non-alphanumeric characters (but keep spaces).
    Basic cleanup so analysis is easier.
    """
    if text is None:
        return ""
    text = str(text).lower()
    # replace non-alphanumeric characters with space
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def count_hits(text: str, vocab: list) -> int:
    """
    Count how many words from vocab appear in text.
    Uses whole-word matching. Returns integer count.
    """
    t = clean_text(text)
    count = 0
    for w in vocab:
        # word boundary to avoid partial matches
        if re.search(rf"\b{re.escape(w)}\b", t):
            count += 1
    return count
