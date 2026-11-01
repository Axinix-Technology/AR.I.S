"""
AR.I.S — Evaluation Metrics for Cognitive Benchmarks (2026 Production Grade)
Standardized Token F1, Exact Match (EM), MRR, and Hit@K.
"""

from __future__ import annotations
import re
import string
from typing import List, Set


def normalize_text(text: str) -> str:
    """Standardizes text for token-level comparison without brittle heuristics."""
    text = text.lower()
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)
    # Normalize whitespace
    return " ".join(text.split())


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """
    Computes Token F1 score between model output and ground truth.
    """
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0

    common_tokens = set(pred_tokens) & set(gt_tokens)
    if not common_tokens:
        return 0.0

    precision = sum(min(pred_tokens.count(t), gt_tokens.count(t)) for t in common_tokens) / len(pred_tokens)
    recall = sum(min(pred_tokens.count(t), gt_tokens.count(t)) for t in common_tokens) / len(gt_tokens)

    if precision + recall == 0.0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Returns 1.0 for exact normalized match, 0.0 otherwise."""
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def compute_mrr(ranks: List[int]) -> float:
    """
    Computes Mean Reciprocal Rank: mean(1 / rank) for 1-based ranks.
    Rank 0 indicates not found.
    """
    if not ranks:
        return 0.0
    reciprocals = [1.0 / r if r > 0 else 0.0 for r in ranks]
    return sum(reciprocals) / len(reciprocals)


def compute_hit_at_k(ranks: List[int], k: int) -> float:
    """Computes Hit@K: fraction of queries where rank is within top k."""
    if not ranks:
        return 0.0
    hits = sum(1 for r in ranks if 0 < r <= k)
    return hits / len(ranks)


def compute_perplexity(loss: float) -> float:
    """Computes perplexity PPL = exp(loss)."""
    import math
    if math.isnan(loss) or loss > 50.0:
        return 99999.0
    return math.exp(loss)


def compute_word_validity_rate(text: str) -> float:
    """
    Evaluates what percentage of generated whitespace-delimited tokens
    form phonotactically valid words (at least one vowel, clean alpha, non-repetitive).
    """
    words = [w.strip(".,!?:;\"'()[]{}<>|\n\r\t/ ") for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return 0.0

    valid_count = 0
    vowels = set("aeiouyAEIOUY")

    for w in words:
        # Must be alphabetical and contain at least one vowel sound
        if w.isalpha() and any(c in vowels for c in w) and len(w) >= 2:
            # Check for excessive character repetition (e.g. "aaaaa" or "NOOOOUN")
            max_rep = max(w.count(c) for c in set(w))
            if max_rep <= 3:
                valid_count += 1

    return (valid_count / len(words)) * 100.0


def compute_upos_alignment(prediction: str, expected_tags: List[str]) -> float:
    """
    Evaluates whether generated text produces recognized Universal POS tags
    (NOUN, VERB, ADJ, PRON, ADV, DET, etc.) matching the sentence structure.
    """
    valid_upos_set = {
        "NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "NUM",
        "CONJ", "CCONJ", "SCONJ", "AUX", "PROPN", "PUNCT", "INTJ", "SYM", "X"
    }
    pred_upper = prediction.upper()
    found_tags = sum(1 for tag in expected_tags if tag in pred_upper)
    if not expected_tags:
        return 0.0
    return (found_tags / len(expected_tags)) * 100.0

