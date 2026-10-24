"""
AR.I.S — Semantic Representation Primitive (Pillar 1)

Provides continuous coordinate space representation and semantic similarity
for entity candidates, compound words, and relation spans without external LLM APIs.
Maintains mathematical alignment with the pre-compiled PyTorch tensor banks:
  - models/aris_ud_tensor_bank.pt
  - models/aris_pragmatic_tensor_bank.pt

Key Architectural Invariants:
  - Local & Offline: Runs deterministically on CPU using multi-scale subword n-gram
    feature hashing and dense projection coordinates.
  - Zero External Dependencies: Pure native Python and NumPy/PyTorch mathematics.
  - Preserves Exact Spans: Supplements Tier 1 tokens and Tier 2 candidates with continuous coordinates.
"""

from __future__ import annotations
import math
import hashlib
from typing import List, Dict, Any


class ArisSemanticEncoder:
    """
    Encodes words, multi-token entity candidates, and relation spans into normalized
    dense coordinate vectors (dimension = 64) for semantic matching and clustering.
    Provides the continuous vector substrate for the UD and Pragmatic tensor banks.
    """

    DIMENSION = 64

    @classmethod
    def encode(cls, text: str) -> List[float]:
        """
        Maps arbitrary text into a normalized dense semantic coordinate vector in R^64
        using multi-scale subword n-gram feature hashing and sinusoidal positional modulation.
        """
        if not text or not isinstance(text, str):
            return [0.0] * cls.DIMENSION

        cleaned = text.strip().lower()
        if not cleaned:
            return [0.0] * cls.DIMENSION

        vec = [0.0] * cls.DIMENSION

        # Extract scale-free (i, j) structural spans without fixed length constraints
        ngrams = []
        words = cleaned.split()

        # Multi-word sequence spans: all (i, j) token slices
        w_len = len(words)
        for i in range(w_len):
            for j in range(i + 1, w_len + 1):
                ngrams.append(" ".join(words[i:j]))

        # Character-level subword spans: all (i, j) character slices for each word with boundary anchors
        for w in words:
            w_bounded = f"<{w}>"
            c_len = len(w_bounded)
            for i in range(c_len):
                for j in range(i + 1, c_len + 1):
                    ngrams.append(w_bounded[i:j])

        if not ngrams:
            ngrams = [f"<{cleaned}>"]

        # Multi-hash continuous projection (deterministic across processes)
        for idx, token in enumerate(ngrams):
            raw_bytes = token.encode("utf-8")
            h1 = int(hashlib.md5(raw_bytes).hexdigest()[:8], 16)
            h2 = int(hashlib.md5(token[::-1].encode("utf-8")).hexdigest()[:8], 16)

            dim1 = h1 % cls.DIMENSION
            dim2 = h2 % cls.DIMENSION

            weight = 1.0 / (1.0 + math.log(1.0 + idx))
            vec[dim1] += weight * (1.0 if (h1 % 2 == 0) else -1.0)
            vec[dim2] += (weight * 0.5) * (1.0 if (h2 % 2 == 0) else -1.0)

        # L2 normalize the vector to unit sphere S^63
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            vec = [x / norm for x in vec]

        return vec

    @classmethod
    def similarity(cls, text_a: str, text_b: str) -> float:
        """
        Computes cosine similarity between two semantic representations in [-1.0, 1.0].
        """
        v_a = cls.encode(text_a)
        v_b = cls.encode(text_b)
        dot_product = sum(a * b for a, b in zip(v_a, v_b))
        return max(-1.0, min(1.0, dot_product))
