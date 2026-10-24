"""
AR.I.S — Canonical Tokenizer & Discrete Vocabulary Primitive (Phase 1)

Constitutional Directives:
  - MANDATORY UNIVERSAL SERIALIZATION BOUNDARY:
    Every piece of language entering AR.I.S (user queries, memory records,
    browser results, command outputs, observations, training examples)
    must pass through the canonical tokenizer and become discrete Token IDs
    before touching the neural substrate.
  - DISCRETE MACHINE SYMBOLS:
    Token IDs are pure discrete integers in {0, ..., |V|-1}.
    Token IDs do NOT contain meaning; they are discrete serialization coordinates.
  - REPRODUCIBLE VOCABULARY ARTIFACT:
    Vocabulary states can be frozen, saved, and loaded to remain strictly coupled
    with model checkpoint parameters. At inference time, unknown words map to [UNK]
    without mutating the vocabulary.
  - SINGLE CANONICAL OUTPUT OBJECT:
    Downstream cognitive and neural layers receive a TokenizedSequence object.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import torch


@dataclass
class TokenizedSequence:
    """
    The universal canonical data structure emitted by the serialization boundary.
    Any language-bearing input entering AR.I.S is converted to this representation.
    """
    input_ids: torch.Tensor          # LongTensor of shape (1, L) or (Batch, L)
    attention_mask: torch.Tensor     # LongTensor of shape (1, L) or (Batch, L)
    tokens: List[str]                # Discrete string tokens
    source: str                      # Source tag: "user_query", "memory_record", "browser_result", "command_output", "observation"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def sequence_length(self) -> int:
        return self.input_ids.shape[-1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_ids": self.input_ids.tolist(),
            "attention_mask": self.attention_mask.tolist(),
            "tokens": self.tokens,
            "source": self.source,
            "sequence_length": self.sequence_length,
            "metadata": self.metadata,
        }


class Vocabulary:
    """
    Bidirectional mapping between discrete textual tokens and integer token IDs.
    Supports explicit freeze/unfreeze cycles and persistent serialization.
    """

    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    MASK_TOKEN = "[MASK]"

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN]

    def __init__(self, special_tokens: Optional[List[str]] = None):
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._lower_to_id: Dict[str, int] = {}
        self._frozen: bool = False
        
        # Register standard special tokens strictly
        specials = special_tokens if special_tokens is not None else self.SPECIAL_TOKENS
        for token in specials:
            self._register(token)

    @property
    def pad_id(self) -> int:
        return self._token_to_id[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self._token_to_id[self.UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self._token_to_id[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self._token_to_id[self.EOS_TOKEN]

    @property
    def mask_id(self) -> int:
        return self._token_to_id[self.MASK_TOKEN]

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        """Freezes the vocabulary to prevent accidental mutation during inference."""
        self._frozen = True

    def unfreeze(self) -> None:
        """Unfreezes the vocabulary for explicit vocabulary expansion/training."""
        self._frozen = False

    def _register(self, token: str) -> int:
        if token not in self._token_to_id:
            idx = len(self._token_to_id)
            self._token_to_id[token] = idx
            self._id_to_token[idx] = token
            self._lower_to_id[token.lower()] = idx
            return idx
        return self._token_to_id[token]

    def add_token(self, token: str) -> int:
        """
        Adds a token to the vocabulary if not present, returning its unique integer ID.
        Raises RuntimeError if the vocabulary is frozen.
        """
        if self._frozen and token not in self._token_to_id:
            raise RuntimeError(
                f"Cannot add token '{token}' to a frozen vocabulary. "
                f"Unknown tokens must resolve to [UNK] at inference time."
            )
        return self._register(token)

    def token_to_id(self, token: str) -> int:
        """Returns the discrete integer ID for the token, falling back to [UNK] if unseen."""
        if token in self._token_to_id:
            return self._token_to_id[token]
        if token.lower() in self._lower_to_id:
            return self._lower_to_id[token.lower()]
        return self.unk_id

    def id_to_token(self, idx: int) -> str:
        """Returns the discrete string symbol for the token ID, falling back to [UNK]."""
        return self._id_to_token.get(idx, self.UNK_TOKEN)

    def contains(self, token: str) -> bool:
        return token in self._token_to_id or token.lower() in self._lower_to_id

    def __len__(self) -> int:
        return len(self._token_to_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": len(self),
            "is_frozen": self._frozen,
            "token_to_id": dict(self._token_to_id),
        }

    def save(self, filepath: str) -> None:
        """Persists the vocabulary to disk as a JSON artifact."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str, freeze_on_load: bool = True) -> "Vocabulary":
        """Loads a persisted vocabulary artifact from disk."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(special_tokens=[])
        vocab._token_to_id = {k: int(v) for k, v in data.get("token_to_id", {}).items()}
        vocab._id_to_token = {int(v): k for k, v in vocab._token_to_id.items()}
        vocab._lower_to_id = {k.lower(): int(v) for k, v in data.get("token_to_id", {}).items()}
        vocab._frozen = freeze_on_load if freeze_on_load else data.get("is_frozen", False)
        return vocab

    @classmethod
    def from_dict(cls, data: Dict[str, Any], freeze_on_load: bool = False) -> "Vocabulary":
        vocab = cls(special_tokens=[])
        vocab._token_to_id = {k: int(v) for k, v in data.get("token_to_id", {}).items()}
        vocab._id_to_token = {int(v): k for k, v in vocab._token_to_id.items()}
        vocab._lower_to_id = {k.lower(): int(v) for k, v in data.get("token_to_id", {}).items()}
        vocab._frozen = freeze_on_load or data.get("is_frozen", False)
        return vocab


class ArisCanonicalTokenizer:
    """
    Universal language serialization interface for AR.I.S.
    Converts arbitrary language and observation strings into canonical TokenizedSequence objects.
    """

    DEFAULT_VOCAB_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "models",
        "aris_phase2_vocabulary.json"
    )

    def __init__(self, vocabulary: Optional[Vocabulary] = None):
        if vocabulary is not None:
            self.vocab = vocabulary
        elif os.path.exists(self.DEFAULT_VOCAB_PATH):
            self.vocab = Vocabulary.load(self.DEFAULT_VOCAB_PATH, freeze_on_load=False)
        else:
            self.vocab = Vocabulary()

    @classmethod
    def segment_unspaced_text(cls, raw: str, vocab: Vocabulary) -> List[Tuple[str, int, int]]:
        """
        Segments unspaced compound or concatenated text into constituent vocabulary tokens
        using dynamic programming over learned vocabulary item existence.
        Preserves exact (token, start, end) offsets within raw string.
        """
        n = len(raw)
        if n < 4 or vocab.contains(raw):
            return [(raw, 0, n)]

        dp: List[Optional[List[Tuple[str, int, int]]]] = [None] * (n + 1)
        dp[0] = []

        for i in range(n):
            if dp[i] is None:
                continue
            for j in range(i + 1, n + 1):
                sub = raw[i:j]
                if vocab.contains(sub):
                    candidate = dp[i] + [(sub, i, j)]
                    if dp[j] is None or len(candidate) < len(dp[j]):
                        dp[j] = candidate

        if dp[n] is not None and len(dp[n]) > 1:
            return dp[n]
        return [(raw, 0, n)]

    def tokenize_raw(self, text: str) -> List[str]:
        """Splits raw text into a discrete list of string tokens."""
        if not text or not isinstance(text, str):
            return []
        return [s[0] for s in self.tokenize_with_spans(text)]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encodes a string into a list of discrete integer Token IDs.
        """
        tokens = self.tokenize_raw(text)
        token_ids = [self.vocab.token_to_id(t) for t in tokens]
        if add_special_tokens:
            token_ids = [self.vocab.bos_id] + token_ids + [self.vocab.eos_id]
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        Deserializes a list of discrete integer Token IDs back into text.
        """
        tokens = []
        specials = {self.vocab.pad_id, self.vocab.unk_id, self.vocab.bos_id, self.vocab.eos_id, self.vocab.mask_id}
        for tid in token_ids:
            if skip_special_tokens and tid in specials:
                continue
            tokens.append(self.vocab.id_to_token(tid))
        return " ".join(tokens)

    def tokenize_with_spans(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Splits raw text into discrete tokens with exact (token, char_start, char_end) coordinates
        using pure character-level lexical scanning and learned vocabulary decomposition (Rule 1 & 7: Zero Regex).
        """
        if not text or not isinstance(text, str):
            return []
        spans: List[Tuple[str, int, int]] = []
        n = len(text)
        i = 0
        while i < n:
            ch = text[i]
            if ch.isspace():
                i += 1
                continue

            # Case 1: Bracketed special tokens, e.g. [PAD], [BOS], [EOS], [UNK], [MASK]
            if ch == '[':
                close_idx = text.find(']', i + 1)
                if close_idx != -1 and (close_idx - i) <= 16:
                    candidate = text[i:close_idx + 1]
                    inner = text[i + 1:close_idx]
                    if self.vocab.contains(candidate) or (inner.isalnum() and inner.isupper()):
                        spans.append((candidate, i, close_idx + 1))
                        i = close_idx + 1
                        continue

            # Case 2: Alphanumeric word constituent (with optional apostrophe contraction, e.g. "don't", "Arun's")
            if ch.isalnum() or ch == '_':
                start = i
                while i < n and (text[i].isalnum() or text[i] == '_'):
                    i += 1
                # Contraction: apostrophe followed by alpha characters
                if i < n - 1 and text[i] in ("'", "’") and text[i + 1].isalpha():
                    i += 1
                    while i < n and text[i].isalpha():
                        i += 1
                raw_word = text[start:i]
                sub_segments = self.segment_unspaced_text(raw_word, self.vocab)
                for sub_tok, s, e in sub_segments:
                    spans.append((sub_tok, start + s, start + e))
                continue

            # Case 3: Symbols & punctuation (single atomic characters)
            spans.append((ch, i, i + 1))
            i += 1

        return spans

    def serialize(
        self,
        text: str,
        source: str = "general",
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TokenizedSequence:
        """
        Single canonical entrypoint for all language-bearing input entering AR.I.S.
        Emits a standard TokenizedSequence with PyTorch tensors and sequence metadata.
        """
        token_matches = self.tokenize_with_spans(text)
        raw_tokens = [m[0] for m in token_matches]
        token_spans = [(m[1], m[2]) for m in token_matches]
        token_ids = [self.vocab.token_to_id(t) for t in raw_tokens]

        if add_special_tokens:
            text_end = len(text) if text else 0
            token_ids = [self.vocab.bos_id] + token_ids + [self.vocab.eos_id]
            raw_tokens = [self.vocab.BOS_TOKEN] + raw_tokens + [self.vocab.EOS_TOKEN]
            token_spans = [(0, 0)] + token_spans + [(text_end, text_end)]

        if max_length is not None:
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                raw_tokens = raw_tokens[:max_length]
                token_spans = token_spans[:max_length]
            elif len(token_ids) < max_length:
                pad_len = max_length - len(token_ids)
                token_ids = token_ids + [self.vocab.pad_id] * pad_len
                raw_tokens = raw_tokens + [self.vocab.PAD_TOKEN] * pad_len
                token_spans = token_spans + [(-1, -1)] * pad_len

        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.tensor(
            [[1 if tid != self.vocab.pad_id else 0 for tid in token_ids]],
            dtype=torch.long,
        )

        meta = metadata or {}
        meta["raw_length"] = len(text) if text else 0
        meta["token_spans"] = token_spans

        return TokenizedSequence(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokens=raw_tokens,
            source=source,
            metadata=meta,
        )

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        add_special_tokens: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Batch encodes texts into PyTorch tensors:
          - input_ids: (Batch_Size, Sequence_Length)
          - attention_mask: (Batch_Size, Sequence_Length) [1 for real tokens, 0 for pad]
        """
        encoded_list = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]
        if not encoded_list:
            return {
                "input_ids": torch.empty((0, 0), dtype=torch.long),
                "attention_mask": torch.empty((0, 0), dtype=torch.long),
            }

        seq_lens = [len(seq) for seq in encoded_list]
        max_len = max_length or max(seq_lens)

        batch_ids = []
        batch_mask = []

        for seq in encoded_list:
            if len(seq) < max_len and padding:
                padded = seq + [self.vocab.pad_id] * (max_len - len(seq))
                mask = [1] * len(seq) + [0] * (max_len - len(seq))
            else:
                padded = seq[:max_len]
                mask = [1] * len(padded)
            batch_ids.append(padded)
            batch_mask.append(mask)

        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
        }

    def save_vocabulary(self, filepath: str) -> None:
        """Saves the coupled vocabulary artifact to disk."""
        self.vocab.save(filepath)

    @classmethod
    def from_vocabulary_file(cls, filepath: str) -> "ArisCanonicalTokenizer":
        """Loads a tokenizer instance from a persisted vocabulary file."""
        vocab = Vocabulary.load(filepath)
        return cls(vocabulary=vocab)
