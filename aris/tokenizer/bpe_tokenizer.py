"""
AR.I.S — Byte-Level BPE Neural Tokenizer (Production Grade)
Fixes Applied from Senior Audit:
  1. No fallback on 0: explicit None check with KeyError / ValueError.
  2. Exact round-trip fidelity: removed .strip() from decode().
  3. Batch decoding support: handles int, 1D List[int], and 2D List[List[int]].
  4. Structured ChatML Protocol (<|im_start|>, <|im_end|>).
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Union, Iterator, Dict, Any
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class ArisTokenizer:
    """
    Byte-Level BPE Tokenizer for AR.I.S with ChatML format and exact whitespace fidelity.
    """

    # AR.I.S Native Chat & Tool Tokens
    SPECIAL_TOKENS = [
        "<|pad|>",
        "<|unk|>",
        "<|bos|>",
        "<|eos|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|tool_call|>",
        "<|tool_result|>",
        "<|thought|>",
    ]

    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        if tokenizer is not None:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
            self._tokenizer.pre_tokenizer = ByteLevel()
            self._tokenizer.decoder = ByteLevelDecoder()

    def get_token_id(self, token: str) -> int:
        """
        Safely retrieves token ID.
        Fixes Critical Bug 1: 0 is NOT treated as False. Explicitly checks for None.
        """
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            raise KeyError(f"Special token '{token}' is missing from the trained vocabulary.")
        return token_id

    @property
    def pad_id(self) -> int:
        return self.get_token_id("<|pad|>")

    @property
    def unk_id(self) -> int:
        return self.get_token_id("<|unk|>")

    @property
    def bos_id(self) -> int:
        return self.get_token_id("<|bos|>")

    @property
    def eos_id(self) -> int:
        return self.get_token_id("<|eos|>")

    @property
    def im_start_id(self) -> int:
        return self.get_token_id("<|im_start|>")

    @property
    def im_end_id(self) -> int:
        return self.get_token_id("<|im_end|>")

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encodes string to token IDs preserving exact whitespace."""
        if not text:
            return []
        encoding = self._tokenizer.encode(text)
        ids = encoding.ids
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def format_chat_message(self, role: str, content: str) -> str:
        """
        Formats a single message according to the AR.I.S ChatML protocol:
        <|im_start|>role
        content<|im_end|>
        """
        return f"<|im_start|>{role}\n{content}<|im_end|>\n"

    def encode_chat(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> List[int]:
        """
        Encodes a multi-turn conversation into token IDs.
        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
            add_generation_prompt: Appends '<|im_start|>assistant\n' for model completion
        """
        formatted_text = ""
        for msg in messages:
            formatted_text += self.format_chat_message(msg["role"], msg["content"])

        if add_generation_prompt:
            formatted_text += "<|im_start|>assistant\n"

        return self.encode(formatted_text, add_special_tokens=False)

    def decode(self, ids: Union[int, List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """
        Decodes token IDs back to string.
        Fixes Critical Bug 2: Does NOT call .strip() — preserves exact whitespace and indentation.
        """
        if isinstance(ids, torch.Tensor):
            if ids.dim() > 1:
                # If 2D batch tensor, decode each item
                return [self.decode(row.tolist(), skip_special_tokens=skip_special_tokens) for row in ids]
            ids = ids.tolist()
        elif isinstance(ids, int):
            ids = [ids]

        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, batch_ids: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Batch decoding method with exact sequence preservation."""
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in batch_ids]

    def train_from_iterator(self, iterator: Iterator[str], vocab_size: int = 8192) -> None:
        """Trains Byte-Level BPE on an iterator of strings."""
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=self.SPECIAL_TOKENS,
            initial_alphabet=ByteLevel.alphabet(),
        )
        self._tokenizer.train_from_iterator(iterator, trainer=trainer)

    def save(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path))

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> ArisTokenizer:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
        tok = Tokenizer.from_file(str(path))
        return cls(tok)
