"""
Unit Tests for AR.I.S Tokenizer V3 (16,384 Vocab, Digits Splitting, Tool Calling Protocol)
"""

from pathlib import Path
import pytest
from aris.tokenizer.bpe_tokenizer import ArisTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_V3_PATH = BASE_DIR / "models/aris_bpe_tokenizer_v3.json"


@pytest.fixture(scope="module")
def tok_v3():
    assert TOKENIZER_V3_PATH.exists(), f"Tokenizer V3 not found at {TOKENIZER_V3_PATH}"
    return ArisTokenizer.load(TOKENIZER_V3_PATH)


def test_v3_vocab_size_and_powers_of_two(tok_v3):
    assert tok_v3.vocab_size == 16384, f"Expected 16,384 vocab, got {tok_v3.vocab_size}"


def test_v3_special_tokens_single_ids(tok_v3):
    expected_tokens = [
        "<|pad|>",
        "<|unk|>",
        "<|bos|>",
        "<|eos|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|thought|>",
        "<|thought_end|>",
        "<|tool_call|>",
        "<|tool_response|>",
        "<|tool_result|>",
        "<|tool_def|>",
    ]
    for token in expected_tokens:
        tok_id = tok_v3.get_token_id(token)
        encoded = tok_v3.encode(token, add_special_tokens=False)
        assert len(encoded) == 1, f"Token {token} should encode to 1 ID, got {encoded}"
        assert encoded[0] == tok_id, f"Token ID mismatch: {encoded[0]} != {tok_id}"


def test_v3_digits_split_individually(tok_v3):
    number_string = "1234567890"
    encoded = tok_v3.encode(number_string, add_special_tokens=False)
    assert len(encoded) == 10, f"Expected 10 individual digit tokens, got {len(encoded)}"


def test_v3_lossless_roundtrip_diverse(tok_v3):
    samples = [
        "def compute_fibonacci(n: int) -> int:\n    if n <= 1: return n\n    return compute_fibonacci(n-1) + compute_fibonacci(n-2)",
        "<|im_start|>user\nCalculate: 14 * 18 = ?<|im_end|>\n<|im_start|>assistant\n<|thought|>\nMultiplying 14 and 18.<|thought_end|>\n252<|im_end|>\n",
        "<|tool_call|>\n{\"tool\": \"calculator\", \"args\": {\"expr\": \"16 * 2\"}}\n<|im_end|>",
        "Tamil test: வணக்கம் உலகம்! 🚀✨",
    ]
    for text in samples:
        encoded = tok_v3.encode(text, add_special_tokens=False)
        decoded = tok_v3.decode(encoded, skip_special_tokens=False)
        assert decoded == text, f"Fidelity error!\nOriginal: {repr(text)}\nDecoded:  {repr(decoded)}"
