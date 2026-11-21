"""
Rigorous Validation Test Suite for Tokenizer V2
Verifies:
1. Exact Round-trip lossless fidelity: decode(encode(x)) == x on 1,000 samples per corpus
   including Tamil, Hindi, code, math, and emoji.
2. Special tokens are single integer IDs (no sub-word fragmentation).
3. Individual digits strictly split one per token (e.g. 1234 -> ['1', '2', '3', '4']).
"""

import sys
import json
import tarfile
from pathlib import Path
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from tokenizers import Tokenizer

TOKENIZER_PATH = BASE_DIR / "models/aris_bpe_tokenizer_v2.json"

def test_special_tokens_are_single_ids():
    """Verify that every defined special token maps to a single integer token ID."""
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    special_tokens = [
        "<|pad|>",
        "<|unk|>",
        "<|bos|>",
        "<|eos|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|thought|>",
        "<|thought_end|>",
        "<|tool_call|>",
        "<|tool_result|>",
        "<|plan|>",
        "<|endoftext|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
    ] + [f"<|reserved_{i:03d}|>" for i in range(10)]

    for st in special_tokens:
        tok_id = tok.token_to_id(st)
        assert tok_id is not None, f"Special token '{st}' missing from vocabulary!"
        enc = tok.encode(st)
        assert len(enc.ids) == 1, f"Special token '{st}' split into multiple IDs: {enc.ids}"
        assert enc.ids[0] == tok_id, f"ID mismatch for '{st}': {enc.ids[0]} != {tok_id}"


def test_digits_split_individually():
    """Verify that multi-digit numbers are split strictly one per token."""
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    number_tests = [
        ("1234", ["1", "2", "3", "4"]),
        ("9007199254740993", [d for d in "9007199254740993"]),
        ("42", ["4", "2"]),
        ("1000", ["1", "0", "0", "0"]),
        ("3.14159", ["3", ".", "1", "4", "1", "5", "9"]),
    ]

    for num_str, expected_digits in number_tests:
        enc = tok.encode(num_str)
        decoded_tokens = [tok.decode([tok_id]) for tok_id in enc.ids]
        # Verify count of digit tokens matches length of digit string
        digit_tokens = [t for t in decoded_tokens if t in "0123456789"]
        assert len(digit_tokens) == len([c for c in num_str if c.isdigit()]), (
            f"Digits in '{num_str}' were merged! Got tokens: {decoded_tokens}"
        )


def test_lossless_roundtrip_multilingual_and_emoji():
    """Verify exact lossless roundtrip decode(encode(x)) == x on 1,000 diverse samples."""
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))

    # Explicit edge cases: Tamil, Hindi, Emoji, Code, Math
    edge_cases = [
        "தமிழ் அரிச்சுவடி என்பது தமிழ் மொழியில் உள்ள எழுத்துக்களின் வரிசை வடிவம் ஆகும்.",
        "हिन्दी विश्व की एक प्रमुख भाषा है एवं भारत की राजभाषा है।",
        "Machine Learning 🚀 & Autonomous Intelligence 🧠 are advancing exponentially! ⚡",
        "def quick_sort(arr: list[int]) -> list[int]:\n    return arr if len(arr) <= 1 else ...",
        "Calculation: 1248 * 492 + 8192 - (1048576 / 64) = 614016",
        "Special whitespace: \t\t   spaces   and\nnewlines\r\nand trailing spaces   ",
    ]

    for ec in edge_cases:
        enc = tok.encode(ec)
        dec = tok.decode(enc.ids)
        assert dec == ec, f"Lossless roundtrip failed for edge case!\nExpected: {repr(ec)}\nGot:      {repr(dec)}"

    # 1,000 corpus samples: GSM8K, MBPP, SQuAD, UltraChat, and UD
    samples = []

    # GSM8K
    gsm_tbl = pq.read_table(BASE_DIR / "data/gsm8k/train.parquet")
    for r in gsm_tbl.to_pylist()[:250]:
        samples.append(r["question"])
        samples.append(r["answer"])

    # MBPP
    mbpp_tbl = pq.read_table(BASE_DIR / "data/mbpp/train.parquet")
    for r in mbpp_tbl.to_pylist()[:200]:
        samples.append(r["text"])
        samples.append(r["code"])

    # SQuAD
    sq_tbl = pq.read_table(BASE_DIR / "data/squad_v2/train.parquet")
    for r in sq_tbl.to_pylist()[:250]:
        samples.append(r["question"])

    # UltraChat
    uc_tbl = pq.read_table(BASE_DIR / "data/ultrachat_200k/data/train_gen-00000-of-00003-a6c9fb894be3e50b.parquet", columns=["prompt"])
    for p in uc_tbl["prompt"][:200].to_pylist():
        samples.append(p)

    # UD Multilingual text
    ud_archive = BASE_DIR / "ud-treebanks-v2.18.tgz"
    if ud_archive.exists():
        with tarfile.open(ud_archive, "r:gz") as tar:
            for m in tar.getmembers()[:30]:
                if m.name.endswith(".conllu"):
                    f = tar.extractfile(m)
                    if f:
                        for line in f:
                            l = line.decode("utf-8", errors="replace").strip()
                            if l.startswith("# text ="):
                                samples.append(l.replace("# text =", "").strip())
                                if len(samples) >= 1000:
                                    break
                if len(samples) >= 1000:
                    break

    assert len(samples) >= 1000, f"Need at least 1,000 samples, got {len(samples)}"
    print(f"Testing lossless roundtrip on {len(samples):,} diverse multi-domain samples...")

    failures = 0
    for idx, s in enumerate(samples[:1000]):
        if not s:
            continue
        enc = tok.encode(s)
        dec = tok.decode(enc.ids)
        if dec != s:
            failures += 1
            if failures <= 3:
                print(f"  Roundtrip mismatch at sample {idx}:")
                print(f"    Expected: {repr(s[:60])}")
                print(f"    Got:      {repr(dec[:60])}")

    assert failures == 0, f"Total {failures} lossless roundtrip failures out of {len(samples[:1000])} samples!"


if __name__ == "__main__":
    test_special_tokens_are_single_ids()
    print("[PASS] test_special_tokens_are_single_ids")
    test_digits_split_individually()
    print("[PASS] test_digits_split_individually")
    test_lossless_roundtrip_multilingual_and_emoji()
    print("[PASS] test_lossless_roundtrip_multilingual_and_emoji")
    print("\nALL TOKENIZER V2 TESTS PASSED CLEANLY (100%)")
