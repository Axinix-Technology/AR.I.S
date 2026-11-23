"""
AR.I.S Tokenizer V3 Training Script (16,384 Vocab, Digits Splitting, Tool Calling Support)
Key Specifications:
  1. Vocab Size: 16,384 (2^14 exact power of 2 for optimal CUDA GEMM performance).
  2. Pre-tokenizer: Sequence([Digits(individual_digits=True), ByteLevel(add_prefix_space=False)]).
  3. Lossless exact ByteLevel round-tripping (UTF-8, Multilingual, Emoji, Code).
  4. Special Control Tokens:
     - ChatML: <|im_start|>, <|im_end|>
     - Deliberation: <|thought|>, <|thought_end|>, <|plan|>
     - Tool Agency: <|tool_call|>, <|tool_response|>, <|tool_result|>, <|tool_def|>
     - FIM (Fill-in-the-Middle): <|fim_prefix|>, <|fim_middle|>, <|fim_suffix|>
     - 48 Reserved slots for future multimodal and hierarchical memory anchors.
"""

import sys
import json
import tarfile
import hashlib
from pathlib import Path
from typing import Iterator
import pyarrow.parquet as pq

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence, ByteLevel, Digits
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SPECIAL_TOKENS = [
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
    "<|plan|>",
    "<|endoftext|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
] + [f"<|reserved_{i:03d}|>" for i in range(48)]


def stream_multidomain_training_data() -> Iterator[str]:
    """Streams representative text across Prose, Code, Math, Chat, and Multilingual domains."""
    print("Streaming diverse multi-domain training text for Tokenizer V3...")

    # 1. Prose: SQuAD + Dolly
    sq_path = DATA_DIR / "squad_v2/train.parquet"
    if sq_path.exists():
        sq_tbl = pq.read_table(sq_path)
        for r in sq_tbl.to_pylist()[:20000]:
            ctx = r.get("context", "")
            q = r.get("question", "")
            if ctx: yield ctx
            if q: yield q

    dolly_path = DATA_DIR / "dolly/databricks-dolly-15k.jsonl"
    if dolly_path.exists():
        with open(dolly_path, "r", encoding="utf-8") as f:
            for line in list(f)[:12000]:
                item = json.loads(line)
                inst = item.get("instruction", "")
                resp = item.get("response", "")
                if inst: yield inst
                if resp: yield resp

    # 2. Math: GSM8K
    gsm_path = DATA_DIR / "gsm8k/train.parquet"
    if gsm_path.exists():
        gsm_tbl = pq.read_table(gsm_path)
        for r in gsm_tbl.to_pylist():
            q = r.get("question", "")
            a = r.get("answer", "")
            if q: yield q
            if a: yield a

    # 3. Code: MBPP
    mbpp_path = DATA_DIR / "mbpp/train.parquet"
    if mbpp_path.exists():
        mbpp_tbl = pq.read_table(mbpp_path)
        for r in mbpp_tbl.to_pylist():
            t = r.get("text", "")
            c = r.get("code", "")
            if t: yield t
            if c: yield c
            for test in r.get("test_list", []):
                yield test

    # 4. Chat: UltraChat + SmolTalk
    uc_path = DATA_DIR / "ultrachat_200k/data/train_gen-00000-of-00003-a6c9fb894be3e50b.parquet"
    if uc_path.exists():
        uc_tbl = pq.read_table(uc_path, columns=["messages"])
        for dialogue in uc_tbl["messages"][:20000].to_pylist():
            for m in dialogue:
                c = m.get("content", "")
                if c: yield c

    smol_path = DATA_DIR / "smoltalk/everyday_conversations.parquet"
    if smol_path.exists():
        smol_tbl = pq.read_table(smol_path)
        for dialogue in smol_tbl["messages"].to_pylist():
            for m in dialogue:
                c = m.get("content", "")
                if c: yield c

    smol_magpie = DATA_DIR / "smoltalk/smol_magpie_ultra.parquet"
    if smol_magpie.exists():
        magpie_tbl = pq.read_table(smol_magpie, columns=["messages"])
        for dialogue in magpie_tbl["messages"][:20000].to_pylist():
            for m in dialogue:
                c = m.get("content", "")
                if c: yield c

    # 5. Multilingual Syntax from UD archive
    ud_archive = BASE_DIR / "ud-treebanks-v2.18.tgz"
    if ud_archive.exists():
        with tarfile.open(ud_archive, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith("-train.conllu")][:8]
            for m in members:
                f = tar.extractfile(m)
                if f:
                    for line in f:
                        line_str = line.decode("utf-8", errors="replace").strip()
                        if line_str and not line_str.startswith("#"):
                            parts = line_str.split("\t")
                            if len(parts) > 1:
                                yield parts[1]


def train_tokenizer_v3(vocab_size: int = 16384, output_path: Path = MODELS_DIR / "aris_bpe_tokenizer_v3.json"):
    print("=" * 80)
    print(f"TRAINING AR.I.S TOKENIZER V3 (Vocab: {vocab_size:,})")
    print("=" * 80)

    # Pre-tokenizer: digits split individually + byte level (no prefix space)
    pre_tok = Sequence([
        Digits(individual_digits=True),
        ByteLevel(add_prefix_space=False),
    ])

    tok = Tokenizer(BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tok
    tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    data_stream = stream_multidomain_training_data()
    tok.train_from_iterator(data_stream, trainer=trainer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(output_path))
    print(f"\nSaved Tokenizer V3 to: {output_path}")

    # Verify key tokens
    print("\nVERIFYING SPECIAL TOKENS IN V3:")
    for t in SPECIAL_TOKENS[:17]:
        tid = tok.token_to_id(t)
        print(f"  {t:<20} -> ID {tid}")

    # Verify lossless roundtrip
    test_texts = [
        "Hello, World! 123 + 456 = 579.",
        "def is_prime(n):\n    if n <= 1: return False\n    return True",
        "Janet's ducks lay 16 eggs per day. 🦆🥚",
        "தமிழ் மற்றும் ஹிந்தி",
        "<|im_start|>assistant\n<|thought|>\nAnalyzing prime algorithm.<|thought_end|>\n<|tool_call|>\n{\"name\": \"run_code\"}<|im_end|>\n"
    ]

    print("\nVERIFYING LOSSLESS ROUNDTRIP:")
    for text in test_texts:
        enc = tok.encode(text)
        dec = tok.decode(enc.ids, skip_special_tokens=False)
        assert dec == text, f"Roundtrip failed!\nOriginal: {repr(text)}\nDecoded:  {repr(dec)}"
        print(f"  Passed ({len(enc.ids)} tokens): {repr(text[:45])}...")

    # Calculate SHA256
    h = hashlib.sha256()
    with open(output_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    print(f"\nTokenizer V3 SHA256: {h.hexdigest()}")
    print("=" * 80)


if __name__ == "__main__":
    train_tokenizer_v3()
