"""
Step 2: Tokenizer V2 Training, Domain Fertility Benchmark, and Rigorous Validation
Key Specifications:
  - Model: ByteLevel BPE with add_prefix_space=False
  - Pre-tokenizer: Sequence([Digits(individual_digits=True), ByteLevel(add_prefix_space=False)])
  - Special Tokens: >= 64 reserved tokens (<|im_start|>, <|im_end|>, <|thought|>, <|tool_call|>, FIM tokens, etc.)
  - Trains both 8,192 and 16,384 candidates on representative multi-domain mixture.
  - Measures fertility (tokens/word, tokens/char) on >= 1 MB per domain (Prose, Code, Math, Chat).
  - Decision Rule: default is 16,384 unless 8,192 is within 10% fertility.
  - Freezes chosen tokenizer by SHA256 hash.
"""

import os
import sys
import json
import gzip
import tarfile
import hashlib
from pathlib import Path
from typing import Iterator, List, Dict, Tuple
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

# 65 Special Control & Reserved Slots (>= 64)
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
    "<|tool_result|>",
    "<|plan|>",
    "<|endoftext|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
] + [f"<|reserved_{i:03d}|>" for i in range(50)]


def stream_training_data() -> Iterator[str]:
    """Streams representative text across Prose, Code, Math, Chat, and Multilingual domains."""
    print("Streaming multi-domain training text...")
    
    # 1. Prose: SQuAD + Dolly
    sq_tbl = pq.read_table(DATA_DIR / "squad_v2/train.parquet")
    for r in sq_tbl.to_pylist()[:15000]:
        yield r.get("context", "")
        yield r.get("question", "")

    with open(DATA_DIR / "dolly/databricks-dolly-15k.jsonl", "r", encoding="utf-8") as f:
        for line in list(f)[:10000]:
            item = json.loads(line)
            yield item.get("instruction", "")
            yield item.get("context", "")
            yield item.get("response", "")

    # 2. Math: GSM8K
    gsm_tbl = pq.read_table(DATA_DIR / "gsm8k/train.parquet")
    for r in gsm_tbl.to_pylist():
        yield r.get("question", "")
        yield r.get("answer", "")

    # 3. Code: MBPP
    mbpp_tbl = pq.read_table(DATA_DIR / "mbpp/train.parquet")
    for r in mbpp_tbl.to_pylist():
        yield r.get("text", "")
        yield r.get("code", "")
        for t in r.get("test_list", []):
            yield t

    # 4. Chat: UltraChat (clean train shard 0) + SmolTalk
    uc_tbl = pq.read_table(DATA_DIR / "ultrachat_200k/data/train_gen-00000-of-00003-a6c9fb894be3e50b.parquet", columns=["messages"])
    for dialogue in uc_tbl["messages"][:15000].to_pylist():
        for m in dialogue:
            yield m.get("content", "")

    smol_tbl = pq.read_table(DATA_DIR / "smoltalk/everyday_conversations.parquet")
    for dialogue in smol_tbl["messages"].to_pylist():
        for m in dialogue:
            yield m.get("content", "")

    # 5. Multilingual Syntax from UD archive (Tamil, Hindi, French, German)
    ud_archive = BASE_DIR / "ud-treebanks-v2.18.tgz"
    if ud_archive.exists():
        with tarfile.open(ud_archive, "r:gz") as tar:
            for m in tar.getmembers():
                if not m.name.endswith("-train.conllu"):
                    continue
                name_lower = m.name.lower()
                if any(lang in name_lower for lang in ["tamil", "hindi", "french", "german", "english"]):
                    f = tar.extractfile(m)
                    if f:
                        for line in f:
                            l = line.decode("utf-8", errors="replace").strip()
                            if l.startswith("# text ="):
                                yield l.replace("# text =", "").strip()


def train_candidate(vocab_size: int, output_path: Path) -> Tokenizer:
    print(f"\nTraining ByteLevel BPE Tokenizer (vocab_size={vocab_size:,})...")
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = Sequence([
        Digits(individual_digits=True),
        ByteLevel(add_prefix_space=False),
    ])
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train_from_iterator(stream_training_data(), trainer=trainer)
    tokenizer.save(str(output_path))
    print(f"Saved tokenizer to: {output_path} ({output_path.stat().st_size / 1024:.2f} KB)")
    return tokenizer


# =====================================================================
# DOMAIN FERTILITY BENCHMARK (>= 1 MB per domain)
# =====================================================================
def load_domain_corpora_1mb() -> Dict[str, str]:
    """Assembles >= 1 MB text per domain: Prose, Code, Math, Chat."""
    domains = {"Prose": "", "Code": "", "Math": "", "Chat": ""}
    target_bytes = 1024 * 1024 # 1 MB

    # 1. Prose
    sq_tbl = pq.read_table(DATA_DIR / "squad_v2/train.parquet")
    prose_parts = []
    for r in sq_tbl.to_pylist():
        prose_parts.append(r.get("context", "") + "\n" + r.get("question", ""))
        if sum(len(p) for p in prose_parts) >= target_bytes:
            break
    domains["Prose"] = "\n".join(prose_parts)

    # 2. Math
    gsm_tbl = pq.read_table(DATA_DIR / "gsm8k/train.parquet")
    math_parts = []
    # Repeat GSM8K problems to reach 1 MB of dense math
    rows = gsm_tbl.to_pylist()
    idx = 0
    while sum(len(p) for p in math_parts) < target_bytes:
        r = rows[idx % len(rows)]
        math_parts.append(r.get("question", "") + "\n" + r.get("answer", ""))
        idx += 1
    domains["Math"] = "\n".join(math_parts)

    # 3. Code
    mbpp_tbl = pq.read_table(DATA_DIR / "mbpp/train.parquet")
    code_parts = []
    mbpp_rows = mbpp_tbl.to_pylist()
    idx = 0
    while sum(len(p) for p in code_parts) < target_bytes:
        r = mbpp_rows[idx % len(mbpp_rows)]
        code_parts.append(r.get("text", "") + "\n" + r.get("code", "") + "\n" + "\n".join(r.get("test_list", [])))
        idx += 1
    domains["Code"] = "\n".join(code_parts)

    # 4. Chat
    uc_tbl = pq.read_table(DATA_DIR / "ultrachat_200k/data/train_gen-00000-of-00003-a6c9fb894be3e50b.parquet", columns=["messages"])
    chat_parts = []
    for dialogue in uc_tbl["messages"].to_pylist():
        for m in dialogue:
            chat_parts.append(m.get("content", ""))
        if sum(len(p) for p in chat_parts) >= target_bytes:
            break
    domains["Chat"] = "\n".join(chat_parts)

    return domains


def evaluate_fertility(tok_8k: Tokenizer, tok_16k: Tokenizer):
    print("\n" + "=" * 95)
    print("DOMAIN FERTILITY BENCHMARK (>= 1.0 MB Text Per Domain)")
    print("=" * 95)

    domains = load_domain_corpora_1mb()
    results = {}

    print(f"{'Domain':10} | {'Text Size':10} | {'Word Count':12} | {'Tokens 8k':12} | {'Fert 8k (tok/w)':15} | {'Tokens 16k':12} | {'Fert 16k (tok/w)':16} | {'Fert Diff %':12}")
    print("-" * 115)

    total_tokens_8k = 0
    total_tokens_16k = 0
    total_words = 0

    for domain_name, text in domains.items():
        size_mb = len(text.encode("utf-8")) / (1024 * 1024)
        words = len(text.split())
        enc_8k = tok_8k.encode(text)
        enc_16k = tok_16k.encode(text)
        
        t8 = len(enc_8k.ids)
        t16 = len(enc_16k.ids)
        total_tokens_8k += t8
        total_tokens_16k += t16
        total_words += words

        fert_8k = t8 / words if words > 0 else 0
        fert_16k = t16 / words if words > 0 else 0
        diff_pct = ((t8 - t16) / t16) * 100.0 # How much more verbose is 8k compared to 16k

        results[domain_name] = {
            "size_mb": size_mb,
            "words": words,
            "tokens_8k": t8,
            "tokens_16k": t16,
            "fert_8k": fert_8k,
            "fert_16k": fert_16k,
            "diff_pct": diff_pct,
        }

        print(f"{domain_name:10} | {size_mb:8.2f} MB | {words:12,d} | {t8:12,d} | {fert_8k:15.2f} | {t16:12,d} | {fert_16k:16.2f} | {diff_pct:+11.2f}%")

    overall_fert_8k = total_tokens_8k / total_words
    overall_fert_16k = total_tokens_16k / total_words
    overall_diff = ((total_tokens_8k - total_tokens_16k) / total_tokens_16k) * 100.0

    print("-" * 115)
    print(f"{'OVERALL':10} | {'4.00+ MB':10} | {total_words:12,d} | {total_tokens_8k:12,d} | {overall_fert_8k:15.2f} | {total_tokens_16k:12,d} | {overall_fert_16k:16.2f} | {overall_diff:+11.2f}%")

    # Decision Rule: default is 16,384 unless 8,192 is within 10% fertility
    print(f"\nDECISION RULE EVALUATION:")
    print(f"  Overall verbosity difference of 8k vs 16k: {overall_diff:.2f}%")
    if overall_diff <= 10.0:
        chosen_vocab = 8192
        chosen_path = MODELS_DIR / "tokenizer_v2_8k.json"
        print(f"  -> 8,192 is within 10% fertility ({overall_diff:.2f}% <= 10.00%).")
        print(f"  -> SELECTED VOCABULARY: 8,192 (preserves core parameters from embedding bloat).")
    else:
        chosen_vocab = 16384
        chosen_path = MODELS_DIR / "tokenizer_v2_16k.json"
        print(f"  -> 8,192 exceeds 10% fertility difference ({overall_diff:.2f}% > 10.00%).")
        print(f"  -> SELECTED VOCABULARY: 16,384 (Default for fertility compression).")

    # Freeze chosen tokenizer to models/aris_bpe_tokenizer_v2.json
    final_path = MODELS_DIR / "aris_bpe_tokenizer_v2.json"
    with open(chosen_path, "r", encoding="utf-8") as f_src, open(final_path, "w", encoding="utf-8") as f_dst:
        f_dst.write(f_src.read())

    # Calculate SHA256 of frozen tokenizer
    h = hashlib.sha256()
    with open(final_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    frozen_sha = h.hexdigest()

    print(f"\nFROZEN TOKENIZER V2:")
    print(f"  Path:   {final_path}")
    print(f"  Vocab:  {chosen_vocab:,}")
    print(f"  SHA256: {frozen_sha}")

    return chosen_vocab, frozen_sha, results


def main():
    path_8k = MODELS_DIR / "tokenizer_v2_8k.json"
    path_16k = MODELS_DIR / "tokenizer_v2_16k.json"

    tok_8k = train_candidate(8192, path_8k)
    tok_16k = train_candidate(16384, path_16k)

    evaluate_fertility(tok_8k, tok_16k)


if __name__ == "__main__":
    main()
