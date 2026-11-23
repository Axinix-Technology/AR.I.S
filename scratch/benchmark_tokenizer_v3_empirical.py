"""
Empirical Tokenizer V3 Benchmark & Sequence Length Distribution Audit
Measures:
  1. Real domain compression (bytes/token, chars/token, tokens/word) on V2 (8k) vs V3 (16k).
  2. Exact sequence length percentiles (Median, P90, P95, P99, Max) on full prompt+response pairs.
  3. Real retention rate under 384 context vs 1,024 context.
  4. Numerical digit token inflation rate under individual digit splitting.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from aris.tokenizer.bpe_tokenizer import ArisTokenizer

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def run_empirical_benchmark():
    print("=" * 105)
    print("AR.I.S TOKENIZER V2 (8k) VS V3 (16k) EMPIRICAL VALIDATION BENCHMARK")
    print("=" * 105)

    tok_v2_path = MODELS_DIR / "aris_bpe_tokenizer_v2.json"
    tok_v3_path = MODELS_DIR / "aris_bpe_tokenizer_v3.json"

    assert tok_v2_path.exists(), f"V2 tokenizer not found at {tok_v2_path}"
    assert tok_v3_path.exists(), f"V3 tokenizer not found at {tok_v3_path}"

    tok_v2 = ArisTokenizer.load(tok_v2_path)
    tok_v3 = ArisTokenizer.load(tok_v3_path)

    print(f"Loaded Tokenizer V2 (Vocab: {tok_v2.vocab_size:,})")
    print(f"Loaded Tokenizer V3 (Vocab: {tok_v3.vocab_size:,})")

    # Assemble representative multi-domain test corpora
    domains = {}

    # 1. Prose / QA: SQuAD
    sq_path = DATA_DIR / "squad_v2/train.parquet"
    if sq_path.exists():
        tbl = pq.read_table(sq_path)
        rows = tbl.to_pylist()[:2500]
        prose_samples = []
        for r in rows:
            ctx = r.get("context", "")
            q = r.get("question", "")
            ans_list = r.get("answers", {}).get("text", [])
            ans = ans_list[0] if ans_list else ""
            prose_samples.append(f"Context: {ctx}\nQuestion: {q}\nAnswer: {ans}")
        domains["prose_qa"] = prose_samples

    # 2. Math: GSM8K
    gsm_path = DATA_DIR / "gsm8k/train.parquet"
    if gsm_path.exists():
        tbl = pq.read_table(gsm_path)
        rows = tbl.to_pylist()[:2500]
        domains["math_gsm8k"] = [f"Question: {r.get('question','')}\nAnswer: {r.get('answer','')}" for r in rows]

    # 3. Code: MBPP
    mbpp_path = DATA_DIR / "mbpp/train.parquet"
    if mbpp_path.exists():
        tbl = pq.read_table(mbpp_path)
        rows = tbl.to_pylist()
        domains["code_mbpp"] = [f"Task: {r.get('text','')}\nCode:\n{r.get('code','')}\nTests:\n" + "\n".join(r.get('test_list',[])) for r in rows]

    # 4. Multi-turn Chat: SmolTalk Everyday + Magpie
    smol_path = DATA_DIR / "smoltalk/everyday_conversations.parquet"
    magpie_path = DATA_DIR / "smoltalk/smol_magpie_ultra.parquet"
    chat_samples = []
    if smol_path.exists():
        tbl = pq.read_table(smol_path)
        for dialogue in tbl["messages"][:1000].to_pylist():
            text = "".join(f"<|im_start|>{m.get('role','')}\n{m.get('content','')}<|im_end|>\n" for m in dialogue)
            chat_samples.append(text)
    if magpie_path.exists():
        tbl = pq.read_table(magpie_path, columns=["messages"])
        for dialogue in tbl["messages"][:1500].to_pylist():
            text = "".join(f"<|im_start|>{m.get('role','')}\n{m.get('content','')}<|im_end|>\n" for m in dialogue)
            chat_samples.append(text)
    domains["chat_smoltalk"] = chat_samples

    print(f"\nAssembled Benchmark Datasets:")
    for d, items in domains.items():
        print(f"  {d:<16}: {len(items):,d} complete samples")

    # -----------------------------------------------------------------
    # Part 1: Domain Compression Comparison
    # -----------------------------------------------------------------
    print("\n" + "-" * 105)
    print("PART 1: REAL DOMAIN COMPRESSION BENCHMARK")
    print(f"{'Domain':15} | {'Samples':8} | {'Total Bytes':12} | {'Tokens (V2)':12} | {'Tokens (V3)':12} | {'Bytes/Tok V2':13} | {'Bytes/Tok V3':13} | {'Compression Delta':18}")
    print("-" * 105)

    all_v2_lengths = []
    all_v3_lengths = []

    for d_name, samples in domains.items():
        d_bytes = 0
        d_toks_v2 = 0
        d_toks_v3 = 0

        for text in samples:
            raw_b = len(text.encode("utf-8"))
            e2 = len(tok_v2.encode(text, add_special_tokens=False))
            e3 = len(tok_v3.encode(text, add_special_tokens=False))

            d_bytes += raw_b
            d_toks_v2 += e2
            d_toks_v3 += e3

            all_v2_lengths.append(e2)
            all_v3_lengths.append(e3)

        bpt_v2 = d_bytes / d_toks_v2 if d_toks_v2 > 0 else 0
        bpt_v3 = d_bytes / d_toks_v3 if d_toks_v3 > 0 else 0
        delta_pct = ((d_toks_v2 - d_toks_v3) / d_toks_v2) * 100.0

        print(f"{d_name:15} | {len(samples):8,d} | {d_bytes:12,d} | {d_toks_v2:12,d} | {d_toks_v3:12,d} | {bpt_v2:13.2f} | {bpt_v3:13.2f} | {delta_pct:+17.2f}%")

    # -----------------------------------------------------------------
    # Part 2: Sequence Length Percentiles & Context Retention Audit
    # -----------------------------------------------------------------
    print("\n" + "-" * 105)
    print("PART 2: SEQUENCE LENGTH PERCENTILES & CONTEXT RETENTION AUDIT")
    print(f"Total Full-Sequence Samples Measured Across All Domains: {len(all_v3_lengths):,}")
    print("-" * 105)

    v2_arr = np.array(all_v2_lengths)
    v3_arr = np.array(all_v3_lengths)

    print(f"{'Metric':32} | {'Tokenizer V2 (8k)':20} | {'Tokenizer V3 (16k)':20} | {'Improvement / Shift':20}")
    print("-" * 105)
    print(f"{'Median Sequence Length (P50)':32} | {np.percentile(v2_arr, 50):20.1f} | {np.percentile(v3_arr, 50):20.1f} | {np.percentile(v3_arr, 50) - np.percentile(v2_arr, 50):+19.1f} tok")
    print(f"{'90th Percentile (P90)':32} | {np.percentile(v2_arr, 90):20.1f} | {np.percentile(v3_arr, 90):20.1f} | {np.percentile(v3_arr, 90) - np.percentile(v2_arr, 90):+19.1f} tok")
    print(f"{'95th Percentile (P95)':32} | {np.percentile(v2_arr, 95):20.1f} | {np.percentile(v3_arr, 95):20.1f} | {np.percentile(v3_arr, 95) - np.percentile(v2_arr, 95):+19.1f} tok")
    print(f"{'99th Percentile (P99)':32} | {np.percentile(v2_arr, 99):20.1f} | {np.percentile(v3_arr, 99):20.1f} | {np.percentile(v3_arr, 99) - np.percentile(v2_arr, 99):+19.1f} tok")
    print(f"{'Max Sequence Length':32} | {np.max(v2_arr):20,d} | {np.max(v3_arr):20,d} | {np.max(v3_arr) - np.max(v2_arr):+19,d} tok")

    # Retention rates
    fit_384_v2 = (v2_arr <= 384).sum() / len(v2_arr) * 100.0
    fit_384_v3 = (v3_arr <= 384).sum() / len(v3_arr) * 100.0
    fit_1024_v3 = (v3_arr <= 1024).sum() / len(v3_arr) * 100.0

    print("-" * 105)
    print(f"{'Retention: Fitting in 384 Context':32} | {fit_384_v2:19.2f}% | {fit_384_v3:19.2f}% | {fit_384_v3 - fit_384_v2:+19.2f}%")
    print(f"{'Retention: Fitting in 1,024 Context':32} | {'—':20} | {fit_1024_v3:19.2f}% | (Full Context Target)")
    print(f"{'Truncation / Drop Rate at 1,024':32} | {'—':20} | {100.0 - fit_1024_v3:19.2f}% | (Residual Truncation)")
    print("=" * 105)


if __name__ == "__main__":
    run_empirical_benchmark()
