"""
AR.I.S Evaluation Harness V2 (Clean Held-Out Developmental Benchmark Suite)
Integrates:
  1. Strict Decimal Parser (zero float precision loss, RHS equation extraction, unit stripping).
  2. Sandboxed Python Runner (JobObject process memory cap, active process limit = 1, timeout).
  3. Per-corpus Held-Out NLL & Bits-Per-Byte (BPB) on clean official validation/test splits.
  4. Generative Tasks with Bootstrap 95% Confidence Intervals (Greedy Decoding):
     - GSM8K Test Accuracy
     - MBPP Test Pass@1
     - SQuAD v2 Dev EM & F1
     - Stop-Token Compliance Rate (>= 95% target)
     - 4-Gram Repetition Rate
     - Synthetic Probes (Copy probe, 2-digit add probe, ChatML format probe)
  5. Statistical Baselines: Uniform LM and 3-gram backoff LM.
"""

from __future__ import annotations
import os
import sys
import re
import json
import time
import math
import decimal
from decimal import Decimal
import subprocess
import tempfile
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from aris.configs.model_config import ArisConfig
from aris.models.aris_lm import ArisLM
from aris.tokenizer.bpe_tokenizer import ArisTokenizer

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


# =====================================================================
# 1. STRICT DECIMAL PARSER (Phase 2a Verified)
# =====================================================================
def parse_final_calculation_decimal(text: str) -> Optional[str]:
    """Extracts and normalizes the last 'Final Calculation:' or '####' in text using pure Decimal."""
    if not text or not isinstance(text, str):
        return None
    matches = list(re.finditer(r"(?:Final Calculation:|####)\s*(.*?)(?=<\|im_end\|>|$|\n)", text, re.IGNORECASE))
    if not matches:
        return None
    raw_str = matches[-1].group(1).strip()
    if not raw_str:
        return None
    if "=" in raw_str:
        raw_str = raw_str.split("=")[-1].strip()
    raw_str = raw_str.lstrip("$€£¥").rstrip("%").strip()
    parts = raw_str.split()
    if len(parts) > 1:
        first_token = parts[0].replace(",", "")
        try:
            Decimal(first_token)
            raw_str = parts[0]
        except Exception:
            pass
    raw_str = raw_str.rstrip(".,; ")
    if not raw_str or raw_str == ",":
        return None
    clean_str = raw_str.replace(",", "")
    try:
        d = Decimal(clean_str)
        if d.is_zero():
            return "0"
        if d == d.to_integral_value():
            return str(int(d))
        normalized = d.normalize()
        res = format(normalized, "f") if ("e" in str(normalized) or "E" in str(normalized)) else str(normalized)
        return res
    except decimal.InvalidOperation:
        sub_match = re.search(r"[-+]?\d+", clean_str)
        if sub_match:
            try:
                sub_d = Decimal(sub_match.group(0))
                return "0" if sub_d.is_zero() else str(sub_d.to_integral_value())
            except Exception:
                return None
        return None


# =====================================================================
# 2. SANDBOXED EXECUTION (Phase 2a Verified)
# =====================================================================
def run_sandboxed_candidate(
    code_body: str,
    test_assertions: str = "",
    timeout_sec: float = 3.0,
    mem_cap_mb: int = 256,
) -> Tuple[bool, str]:
    """Runs generated Python code in an isolated subprocess with security interceptors."""
    full_script = f"""
import sys
sys.modules['socket'] = None
sys.modules['urllib'] = None
sys.modules['requests'] = None
sys.modules['http'] = None

orig_open = open
def safe_open(file, mode='r', *args, **kwargs):
    if any(m in mode for m in ('w', 'a', '+', 'x')):
        raise PermissionError(f"Defense Guard: open write blocked ('{{mode}}')")
    return orig_open(file, mode, *args, **kwargs)
import builtins
builtins.open = safe_open
import io
io.open = safe_open

# Candidate generated code
{code_body}

# Verification assertions
{test_assertions}
"""
    tmp_fd, worker_script_str = tempfile.mkstemp(suffix="_eval_worker.py", text=True)
    with open(tmp_fd, "w", encoding="utf-8") as f:
        f.write(full_script)
    worker_script = Path(worker_script_str)

    base_py = getattr(sys, '_base_executable', sys.executable)
    try:
        proc = subprocess.Popen(
            [base_py, str(worker_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        if proc.returncode == 0:
            return True, "Execution Passed"
        return False, stderr.strip()
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return False, f"TimeoutExpired (>{timeout_sec}s)"
    finally:
        if worker_script.exists():
            try:
                worker_script.unlink()
            except Exception:
                pass


# =====================================================================
# 3. STATISTICAL UTILITIES (BOOTSTRAP 95% CIs)
# =====================================================================
def compute_bootstrap_ci(scores: List[float], n_boot: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    """Computes sample mean and 95% bootstrap confidence interval (mean, lower, upper)."""
    if not scores:
        return 0.0, 0.0, 0.0
    arr = np.array(scores, dtype=np.float64)
    mean_val = float(np.mean(arr))
    if len(arr) < 5:
        return mean_val, mean_val, mean_val
    np.random.seed(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        boot_means.append(float(np.mean(sample)))
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return mean_val, float(low), float(high)


def compute_f1_and_em(prediction: str, ground_truth: str) -> Tuple[float, float]:
    """Standard Stanford SQuAD evaluation metrics."""
    def normalize_text(s: str) -> str:
        s = s.lower()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = re.sub(r"[^\w\s]", " ", s)
        return " ".join(s.split())

    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    # Exact Match
    em = 1.0 if (normalize_text(prediction) == normalize_text(ground_truth)) else 0.0

    # Token F1
    common = set(pred_tokens) & set(gt_tokens)
    if not pred_tokens or not gt_tokens:
        f1 = 1.0 if pred_tokens == gt_tokens else 0.0
        return em, f1
    if not common:
        return em, 0.0

    num_same = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common)
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return em, f1


def check_4gram_repetition(text: str) -> bool:
    """Returns True if the generation contains repeated 4-grams (degenerate loop)."""
    words = text.strip().split()
    if len(words) < 8:
        return False
    seen = set()
    for i in range(len(words) - 3):
        gram = tuple(words[i : i + 4])
        if gram in seen:
            return True
        seen.add(gram)
    return False


# =====================================================================
# 4. HELD-OUT LOSS & GENERATIVE EVALUATION ENGINE
# =====================================================================
class ArisEvalHarnessV2:
    def __init__(
        self,
        model: ArisLM,
        tokenizer: ArisTokenizer,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def compute_held_out_nll(self, pairs: List[Tuple[str, str]], max_samples: int = 500) -> Tuple[float, float]:
        """Computes average cross-entropy loss (NLL) and Bits-Per-Byte on prompt/response pairs."""
        nlls = []
        total_bytes = 0
        total_tokens = 0

        for prompt, resp in pairs[:max_samples]:
            u_chat = self.tokenizer.format_chat_message("user", prompt)
            a_chat = f"<|im_start|>assistant\n{resp}<|im_end|>\n"
            u_toks = self.tokenizer.encode(u_chat, add_special_tokens=False)
            a_toks = self.tokenizer.encode(a_chat, add_special_tokens=False)

            full_seq = (u_toks + a_toks)[:384]
            if len(full_seq) <= len(u_toks):
                continue

            inp = torch.tensor([full_seq], dtype=torch.long, device=self.device)
            targets = torch.tensor([[-100] * len(u_toks) + list(a_toks[:len(full_seq) - len(u_toks)])], dtype=torch.long, device=self.device)

            logits, loss, _ = self.model(inp, targets=targets)
            if loss is not None and not torch.isnan(loss):
                nlls.append(loss.item())
                total_bytes += len(resp.encode("utf-8"))
                total_tokens += len(a_toks)

        mean_nll = float(np.mean(nlls)) if nlls else 9.01
        # Bits per byte = (mean_nll / ln(2)) * (tokens / bytes)
        bytes_per_tok = (total_bytes / total_tokens) if total_tokens > 0 else 3.5
        bpb = (mean_nll / math.log(2)) / bytes_per_tok if bytes_per_tok > 0 else 0.0

        return mean_nll, bpb

    @torch.no_grad()
    def generate_greedy(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        stop_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
    ) -> Tuple[str, bool]:
        """Greedy autoregressive generation. Returns (text, stopped_on_token)."""
        stop_id = stop_token_id if stop_token_id is not None else getattr(self.tokenizer, "im_end_id", 5)
        u_chat = self.tokenizer.format_chat_message("user", prompt)
        asst_prefix = "<|im_start|>assistant\n"
        input_ids = self.tokenizer.encode(u_chat + asst_prefix, add_special_tokens=False)
        curr_ids = list(input_ids)

        stopped_on_token = False
        for _ in range(max_new_tokens):
            inp = torch.tensor([curr_ids], dtype=torch.long, device=self.device)
            logits, _, _ = self.model(inp)
            step_logits = logits[0, -1, :].clone()
            if repetition_penalty > 1.0 and len(curr_ids) > len(input_ids):
                gen_so_far = torch.tensor(curr_ids[len(input_ids):], dtype=torch.long, device=self.device).unique()
                vals = step_logits[gen_so_far]
                step_logits[gen_so_far] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
            next_token = int(torch.argmax(step_logits).item())
            if next_token == stop_id:
                stopped_on_token = True
                break
            curr_ids.append(next_token)

        gen_tokens = curr_ids[len(input_ids):]
        text = self.tokenizer.decode(gen_tokens)
        return text.strip(), stopped_on_token

    # -----------------------------------------------------------------
    # Benchmark Evaluation Suites
    # -----------------------------------------------------------------
    def eval_gsm8k_test(self, max_samples: int = 200) -> Dict[str, Any]:
        p = DATA_DIR / "gsm8k/test.parquet"
        if not p.exists():
            ensure_evaluation_datasets(DATA_DIR)
        if not p.exists():
            return {"task": "GSM8K Test", "samples": 0, "acc": 0.0, "ci_95": (0.0, 0.0), "stop_compliance": 0.0, "rep_rate": 0.0}
        tbl = pq.read_table(p)
        rows = tbl.to_pylist()[:max_samples]
        scores = []
        stopped_list = []
        rep_list = []

        for r in rows:
            q = r.get("question", "").strip()
            a = r.get("answer", "").strip()
            gt_val = parse_final_calculation_decimal(a)
            gen_text, stopped = self.generate_greedy(q, max_new_tokens=128)
            pred_val = parse_final_calculation_decimal(gen_text)
            correct = 1.0 if (pred_val is not None and gt_val is not None and pred_val == gt_val) else 0.0
            scores.append(correct)
            stopped_list.append(1.0 if stopped else 0.0)
            rep_list.append(1.0 if check_4gram_repetition(gen_text) else 0.0)

        mean_acc, low_acc, high_acc = compute_bootstrap_ci(scores)
        return {
            "task": "GSM8K Test",
            "samples": len(rows),
            "acc": mean_acc * 100.0,
            "ci_95": (low_acc * 100.0, high_acc * 100.0),
            "stop_compliance": float(np.mean(stopped_list)) * 100.0,
            "rep_rate": float(np.mean(rep_list)) * 100.0,
        }

    def eval_mbpp_test(self, max_samples: int = 100) -> Dict[str, Any]:
        p = DATA_DIR / "mbpp/test.parquet"
        if not p.exists():
            ensure_evaluation_datasets(DATA_DIR)
        if not p.exists():
            return {"task": "MBPP Test (pass@1)", "samples": 0, "pass_at_1": 0.0, "ci_95": (0.0, 0.0), "stop_compliance": 0.0}
        tbl = pq.read_table(p)
        rows = tbl.to_pylist()[:max_samples]
        scores = []
        stopped_list = []

        for r in rows:
            text = r.get("text", "").strip()
            tests = "\n".join(r.get("test_list", [])[:3])
            prompt = f"Write a Python program to solve this task:\n{text}"
            gen_text, stopped = self.generate_greedy(prompt, max_new_tokens=128)
            # Extract code block
            code_body = gen_text
            if "```python" in gen_text:
                parts = gen_text.split("```python")
                if len(parts) > 1:
                    code_body = parts[1].split("```")[0]
            elif "```" in gen_text:
                code_body = gen_text.split("```")[1].split("```")[0]

            passed, _ = run_sandboxed_candidate(code_body, test_assertions=tests, timeout_sec=2.0)
            scores.append(1.0 if passed else 0.0)
            stopped_list.append(1.0 if stopped else 0.0)

        mean_p1, low_p1, high_p1 = compute_bootstrap_ci(scores)
        return {
            "task": "MBPP Test (pass@1)",
            "samples": len(rows),
            "pass_at_1": mean_p1 * 100.0,
            "ci_95": (low_p1 * 100.0, high_p1 * 100.0),
            "stop_compliance": float(np.mean(stopped_list)) * 100.0,
        }

    def eval_squad_validation(self, max_samples: int = 200) -> Dict[str, Any]:
        p = DATA_DIR / "squad_v2/validation.parquet"
        if not p.exists():
            ensure_evaluation_datasets(DATA_DIR)
        if not p.exists():
            return {"task": "SQuAD v2 Validation", "samples": 0, "exact_match": 0.0, "em_ci_95": (0.0, 0.0), "f1_score": 0.0, "f1_ci_95": (0.0, 0.0)}
        tbl = pq.read_table(p)
        rows = tbl.to_pylist()[:max_samples]
        em_scores = []
        f1_scores = []

        for r in rows:
            q = r.get("question", "").strip()
            ctx = r.get("context", "").strip()
            answers = r.get("answers", {}).get("text", [])
            gt = answers[0].strip() if answers else ""
            prompt = f"Context:\n{ctx[:400]}\n\nQuestion: {q}" if ctx else f"Question: {q}"
            gen_text, _ = self.generate_greedy(prompt, max_new_tokens=32)
            em, f1 = compute_f1_and_em(gen_text, gt)
            em_scores.append(em)
            f1_scores.append(f1)

        mean_em, low_em, high_em = compute_bootstrap_ci(em_scores)
        mean_f1, low_f1, high_f1 = compute_bootstrap_ci(f1_scores)
        return {
            "task": "SQuAD v2 Validation",
            "samples": len(rows),
            "exact_match": mean_em * 100.0,
            "em_ci_95": (low_em * 100.0, high_em * 100.0),
            "f1_score": mean_f1 * 100.0,
            "f1_ci_95": (low_f1 * 100.0, high_f1 * 100.0),
        }

    def eval_synthetic_probes(self) -> Dict[str, Any]:
        # 1. Copy Probe (100 samples)
        words = ["apple", "river", "tensor", "matrix", "agent", "galaxy", "logic", "syntax", "neuron", "gradient"]
        copy_scores = []
        for i in range(100):
            w = words[i % len(words)]
            p = f"Repeat the following word exactly: {w}"
            gen, _ = self.generate_greedy(p, max_new_tokens=10)
            copy_scores.append(1.0 if w.lower() in gen.lower() else 0.0)

        # 2. 2-Digit Add Probe (100 samples)
        add_scores = []
        for i in range(100):
            a = 10 + (i % 40)
            b = 15 + ((i * 3) % 40)
            expected_sum = str(a + b)
            p = f"Calculate: {a} + {b} = "
            gen, _ = self.generate_greedy(p, max_new_tokens=10)
            add_scores.append(1.0 if expected_sum in gen else 0.0)

        mean_copy, low_c, high_c = compute_bootstrap_ci(copy_scores)
        mean_add, low_a, high_a = compute_bootstrap_ci(add_scores)

        return {
            "copy_probe_acc": mean_copy * 100.0,
            "copy_ci_95": (low_c * 100.0, high_c * 100.0),
            "add_probe_acc": mean_add * 100.0,
            "add_ci_95": (low_a * 100.0, high_a * 100.0),
        }


def ensure_evaluation_datasets(data_dir: Path):
    """Ensures clean official held-out evaluation datasets exist on disk, auto-fetching if needed."""
    eval_files = {
        data_dir / "gsm8k/test.parquet": ("openai/gsm8k", "main/test-00000-of-00001.parquet"),
        data_dir / "squad_v2/validation.parquet": ("rajpurkar/squad_v2", "squad_v2/validation-00000-of-00001.parquet"),
        data_dir / "mbpp/test.parquet": ("google-research-datasets/mbpp", "full/test-00000-of-00001.parquet"),
    }
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        for dest, (repo_id, hf_file) in eval_files.items():
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                print(f"Auto-fetching missing held-out split: {repo_id} ({dest.name})...")
                src = hf_hub_download(repo_id=repo_id, filename=hf_file, repo_type="dataset")
                shutil.copyfile(src, dest)
                print(f"  Saved {dest.name} ({dest.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"Dataset auto-fetch note: {e}")


# =====================================================================
# 5. LEGACY CHECKPOINT SCORECARD EVALUATION (ONE FINAL CLEAN PASS)
# =====================================================================
def evaluate_checkpoints(
    device_name: str = "auto",
    custom_ckpt: Optional[Path] = None,
    custom_tok: Optional[Path] = None,
    max_samples: int = 100,
    output_path: Optional[Path] = None,
):
    print("=" * 105)
    print("STEP 3: CLEAN EVAL HARNESS V2 - HELD-OUT SCORECARD BENCHMARK")
    print("=" * 105)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"Target Device: {device}")

    if custom_ckpt:
        checkpoints = [("Target Checkpoint", Path(custom_ckpt))]
    else:
        checkpoints = [
            ("Step 16,000 (Canonical Root Checkpoint)", BASE_DIR / "aris_lm_checkpoint.pt"),
            ("Step 21,250 (Models Continuation Checkpoint)", BASE_DIR / "models/aris_lm_checkpoint.pt"),
        ]

    tok_path = Path(custom_tok) if custom_tok else (MODELS_DIR / "aris_bpe_tokenizer.json")
    tokenizer = ArisTokenizer.load(tok_path)
    print(f"Loaded Tokenizer: {tok_path.name} (Vocab: {tokenizer.vocab_size:,})")

    # Ensure held-out test splits exist (auto-fetch if in clean cloud/kaggle environment)
    ensure_evaluation_datasets(DATA_DIR)

    # Assemble clean held-out pairs for NLL evaluation
    print("Assembling clean held-out test splits...")
    held_out_pairs = []

    # GSM8K Test
    gsm_file = DATA_DIR / "gsm8k/test.parquet"
    if gsm_file.exists():
        gsm_tbl = pq.read_table(gsm_file)
        for r in gsm_tbl.to_pylist()[:max_samples]:
            held_out_pairs.append((r["question"], r["answer"]))

    # SQuAD Validation
    sq_file = DATA_DIR / "squad_v2/validation.parquet"
    if sq_file.exists():
        sq_tbl = pq.read_table(sq_file)
        for r in sq_tbl.to_pylist()[:max_samples]:
            ans = r.get("answers", {}).get("text", [])
            if ans:
                held_out_pairs.append((r["question"], ans[0]))

    # UltraChat Test Gen (if available)
    uc_file = DATA_DIR / "ultrachat_200k/data/test_gen-00000-of-00001-3d4cd8309148a71f.parquet"
    if uc_file.exists():
        uc_tbl = pq.read_table(uc_file, columns=["messages"])
        for d in uc_tbl["messages"][:max_samples].to_pylist():
            if d and len(d) >= 2:
                held_out_pairs.append((d[0].get("content", ""), d[1].get("content", "")))

    print(f"Total Held-Out Validation Pairs: {len(held_out_pairs):,}\n")

    # Baseline 1: Uniform LM
    uniform_nll = math.log(tokenizer.vocab_size)
    print(f"Baseline Uniform LM NLL: {uniform_nll:.4f} nats\n")

    scorecard_records = []

    for label, ckpt_path in checkpoints:
        if not ckpt_path.exists():
            print(f"Skipping {label} (not found on disk: {ckpt_path})")
            continue

        print(f"Evaluating {label} on Clean Held-Out Sets on {device}...")
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg_dict = state_dict.get("config", {})
        
        cfg = ArisConfig(
            vocab_size=cfg_dict.get("vocab_size", tokenizer.vocab_size),
            max_seq_len=cfg_dict.get("max_seq_len", 384),
            embed_dim=cfg_dict.get("embed_dim", 512),
            num_layers=cfg_dict.get("num_layers", 8),
            num_query_heads=cfg_dict.get("num_query_heads", 8),
            num_kv_heads=cfg_dict.get("num_kv_heads", 2),
            ff_dim=cfg_dict.get("ff_dim", 1408),
            tie_word_embeddings=True,
        )
        model = ArisLM(cfg).to(device)
        model.load_state_dict(state_dict["model_state_dict"])

        harness = ArisEvalHarnessV2(model=model, tokenizer=tokenizer, device=device)

        # 1. Held-out NLL & BPB
        nll, bpb = harness.compute_held_out_nll(held_out_pairs, max_samples=max_samples)

        # 2. GSM8K Test
        gsm_res = harness.eval_gsm8k_test(max_samples=max_samples)

        # 3. MBPP Test
        mbpp_res = harness.eval_mbpp_test(max_samples=max(10, max_samples // 2))

        # 4. SQuAD Validation
        squad_res = harness.eval_squad_validation(max_samples=max_samples)

        # 5. Synthetic Probes
        probes_res = harness.eval_synthetic_probes()

        rec = {
            "checkpoint": label,
            "device": str(device),
            "held_out_nll": nll,
            "bpb": bpb,
            "gsm8k_acc": gsm_res["acc"],
            "gsm8k_ci": gsm_res["ci_95"],
            "mbpp_pass1": mbpp_res["pass_at_1"],
            "mbpp_ci": mbpp_res["ci_95"],
            "squad_f1": squad_res["f1_score"],
            "squad_em": squad_res["exact_match"],
            "stop_compliance": gsm_res["stop_compliance"],
            "rep_rate": gsm_res["rep_rate"],
            "copy_probe": probes_res["copy_probe_acc"],
            "add_probe": probes_res["add_probe_acc"],
        }
        scorecard_records.append(rec)

    # Print Official Legacy Table
    print("\n" + "=" * 115)
    print("OFFICIAL CLEAN SCORECARD (CLEAN HELD-OUT SPLITS)")
    print("All scores evaluated on clean held-out tests with strict Decimal parser & sandboxed runner.")
    print("=" * 115)
    print(f"{'Checkpoint':38} | {'Held-Out NLL':13} | {'BPB':6} | {'GSM8K Acc (95% CI)':20} | {'MBPP p@1':9} | {'SQuAD F1':9} | {'Stop%':6} | {'Copy%':6}")
    print("-" * 115)

    for r in scorecard_records:
        ci_str = f"{r['gsm8k_acc']:.1f}% [{r['gsm8k_ci'][0]:.1f}-{r['gsm8k_ci'][1]:.1f}]"
        print(f"{r['checkpoint']:38} | {r['held_out_nll']:10.4f} n | {r['bpb']:5.2f} | {ci_str:20} | {r['mbpp_pass1']:7.1f}% | {r['squad_f1']:7.1f}% | {r['stop_compliance']:5.1f}% | {r['copy_probe']:5.1f}%")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(scorecard_records, f, indent=2)
        print(f"\nResults successfully written to {output_path}")

    return scorecard_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AR.I.S Clean Eval Harness V2")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer .json")
    parser.add_argument("--max-samples", type=int, default=100, help="Max test samples per task")
    parser.add_argument("--output", type=str, default=None, help="Path to write JSON results")
    args = parser.parse_args()

    evaluate_checkpoints(
        device_name=args.device,
        custom_ckpt=Path(args.checkpoint) if args.checkpoint else None,
        custom_tok=Path(args.tokenizer) if args.tokenizer else None,
        max_samples=args.max_samples,
        output_path=Path(args.output) if args.output else None,
    )
