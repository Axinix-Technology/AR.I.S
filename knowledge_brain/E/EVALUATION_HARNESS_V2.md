# Clean Evaluation Harness V2 & Developmental Benchmark Protocol

## 1. Overview
The **AR.I.S Clean Evaluation Harness V2** (`aris/evaluation/eval_harness_v2.py`) provides an unpolluted, reproducible developmental evaluation suite designed for rigorous assessment of foundational decoder models.

It eliminates the contamination and reproducibility pitfalls of legacy train-set evaluations by strictly evaluating models against official held-out test and validation splits.

---

## 2. Core Architectural Pillars

### A. Strict Decimal Parser
- **Mathematical Exactness**: Operates strictly on Python `decimal.Decimal` with arbitrary-precision arithmetic. Completely eliminates IEEE 754 floating-point inaccuracies.
- **RHS Extraction**: Recursively extracts right-hand values from equations (e.g., `5 + 7 = 12` $\implies$ `12`).
- **Currency & Unit Stripping**: Automatically strips prefix currency symbols (`$`, `€`, `£`, `¥`) and suffix percentage marks (`%`).
- **Safe Fallback**: Rejects invalid comma-only strings (`","`) or empty matches, returning `None`. Correctly normalizes `-0` and `0.00` to `"0"`.

### B. Sandboxed Code Runner
- **Subprocess Isolation**: Candidate Python code blocks generated for MBPP problems are executed in a child subprocess.
- **Security Interceptors**: 
  - Network modules (`socket`, `urllib`, `requests`, `http`) are hard-disabled (`None`).
  - File write operations (`'w'`, `'a'`, `'+'`, `'x'`) raise explicit `PermissionError` defense guards.
- **Execution Limits**: Strict execution timeouts (2.0–3.0 seconds) and memory limits ensure infinite loops and memory exhaustion do not hang evaluation.

### C. Held-Out Evaluation Suites
1. **Official GSM8K Test**: Evaluates mathematical reasoning accuracy on unseen test problems using greedy generation and Decimal verification.
2. **Official MBPP Test (pass@1)**: Evaluates code generation and algorithmic logic against unit test assertions in the sandboxed runner.
3. **Official SQuAD v2 Validation**: Measures Exact Match (EM) and Stanford token-level F1 on comprehension questions.
4. **Held-Out NLL & Bits-Per-Byte (BPB)**: Evaluates general language modeling predictive entropy across held-out splits of UltraChat, SQuAD, and GSM8K.
5. **Quality & Repetition Diagnostics**:
   - Stop-token compliance rate: percentage of generations correctly terminated by `<|im_end|>`.
   - 4-gram repetition rate: detection of degenerate repetition loops.
   - Synthetic probes: Copy probe and 2-digit addition sanity probes.

### D. Statistical Confidence Intervals
- Implements empirical non-parametric bootstrap resampling ($N=1000$ iterations) to report true 95% Confidence Intervals $[low, high]$ for all benchmark accuracy and pass rates.

---

## 3. Kaggle & Google Colab GPU Execution

The harness automatically detects and leverages CUDA GPUs:

```bash
# Auto-detect GPU (defaults to CUDA if available):
python aris/evaluation/eval_harness_v2.py --device auto

# Evaluate specific checkpoint:
python aris/evaluation/eval_harness_v2.py --checkpoint models/aris_lm_checkpoint.pt --device cuda --max-samples 100 --output scratch/eval_scorecard.json
```

---

## 4. Empirical Legacy Baseline Scorecard (Clean Held-Out Splits)
- **Execution Environment**: Kaggle Cloud GPU (NVIDIA Tesla T4, CUDA 13.0)
- **Commit**: `0cf8a88`
- **Source Log**: `scratch/eval_legacy_clean.log`

| Checkpoint | Held-Out NLL | BPB | GSM8K Acc (95% CI) | MBPP pass@1 | SQuAD v2 F1 | Stop% | Copy Probe% |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Uniform LM Baseline** | 8.3178 nats | — | 0.0% | 0.0% | 0.0% | — | — |
| **Step 21,250 (Legacy)** | **1.4734 nats** | **1.02** | **0.0% [0.0 - 0.0]** | **68.0%** | **8.1%** | **26.0%** | **20.0%** |

### Key Observations:
1. **Predictive Entropy**: Held-out NLL of 1.4734 nats (1.02 BPB) vs 8.3178 nats confirms significant language modeling compression on unseen held-out splits.
2. **Reasoning Deficit**: GSM8K held-out accuracy is 0.0% with the strict Decimal parser, confirming that earlier legacy reported numbers were train-set memorization artifacts.
3. **Algorithmic Capability**: MBPP pass@1 is 68.0% on held-out Python coding problems, demonstrating preserved code execution affordance.
4. **Structural Flaws**: Low stop compliance (26.0%) and low copy probe (20.0%) validate the need for Tokenizer V2, single-digit tokenization, and Data Pipeline V2 zero-slice truncation.
