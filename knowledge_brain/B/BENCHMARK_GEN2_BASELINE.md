# Gen 2 Milestone Benchmark Baseline

## Overview
This document records the official, frozen baseline evaluation metrics for AR.I.S **Gen 2 Small** (26.75M parameters) evaluated at **117M tokens (Step 9,500)** across the clean held-out evaluation suite (`eval_harness_v2.py`). 

Every future scaling milestone—starting with **Gen 3 (63.6M parameters, 1,024 context window, Tokenizer V3)**—must be measured 1:1 on these exact benchmark tasks to isolate true capability progression.

---

## 1. Frozen Scorecard

- **Model Rung**: `small` (26,747,392 Parameters, $d=512, L=8$)
- **Active Tokens Evaluated**: ~117M tokens (Step 9,500)
- **Decoding**: Greedy Argmax ($T = 0, \text{repetition\_penalty} = 1.0$)
- **Source Artifact**: `models/eval_results_gen2_baseline.json`

| Metric / Task | Gen 2 Baseline Score | 95% Bootstrap CI | Evaluation Details |
| :--- | :--- | :--- | :--- |
| **Held-Out NLL** | **$2.1516\text{ nats}$** | — | Clean held-out splits (SQuAD v2, GSM8K, UltraChat) |
| **Held-Out BPB** | **$1.4758\text{ BPB}$** | — | Normalized Bits-Per-Byte across unseen text |
| **GSM8K Test Accuracy** | **$3.0\%$** | $[0.0\%, 7.0\%]$ | 100 test samples, strict Decimal parser |
| **MBPP Coding Pass@1** | **$0.0\%$** | $[0.0\%, 0.0\%]$ | 50 test samples, sandboxed subprocess unit test runner |
| **SQuAD v2 F1 Score** | **$10.57\%$** | — | 100 test samples, extractive comprehension |
| **SQuAD v2 Exact Match**| **$7.0\%$** | — | Exact string match on answer spans |
| **4-Gram Repetition Rate**| **$96.0\%$** | — | Percentage of generations containing looped 4-grams |
| **Stop Compliance** | **$35.0\%$** | — | Termination on `<|im_end|>` under pure greedy argmax |
| **Synthetic Copy Probe** | **$10.0\%$** | — | Short token sequence copy accuracy |

---

## 2. Gen 3 Target Criteria for Progression

To validate that Gen 3 represents true cognitive maturation over Gen 2, Gen 3 must meet or exceed these explicit criteria on the identical evaluation suite:

1. **Held-Out NLL**: $\le \mathbf{1.75\text{ nats}}$ ($\ge 18\%$ reduction in validation cross-entropy).
2. **GSM8K Accuracy**: $\ge \mathbf{10.0\%}$ (benefiting from continuous math streaming and digit-split Tokenizer V3).
3. **MBPP Coding Pass@1**: $\ge \mathbf{5.0\%}$ (benefiting from uncapped continuous Python code training).
4. **SQuAD v2 F1**: $\ge \mathbf{25.0\%}$ (improved contextual reading across the 1,024 window).
5. **Stop Compliance**: $\ge \mathbf{85.0\%}$ (breaking greedy attractor loops via higher model capacity and repetition penalty).
6. **Tool-Call Schema Accuracy**: $\ge \mathbf{80.0\%}$ on emitting syntactically valid JSON tool arguments within `<|tool_call|>`.
