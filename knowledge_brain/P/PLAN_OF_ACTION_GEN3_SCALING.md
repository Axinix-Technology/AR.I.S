# Plan of Action: Clean Verification & Gen 3 (2×) Scaling Architecture

## 1. Context & Executive Summary
This document records the strategic transition roadmap for AR.I.S from Phase 2b/3 (Small Rung 26.75M Pilot validation) to Phase 3 (Gen 3 $2\times$ Parameter Doubling to $53.5\text{M}$ with advanced cognitive curricula).

Detailed operational specifications are maintained in [.agents/plan_of_action.md](file:///d:/Axinix%20Projects/AR.I.S/.agents/plan_of_action.md).

---

## 2. Core Gates

### Gate 1: Phase 2b Small Pilot Completion (26.75M)
- Model: $d_{model}=512, L=8$, GQA 8/2, SwiGLU 1408, RoPE, RMSNorm.
- Execution: Kaggle Tesla T4 GPU, AMP torch.bfloat16.
- Invariants: Zero-slice truncation, response-only loss masking, bit-identical resumption.

### Gate 2: Clean Held-Out Evaluation
- Benchmark suite: `aris/evaluation/eval_harness_v2.py`.
- Metrics: Stop-token compliance ($\ge 75\%$), Copy probe ($\ge 50\%$), MBPP pass@1, Decimal GSM8K test accuracy.

### Gate 3: Core Runtime Graduation
- Swap system defaults in `aris/config.py` to point to `models/aris_lm_clean_small_pilot.pt` and `models/aris_bpe_tokenizer_v2.json`.

---

## 3. Gen 3 (53.5M) $2\times$ Parameter Scaling Specification
- **Hidden Dim ($d_{model}$)**: 640
- **Layers ($L$)**: 12 (50% deeper multi-hop relational capacity)
- **Attention**: GQA with 10 query heads, 2 KV heads (head dim 64)
- **SwiGLU ($d_{ff}$)**: 1,792
- **Parameters**: 53,494,784 (~53.5M)
- **5 Cognitive Curricula**:
  1. Deep Chain-of-Thought Reasoning (`<|thought|>` via Open-Thoughts)
  2. Competition Math & Multi-Step Logic (NuminaMath / Hendrycks MATH)
  3. Advanced Software Engineering & Debugging (Python-Edu / Evol-Code)
  4. High-Density Synthetic Textbooks (FineWeb-Edu Score 4-5 / Cosmopedia v2)
  5. Agentic Tool Calling & Function Schemas (Hermes Function Calling)
