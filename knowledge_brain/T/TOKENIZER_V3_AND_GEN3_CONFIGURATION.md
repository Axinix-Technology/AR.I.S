# Tokenizer V3 & Gen 3 (1,024-Token Context) Architecture

## Overview
This document records the architectural specification and empirical validation of **AR.I.S Tokenizer V3** and the **Gen 3 Model Configuration** scaled to a **1,024-token active context window** and **16,384 vocabulary size**.

---

## 1. Tokenizer V3 Specifications (`aris_bpe_tokenizer_v3.json`)

- **Vocabulary Size**: **16,384** ($2^{14}$ exact power of 2 for maximum CUDA Tensor Core efficiency).
- **Pre-Tokenizer Pipeline**: `Sequence([Digits(individual_digits=True), ByteLevel(add_prefix_space=False)])`.
  - Splits all digits (0–9) individually to prevent arbitrary number merging and preserve arithmetic precision.
- **Decoder**: `ByteLevelDecoder()` ensuring lossless whitespace, indent, and binary round-tripping.
- **Native Tool-Calling & Deliberation Special Tokens**:
  - `0`: `<|pad|>`
  - `1`: `<|unk|>`
  - `2`: `<|bos|>`
  - `3`: `<|eos|>`
  - `4`: `<|im_start|>`
  - `5`: `<|im_end|>`
  - `6`: `<|thought|>`
  - `7`: `<|thought_end|>`
  - `8`: `<|tool_call|>`
  - `9`: `<|tool_response|>`
  - `10`: `<|tool_result|>`
  - `11`: `<|tool_def|>`
  - `12`: `<|plan|>`
  - `13`: `<|endoftext|>`
  - `14`: `<|fim_prefix|>`
  - `15`: `<|fim_middle|>`
  - `16`: `<|fim_suffix|>`
  - `17–64`: 48 Reserved future control slots.
- **Artifact SHA256**: `6f8ec982d060b553c0f7ddec4241a7d5b9464a0b831e02714b30e1a74fd09685`.

---

## 2. Gen 3 Model Architecture Specifications

- **Model Rung**: `gen3` / `medium`
- **Total Parameters**: **63,585,920** (~63.6M Parameters)
- **Active Context Window**: **1,024 tokens** (2.67× expansion over Gen 2's 384 tokens)
- **Hidden Dimension ($d_{model}$)**: 640
- **Number of Layers ($L$)**: 12 Decoder Blocks
- **Attention Type**: Grouped-Query Attention (GQA 5:1)
  - **Query Heads**: 10 (Head dimension = 64)
  - **KV Heads**: 2 (Head dimension = 64)
- **SwiGLU Intermediate Dimension ($d_{ff}$)**: 1,792
- **Positional Encoding**: Rotary Position Embeddings (RoPE) with `rope_theta = 25000.0`.
- **Weight Tying**: `tie_word_embeddings = True` (`tok_emb.weight` shared with `lm_head.weight`).

---

## 3. Empirical Domain Compression & Sequence Length Audit

Audited across **7,874 real manifest samples** spanning Code (MBPP), Math (GSM8K), Prose (SQuAD), and Multi-turn Chat (SmolTalk & Magpie):

### A. Domain Compression Benchmark (V2 8k vs V3 16k)
| Domain | Samples | Raw Bytes | V2 Tokens (8k) | V3 Tokens (16k) | V2 Bytes/Tok | V3 Bytes/Tok | Compression Gain |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Multi-turn Chat** | 2,500 | 11,598,277 | 3,069,012 | 2,627,792 | 3.78 | **4.41** | **+14.38%** |
| **Code (MBPP)** | 374 | 171,474 | 84,585 | 74,872 | 2.03 | **2.29** | **+11.48%** |
| **Prose / QA (SQuAD)** | 2,500 | 2,182,250 | 639,690 | 570,757 | 3.41 | **3.82** | **+10.78%** |
| **Math (GSM8K)** | 2,500 | 1,341,013 | 516,704 | 487,154 | 2.60 | **2.75** | **+5.72%** |

*Note on Math*: Math compression gain is +5.72% (vs +11% to +14% for code/chat) because `Digits(individual_digits=True)` deliberately preserves single-digit atomicity, trading slight token expansion for exact numeric calculation fidelity.

### B. Sequence Length Percentiles & Context Retention (7,874 Samples)
| Metric | Tokenizer V2 (8k) | Tokenizer V3 (16k) | Empirical Shift |
| :--- | :--- | :--- | :--- |
| **Median (P50)** | 240.0 tokens | **219.0 tokens** | $-21.0$ tokens |
| **90th Percentile (P90)** | 1,758.4 tokens | **1,493.0 tokens** | $-265.4$ tokens |
| **95th Percentile (P95)** | 2,152.3 tokens | **1,835.3 tokens** | $-317.0$ tokens |
| **99th Percentile (P99)** | 3,285.9 tokens | **2,741.3 tokens** | $-544.6$ tokens |
| **Retention in 384 Window** | 76.53% | **78.49%** | Explains the 23.5% drops in Gen 2 |
| **Retention in 1,024 Window**| — | **83.48%** | **83.48% fit with zero context shrinking** |
| **Residual Truncation (>1024)**| — | **16.52%** | Long 8-12 turn Magpie dialogues |
