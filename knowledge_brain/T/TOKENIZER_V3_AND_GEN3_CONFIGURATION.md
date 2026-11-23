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

## 3. Benefits of Combined 16k Vocab + 1,024 Context

1. **Elimination of Sequence Truncation**:
   - In Gen 2 (384 context), 160,988 long reasoning and coding samples were dropped.
   - At 1,024 tokens with 16k compression ($\sim 4.5$ bytes/token $\approx 4,600$ characters per sequence), over 99% of multi-hop math proofs, unit test suites, and multi-turn tool calling dialogues fit natively within the context.
2. **Native Tool Agency**:
   - With `<|tool_call|>` and `<|tool_response|>`, AR.I.S natively emits and receives tool execution payloads without character-level parsing workarounds.
