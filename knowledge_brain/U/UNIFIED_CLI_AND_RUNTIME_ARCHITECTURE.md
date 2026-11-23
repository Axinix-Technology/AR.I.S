# Unified CLI & Cognitive Runtime Architecture (`run.py`)

## Overview
AR.I.S operates through a single consolidated entrypoint: `run.py`. All legacy, intermediate, and experimental training scripts (including `train_clean.py`) have been unified into `aris/training/trainer.py` and exposed directly via standard CLI subcommands.

---

## 1. Core Runtime Commands

### A. Training Modern Language Models (`train-lm`)
Executes the proven manifest-driven data pipeline (`ArisDataPipelineV2`) with zero-slice truncation, response-only loss masking, automatic mixed precision (AMP bfloat16), cosine learning rate scheduling, and milestone checkpoint archiving.

```bash
# Train Small (26.75M) on 10x Chinchilla budget
python run.py train-lm --rung small --budget-mult 10.0 --device cuda

# Train Gen 3 (58.34M, 2x Scaling)
python run.py train-lm --rung gen3 --budget-mult 10.0 --device cuda

# Quick 1.0x Pilot Run
python run.py train-lm --rung small --pilot
```

**Key Parameters**:
- `--rung`: `small` (26.75M params, $d=512, L=8$), `medium` / `gen3` (58.34M params, $d=640, L=12$), `pico` (7.63M params, $d=256, L=4$).
- `--budget-mult`: Chinchilla scaling multiple (default: `10.0`).
- `--resume-from`: Path to explicit checkpoint to continue training.
- `--save-every`: Checkpoint save frequency (milestones archived every 2,500 steps).

---

### B. Interactive Cognitive Dialogue (`chat`)
Launches the conversational runtime using `ArisLM` with KV Caching and proven repetition penalty ($1.15$ default) to prevent greedy attractor loops.

```bash
python run.py chat --checkpoint models/aris_lm_clean_small_15x.pt --repetition-penalty 1.15
```

---

### C. Standardized Benchmark Evaluation (`eval`)
Executes clean held-out evaluation across official benchmarks (GSM8K Test with Decimal parser, MBPP Test with sandboxed runner, SQuAD v2 Dev, and Stop Compliance).

```bash
python run.py eval --checkpoint models/aris_lm_clean_small_15x.pt --samples 100 --device cuda
```

---

## 2. Standardized Artifact References

In `aris/config.py`, all default paths point exclusively to the verified, frozen production artifacts:
- **Default Tokenizer**: `models/aris_bpe_tokenizer_v2.json` (Vocab size: 8,192).
- **Default Checkpoint**: `models/aris_lm_clean_small_15x.pt`.
- **Default Context Window**: 384 tokens.
- **Precision**: AMP `torch.bfloat16` on CUDA (fallback to float32 on CPU).
