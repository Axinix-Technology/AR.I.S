# AR.I.S Development Backlog

This document tracks non-blocking tasks, future optimizations, and technical debt items identified during development. Each item is addressed when modifying relevant modules.

## Backlog Items
1. [COMPLETED in Step 1] **Linguistic UD parser**: Manifest-driven filtering strictly includes `*-train.conllu` in `aris/data/manifest.py`.
2. [COMPLETED in Step 2] **Tokenizer digits**: Single-digit pre-tokenization splitting implemented via `Sequence([Digits(individual_digits=True), ByteLevel(add_prefix_space=False)])`.
3. [COMPLETED in Step 1] **DataLoader synchronization**: Bit-identical stream cursor and RNG state resumption implemented in `aris/data/pipeline_v2.py`.
4. [COMPLETED in Step 3] **Sandboxed evaluation**: Tempfile isolation and standard interceptors verified in `aris/evaluation/eval_harness_v2.py`.
5. **Container Sandbox**: Migrate to WSL/Docker rootless container for untrusted code execution on Linux/Cloud hosts.
6. **Distributed Multi-GPU Training (DDP)**: Integrate `torch.distributed` launcher for 2x T4 GPU acceleration on Kaggle.
