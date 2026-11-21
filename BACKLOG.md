# AR.I.S Development Backlog

This document tracks non-blocking tasks, future optimizations, and technical debt items identified during development. Each item is addressed when modifying relevant modules.

## Backlog Items
1. **Linguistic UD parser**: Update `_iter_ud()` in `aris/data/dataset.py` to filter explicitly on `-train.conllu` to prevent dev/test split leakage.
2. **Tokenizer regex**: Evaluate pre-tokenization regex split pattern for unicode punctuation and single-digit representation.
3. **DataLoader multi-worker synchronization**: Add persistent epoch and cursor tracking to support cross-session checkpoint resumption.
4. **Sandboxed evaluation**: Migrate to WSL/Docker rootless container for adversarial code execution beyond trusted model output.
