# Data Audit & Empirical Evaluation Foundation

This record establishes the verified empirical foundations for dataset integrity, token distributions, strict evaluation parsers, execution sandboxing, and scaling ladder parameters for AR.I.S.

---

## 1. Non-Negotiable Process Directives
1. **Zero Unverified Numbers**: Every quantitative claim must be directly output by a committed verification script in `scratch/` with an accompanying empirical log file.
2. **Permanent Legacy Quarantine**: All legacy scorecard numbers (Steps 16,000 and 21,250) are archived and quarantined as `LEGACY / TRAIN-SET / IRREPRODUCIBLE` due to prompt leakage, training split evaluation, and uncalibrated 50-token substring heuristic matching.

---

## 2. SmolTalk Structure & Discrepancy Resolution
- **File Distribution**:
  - `everyday_conversations.parquet`: 2,260 rows $\implies$ **8,630 (user, assistant) turn pairs**. Post-truncation mean ($\le 384$ tokens): **62.21**, supervised mean: **42.15**, zero-sup: **0.00%**.
  - `smol_magpie_ultra.parquet`: 68,257 rows $\implies$ **204,771 (user, assistant) turn pairs**. Post-truncation mean ($\le 384$ tokens): **369.30**, supervised mean: **287.09**, zero-sup: **0.14%**.
- **Dataset Totals**:
  - Total Raw Rows: **70,517**
  - Total Yieldable Turn Pairs: **213,401**
  - Combined Weighted Mean Non-Pad Tokens ($\le 384$): **356.88**
  - Combined Weighted Mean Supervised Tokens: **277.19**
- **Discrepancy Root Cause**:
  `dataset.py` iterated files alphabetically. `everyday_conversations.parquet` sorted first. Because it yielded 8,630 turn pairs, a naive head-sample of 2,000 drew 100% of its data from `everyday_conversations`, misrepresenting the full corpus where `smol_magpie_ultra` accounts for 96.0% (204,771 / 213,401) of all conversational pairs.

---

## 3. UltraChat Split Deduplication & Hygiene
- **Assistant Response Audit**: All 9 shards (`*_gen` and `*_sft`) contain full assistant turns (verified across 20/20 sampled rows per file).
- **Excluded Splits (74,524 Rows Total)**:
  - `test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet` (23,110 rows): Official held-out test split.
  - `test_gen-00000-of-00001-3d4cd8309148a71f.parquet` (28,304 rows): Official held-out test split.
  - `train_sft.parquet` (23,110 rows): Byte-for-byte duplicate of `test_sft` (SHA256: `c18fd6e77395577652bdefbc5a87044ea799e989451ecfca5c6cf977ae5c6f70`).
- **Clean Training Splits (6 Files, 463,897 Rows Total)**:
  - `train_gen-00000..00002`: 256,032 rows.
  - `train_sft-00000..00002`: 207,865 rows.
  - Intra-shard prompt deduplication sample (24,000 rows): 0.00% duplicates.

---

## 4. Strict Decimal Evaluation & Sandboxed Execution
- **Strict Decimal Parser**:
  - Replaces floating-point math with Python `decimal.Decimal` to eliminate precision loss on integers $> 2^{53}$.
  - Handles negative zero (`-0`, `-0.0` $\to$ `0`), currency/percent symbols, scientific notation (`1e3` $\to$ `1000`), thousands commas, and extracts the final match from internal scratchpads.
- **Windows Sandboxed Execution**:
  - Enforces execution timeouts via subprocess termination.
  - Limits memory ceiling via Job Object limits.
  - Prohibits fork-bomb child creation via `ActiveProcessLimit = 1`.
  - Blocks network and unauthorized filesystem writes.

---

## 5. Scaling Ladder Parameters (Context = 2,048)
Calculated via `cfg.expected_param_count()`:

| Rung | Vocab | Tied Unique Params | Untied Params | Embedding Share | $d_{model}$ | Layers |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pico (~5M)** | 8,192 | 6,524,160 | 8,621,312 | 32.14% | 256 | 6 |
| **Pico (~5M)** | 16,384 | 8,621,312 | 12,815,616 | 48.65% | 256 | 6 |
| **Pico (~5M)** | 32,768 | 12,815,616 | 21,204,224 | 65.46% | 256 | 6 |
| **Small (~25M)** | 8,192 | 26,747,392 | 30,941,696 | 15.68% | 512 | 8 |
| **Small (~25M)** | 16,384 | 30,941,696 | 39,330,304 | 27.11% | 512 | 8 |
| **Small (~25M)** | 32,768 | 39,330,304 | 56,107,520 | 42.66% | 512 | 8 |
| **Medium (~100M)**| 8,192 | 106,980,096 | 113,271,552 | 5.88% | 768 | 16 |
| **Medium (~100M)**| 16,384 | 113,271,552 | 125,854,464 | 11.11% | 768 | 16 |
| **Medium (~100M)**| 32,768 | 125,854,464 | 151,020,288 | 20.00% | 768 | 16 |
| **Base (~350M)** | 8,192 | 278,971,392 | 287,360,000 | 3.01% | 1024 | 24 |
| **Base (~350M)** | 16,384 | 287,360,000 | 304,137,216 | 5.84% | 1024 | 24 |
| **Base (~350M)** | 32,768 | 304,137,216 | 337,691,648 | 11.03% | 1024 | 24 |
