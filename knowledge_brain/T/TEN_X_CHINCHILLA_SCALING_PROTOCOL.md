# Ten-X (10×) Chinchilla Scaling Protocol & High-Density QA Ingestion

## 1. Executive Summary & Purpose
Following the empirical validation of the **5% Pilot Run** (which demonstrated held-out NLL drop from 9.01 to 1.76 nats, mathematical deduction emergence on GSM8K at 1.0%, and a doubling of stop compliance to 56%), AR.I.S transitions to the **10× Chinchilla Scaling Budget**. 

Instead of jumping directly from 5% (1× parameters) to 20× parameters, the 10× intermediate scaling phase tests empirical loss decay curves, associative semantic memory depth, and reasoning emergence under controlled compute expenditure.

---

## 2. Quantitative Step & Compute Specifications

| Dimension | Specification |
| :--- | :--- |
| **Model Rung** | SMALL Gen 2 (8 layers, $d_{\text{model}}=512$, GQA 8/2, SwiGLU 1408, RoPE, RMSNorm) |
| **Parameter Count ($N$)** | **26,747,392** parameters |
| **Chinchilla Target Multiplier** | **10.0×** parameter tokens |
| **Target Token Volume** | $26,747,392 \times 10 = \mathbf{267,473,920\text{ tokens}}$ |
| **Context Window ($L$)** | 384 tokens |
| **Micro-Batch Size ($B$)** | 16 sequences |
| **Gradient Accumulation ($A$)** | 2 micro-steps |
| **Tokens per Optimizer Step** | $16 \times 384 \times 2 = \mathbf{12,288\text{ tokens/step}}$ |
| **Total Optimizer Steps** | $\lceil 267,473,920 / 12,288 \rceil = \mathbf{21,768\text{ steps}}$ |
| **Warmup Steps** | 500 steps (linear warmup to $6.0 \times 10^{-4}$) |
| **Learning Rate Schedule** | Cosine decay to $6.0 \times 10^{-5}$ at step 21,768 |
| **Target Hardware** | Kaggle Cloud Tesla T4 GPU (14.56 GB VRAM) |
| **Throughput Benchmark** | $\approx 15,400\text{ tokens/second}$ |
| **Estimated Wall-Clock Time** | $\approx 267,473,920 / 15,400 = 17,368\text{ seconds} \approx \mathbf{4.83\text{ hours}}$ |
| **Kaggle Session Ceiling** | 12 hours max (fits comfortably in a single run) |

---

## 3. High-Density QA Corpora Expansion

To ensure the model learns complex reasoning rather than shallow conversational patterns, two high-density QA corpora have been added into `aris/data/manifest.py` and streamed via `ArisDataPipelineV2`:

1. **SmolTalk OpenHermes 100k** (`data/smoltalk/openhermes_100k.parquet`):
   - **95,000 diverse question-answer pairs** spanning programming, algorithmic logic, science, humanities, and synthetic reasoning.
   - Uses strict ChatML `<|im_start|>` and `<|im_end|>` turn framing.
2. **SmolTalk NuminaMath CoT 100k** (`data/smoltalk/numina_cot_100k.parquet`):
   - **106,147 step-by-step mathematical reasoning chains** using Chain-of-Thought (CoT) format.
   - Eliminates direct-answer memorization and enforces cognitive derivation steps.

Combined with SQuAD 2.0 (86k), Databricks Dolly-15k (15k), SmolTalk Magpie Ultra (20k), HotpotQA (90k), and UltraChat (200k), AR.I.S has access to over **500,000 high-density QA turns**.

---

## 4. Inference Repetition Penalty Substrate

During greedy autoregressive generation, early-stage models can enter token attractor loops due to attending disproportionately to recent tokens.
`ArisInferenceEngine.generate` and `ArisEvalHarnessV2.generate_greedy` now support multiplicative repetition penalty:

$$\text{logits}[t] = \begin{cases} \text{logits}[t] / \rho & \text{if } \text{logits}[t] > 0 \\ \text{logits}[t] \times \rho & \text{if } \text{logits}[t] \le 0 \end{cases}$$

where $\rho = 1.15$, penalizing previously generated tokens without altering zero-point invariants.

---

## 5. Official Execution Instructions (Kaggle / Colab)

### Cell 1: Environment & Repository Setup
```bash
!git clone https://github.com/Axinix-Technology/AR.I.S.git
%cd AR.I.S
!git checkout main
!git pull
!pip install -q datasets huggingface_hub pyarrow
```

### Cell 2: Download Foundation & High-Density QA Corpora
```bash
!python scratch/download_expanded_datasets.py
```

### Cell 3: Launch 10× Chinchilla Clean Training (~4.8 Hours)
```bash
!python aris/training/train_clean.py \
  --rung small \
  --budget-mult 10.0 \
  --device cuda \
  --save-every 500 \
  --log-every 50 \
  --checkpoint aris_lm_clean_small_10x.pt
```

### Cell 4: Evaluate Checkpoint on Held-Out Benchmark Scorecard
```bash
!python aris/evaluation/eval_harness_v2.py \
  --checkpoint models/aris_lm_clean_small_10x.pt \
  --tokenizer models/aris_bpe_tokenizer_v2.json \
  --device cuda \
  --output scratch/eval_small_10x.json
```
