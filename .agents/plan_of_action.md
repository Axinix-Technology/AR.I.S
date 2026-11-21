# AR.I.S — Strategic Plan of Action: Clean Verification & Gen 3 (2×) Scaling

## 1. Executive Summary & Current Execution State
- **Current Milestone**: Phase 2b/3 Small Rung Pilot Pretraining (~26.75M parameters, 8 layers, $d_{model}=512$, GQA 8/2, SwiGLU 1408).
- **Runtime Environment**: Kaggle Cloud GPU (NVIDIA Tesla T4, CUDA 13.0, AMP torch.bfloat16).
- **Empirical Progress**:
  - Training Mode: 5% Pilot Run ($26,747,392$ tokens / $2,177$ steps).
  - Step 1,340 / 2,177 reached (**61.6% completed**, 16.46M tokens trained).
  - Loss Trajectory: Steep monotonic descent from **7.52** $\to$ **3.75** $\to$ **2.12** $\to$ **2.37**.
  - Throughput: Steady **$\sim 11,200\text{ tok/s}$**; healthy gradient norm ($\le 1.05$); periodic checkpoints at steps 200, 400, 600, 800, 1000, 1200.

---

## 2. Immediate Milestone: Pilot Verification & Runtime Graduation

### Stage 2.1: Pilot Completion & Checkpoint Freeze
- **Target Step**: Step 2,177 ($100\%$ of pilot budget).
- **Target Output**: `models/aris_lm_clean_small_pilot.pt`.

### Stage 2.2: Clean Evaluation Benchmark Execution
Run the unpolluted developmental evaluation suite on official held-out splits:
```bash
python aris/evaluation/eval_harness_v2.py \
  --checkpoint models/aris_lm_clean_small_pilot.pt \
  --tokenizer models/aris_bpe_tokenizer_v2.json \
  --device cuda \
  --max-samples 100 \
  --output scratch/eval_pilot_clean.json
```
- **Acceptance Criteria**:
  1. Stop-token compliance rate $\ge 75\%$ (confirming elimination of the legacy 26% babbling flaw).
  2. Copy probe accuracy $\ge 50\%$ (confirming single-digit byte-level preservation).
  3. Stable held-out NLL on unseen SQuAD, GSM8K, and UltraChat validation splits.

### Stage 2.3: Runtime Graduation to Core AR.I.S Brain
Upon passing evaluation gates, graduate the clean model as the default brain in `aris/config.py`:
- `TOKENIZER_PATH = MODELS_DIR / "aris_bpe_tokenizer_v2.json"`
- `LM_CHECKPOINT_PATH = MODELS_DIR / "aris_lm_clean_small_pilot.pt"`
- Activates clean model across `run.py chat`, `AutonomousDeliberationEngine`, and `HybridGraphRetriever`.

---

## 3. The Next Level: Gen 3 (2× Parameter Doubling Staircase)

### Architecture Specification: Gen 3 Medium-Lite (~53.5M Parameters)
To scale beyond basic syntax into true multi-step relational cognition, double parameter capacity:

$$\mathbf{26.75M} \text{ (Gen 2 Small)} \longrightarrow \mathbf{53.5M} \text{ (Gen 3)}$$

- **Hidden Dimension ($d_{model}$)**: $640$ (up from 512)
- **Layer Depth ($L$)**: $12$ layers (up from 8, providing 50% more depth for 12-hop deductions)
- **Attention Configuration**: GQA with 10 Query Heads, 2 KV Heads (Head Dim: 64)
- **SwiGLU Intermediate Dimension ($d_{ff}$)**: $1,792$
- **Vocabulary**: 8,192 (Tokenizer V2 ByteLevel BPE with individual single-digit splitting)
- **Tied Word Embeddings**: $8,192 \times 640 = 5,242,880$ params
- **Hardware Footprint**: ~1.2 GB VRAM in bfloat16 (~8,500 tok/s on Kaggle Tesla T4).

---

## 4. Upgraded Cognitive Data Curriculum for Gen 3 (53.5M)

A $53.5\text{M}$ brain requires high-density cognitive training data to fully utilize its representational rank:

```
                                  53.5M AR.I.S BRAIN
                                         │
        ┌───────────────┬────────────────┼───────────────┬───────────────┐
        ▼               ▼                ▼               ▼               ▼
1. Deep Chain-of-   2. Competition   3. Software     4. Synthetic    5. Agentic Tool
   Thought (CoT)       Math & Logic     Engineering     Textbooks       Calling & SQL
```

### Pillar 1: Deep Chain-of-Thought (CoT) & Autonomous Deliberation
- **Corpora**: `Open-Thoughts` / `Magpie-Reasoning`
- **Objective**: Trains AR.I.S to produce structured `<|thought|>` deduction blocks before answering, directly empowering the `AutonomousDeliberationEngine`.

### Pillar 2: Advanced Mathematical Deduction & Theorem Proving
- **Corpora**: `NuminaMath-CoT` / `Hendrycks MATH`
- **Objective**: Moves beyond grade-school arithmetic into multi-step algebra, geometry, calculus, and constraint verification.

### Pillar 3: Professional Software Engineering & Algorithmic Logic
- **Corpora**: `HuggingFaceTB/smollm-corpus` (Python-Edu) / `Evol-Instruct-Code`
- **Objective**: Teaches object-oriented architecture, edge-case unit testing, and automated algorithmic debugging.

### Pillar 4: High-Density Science & Synthetic Textbooks
- **Corpora**: `HuggingFaceFW/fineweb-edu` (Score 4–5) / `Cosmopedia v2`
- **Objective**: Ingests college-level concepts across physics, biology, computer science, and systems engineering.

### Pillar 5: Agentic Tool Use & Function Schema Calling
- **Corpora**: `Hermes-Function-Calling` / `ToolBench`
- **Objective**: Teaches structured JSON tool calling, relational SQL queries, and AR.I.S Graph RAG memory traversal.

---

## 5. Execution Roadmap & Timeline

| Phase | Milestone | Compute & Data | Success Gate |
| :--- | :--- | :--- | :--- |
| **Phase 2b (Active)** | 26.75M Small Pilot Run | Kaggle T4 (~15 mins remaining) | Step 2,177 completed, Loss $\le 2.3$ |
| **Phase 2b Gate** | Clean Evaluation Scorecard | Held-Out Test Splits | Stop% $\ge 75\%$, Copy% $\ge 50\%$ |
| **Phase 3 Transition** | AR.I.S Runtime Switch | `aris/config.py` update | Clean model active in `run.py chat` |
| **Phase 3 Scale-Up** | Gen 3 ($53.5\text{M}$) Model Setup | `ArisConfig(embed_dim=640, layers=12)` | Clean architecture verified |
| **Phase 3 Curriculum** | 5-Pillar Data Ingestion | Manifest-driven high-density sets | Zero-slice truncation verified |
| **Phase 3 Training** | Gen 3 5% Pilot & Full Run | Kaggle T4 / Multi-GPU DDP | Monotonic loss convergence |
