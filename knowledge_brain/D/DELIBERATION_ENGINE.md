# Autonomous Deliberation Engine & Neuro-Symbolic Cognitive Loop

## 1. Overview & Architectural Motivation
Traditional chatbots generate output through immediate, unverified autoregressive decoding, which often causes hallucinations and prevents reasoning across disparate facts. The AR.I.S **Autonomous Deliberation Engine** operates as an end-to-end cognitive control loop:

```
               USER PROMPT
                    │
                    ▼
       ┌─────────────────────────┐
Stage 1│ Hybrid Graph RAG Search │  <── (ArisGraphStore + Reciprocal Rank Fusion)
       └────────────┬────────────┘
                    │  [Pulls related entities, relations, & evidence]
                    ▼
       ┌─────────────────────────┐
Stage 2│ Multi-Hop Context Build │  <── (Compiles structured background narrative)
       └────────────┬────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
Stage 3│   Neural Deliberation   │  <── (ArisLM generates <|thought|> chain + answer)
       └────────────┬────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
Stage 4│   Epistemic Governor    │  <── (Validates confidence & factual integrity)
       └────────────┬────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
Stage 5│   Memory Consolidation  │  ──> (Stores verified entities back into Graph)
       └─────────────────────────┘
                    │
                    ▼
             FINAL RESPONSE
```

---

## 2. The 5-Stage Cognitive Deliberation Cycle

### Stage 1: Self-Directed Retrieval (Graph RAG)
Before generating tokens, AR.I.S inspects the incoming user prompt and queries its internal `HybridGraphRetriever`. It searches `ArisGraphStore` using a hybrid of dense embedding similarity and multi-hop entity traversal, applying Reciprocal Rank Fusion (RRF) to retrieve the most relevant known facts.

### Stage 2: Context Augmentation
Retrieved background facts are synthesized into a grounded narrative. If relevant entities exist in memory, the user turn is augmented with `Background Information:\n{memory_context}\nQuestion: {user_input}`.

### Stage 3: Neural Deliberation with Latent Thought Chains
The prompt is passed to `ArisLM` with ChatML framing (`<|im_start|>user ... <|im_end|>\n<|im_start|>assistant`). Using vectorized KV-caching and an anti-loop repetition penalty ($\rho = 1.15$), the model generates an internal deliberation chain:
```text
<|thought|>
[Deconstruction of question, relational step, formula]
Answer: [Direct concise answer to user]
<|im_end|>
```
`AutonomousDeliberationEngine` cleanly separates the internal `<|thought|>` chain from the user-facing response.

### Stage 4: Epistemic Authority Gating
Extracted entities and conclusions are evaluated by `EpistemicGovernor` (`aris/core/governance.py`). Candidate knowledge must satisfy confidence thresholds before being permitted to alter the system's long-term memory graph.

### Stage 5: Continuous Memory Consolidation
Salient entities and relations observed in dialogue are written back into `ArisGraphStore`:
- First-time entities are initialized with node embeddings and an access counter of 1.
- Corroborated entities increment their access counters and are promoted to `verified` status under governor approval.
- This gives AR.I.S continuous, lifelong cumulative memory without retraining weights.

---

## 3. Key Components in Codebase

| Component | File Path | Role |
| :--- | :--- | :--- |
| **`AutonomousDeliberationEngine`** | `aris/core/deliberation.py` | Cognitive controller orchestrating the 5-stage loop |
| **`HybridGraphRetriever`** | `aris/memory/retriever.py` | Dense vector + multi-hop graph path retrieval |
| **`ArisGraphStore`** | `aris/memory/graph_store.py` | In-memory & serialized entity-relation knowledge graph |
| **`EpistemicGovernor`** | `aris/core/governance.py` | 4-tier truth and memory write access gating |
| **`ArisInferenceEngine`** | `aris/inference/generation.py` | Vectorized KV-cache autoregressive token generation |

---

## 4. Execution via Interactive Chat

The Neuro-Symbolic Deliberation Engine is activated automatically in `run.py chat`:

```bash
python run.py chat \
  --checkpoint models/aris_lm_clean_small_15x.pt \
  --tokenizer models/aris_bpe_tokenizer_v2.json \
  --repetition-penalty 1.15 \
  --max-tokens 128
```

During the interactive session:
- If internal deliberation occurred, `[Thought]: ...` is displayed alongside the response.
- Background memories retrieved from past interactions are continuously factored into subsequent answers.
