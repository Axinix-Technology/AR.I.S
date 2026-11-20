# QA & High-Density Instruction Dataset Architecture

## 1. Executive Summary & Root Cause Analysis
During empirical testing of the 16k-step training checkpoint (`aris_lm_checkpoint.pt`), milestone evaluation demonstrated a high Lexical Grounding validity (**91.07%**) and Grammatical POS Awareness (**54.14%**), but a severe plateau in Question Answering and Reasoning (Stage 3 & 4 F1 at **0.00%**). 

The root cause of the "only proper words updating" symptom was diagnosed:
1. **Chatter Over-Saturation (UltraChat Bias)**: Training on massive continuous dialogue without factual QA pairs teaches the neural network English phonotactics and conversational phrasing, but fails to instill question $\to$ direct answer mapping.
2. **Missing QA in Cloud Downloads**: `download_expanded_datasets.py` used in cloud environments lacked SQuAD, HotpotQA, and QA-SRL.
3. **Thought-Token Formatting Disconnect**: Training formatted reasoning chains without unified `Answer:` tags, leading evaluation routines to truncate early before reaching the factual response.

---

## 2. Recommended Gold-Standard Datasets (Zero Cost / 100% Free)
Proprietary paid datasets are entirely unnecessary. State-of-the-art compact LLMs (such as Hugging Face SmolLM2 and AllenAI Tülu 3) achieve elite QA performance using freely accessible, permissively licensed open corpora:

| Dataset | Provider / Source | License | Size | Cognitive Purpose in AR.I.S |
| :--- | :--- | :--- | :--- | :--- |
| **SQuAD 2.0** | Stanford NLP (`rajpurkar/squad_v2`) | CC BY-SA 4.0 | ~130K pairs | Factual reading comprehension, exact answer extraction, non-hallucination. |
| **Databricks Dolly-15k** | Databricks (`databricks/databricks-dolly-15k`) | Apache 2.0 | 15K pairs | Human-authored instruction following, closed QA, open QA, and summarization. |
| **SmolTalk** | Hugging Face (`HuggingFaceTB/smoltalk`) | Apache 2.0 | Curated | Compact LLM specialist data: everyday dialogue, direct QA, concise identity grounding. |
| **HotpotQA** | Stanford / CMU (`hotpotqa/hotpot_qa`) | CC BY-SA 4.0 | 90K pairs | Multi-hop cross-paragraph deduction and evidence synthesis. |
| **GSM8K** | OpenAI (`openai/gsm8k`) | MIT | 8.5K pairs | Grade school multi-step arithmetic reasoning. |
| **MBPP** | Google Research (`google-research-datasets/mbpp`) | Apache 2.0 | 1K pairs | Python algorithmic problem solving and unit test assertions. |

---

## 3. Data Integration & Cognitive Pipeline

### 3.1 Unified ChatML Formatting
Every QA example is tokenized into standardized ChatML format with latent thought markers:

```
<|im_start|>user
Context: [Optional background context]
Question: [Direct inquiry]<|im_end|>
<|im_start|>assistant
<|thought|>
[Cognitive deduction and evidence extraction]
Answer: [Crisp, direct target answer]<|im_end|>
```

### 3.2 Dynamic Interleaving Strategy
The `EnterpriseArisDataset` interleaves corpora in a balanced 1:1:1:1 cognitive curriculum:
- **30% Direct Factual QA & Comprehension** (SQuAD 2.0 + Dolly-15k)
- **20% Concise Everyday Interaction & Identity** (SmolTalk / Custom Chat)
- **20% Multi-Hop & Causal Reasoning** (HotpotQA + QA-SRL)
- **15% Algorithmic Code Synthesis** (MBPP)
- **15% Step-by-Step Problem Solving** (GSM8K)
