# AR.I.S Semantic Understanding Training & Evaluation Protocol

## Overview
The **Semantic Understanding Training & Evaluation Protocol** is an objective benchmark suite designed to evaluate the internal cognitive pipeline of AR.I.S at its current developmental stage.

Crucially, this protocol enforces strict, non-negotiable operational principles:
1. **Zero External Dependencies**: Operates strictly offline. No external LLMs, no external APIs (e.g. OpenRouter, OpenAI), no GPU servers, and no external pre-trained embedding vectors.
2. **Cognitive Provenance Tracking**: Dissects the entire cognitive trajectory:
   $$\text{INPUT} \to \text{TOKENIZATION} \to \text{CANDIDATES} \to \text{SEMANTIC ENCODING} \to \text{RELATIONAL BINDING} \to \text{GRAPH PATH} \to \text{ANSWER}$$
3. **Representation vs. Answer Decoupling**: A correct final answer is **never** treated as proof of correct semantic representation. Each internal stage is audited independently.

---

## Benchmark Datasets

### 1. Training Knowledge Facts (`aris_training_facts.json`)
Consists of canonical ground-truth statements stored into AR.I.S brain memory:
- F0001: Arun works at Axinix.
- F0002: Axinix is located in Chennai.
- F0003: Chennai is located in Tamil Nadu.
- F0004: Tamil Nadu is located in India.
- F0005: India is located in Asia.
- F0006: Asia is located on Earth.
- F0007: Dhivya is a colleague of Arun.
- F0008: Dhivya works at Axinix.
- F0009: Ravi manages Arun.
- F0010: Ravi works at Axinix.
- F0011: Meena works at WorkHub.
- F0012: Axinix uses Python.
- F0013: Python supports Django.
- F0014: Moon orbits Earth.
- F0015: Earth orbits Sun.
- F0016: Sun is a star.

### 2. Question Training & Evaluation Dataset (`aris_question_training.json`)
Comprises 35 questions distributed across 10 distinct evaluation categories and three split partitions (`TRAIN`, `DEV`, `TEST`):
- **Category A**: Direct Relation (1-Hop factual retrieval)
- **Category B**: Paraphrase (lexical and structural reformulation)
- **Category C**: Spelling Noise (typos and character corruption)
- **Category D**: Grammar Noise (syntactic variations and ungrammatical prompts)
- **Category E**: Directionality (subject/object inversions and relation flips)
- **Category F**: Multi-Hop (2-hop, 3-hop, 4-hop, and 5-hop inductive reasoning chains)
- **Category G**: Unsupported (explicit ungrounded queries; tests hallucination resistance)
- **Category H**: Semantic Equivalence (alternative relational verbs, e.g. "employed by")
- **Category I**: Entity Variation (casing, prefixes, compound formulations)
- **Category J**: Correction / Feedback (denial of negative hypotheses, e.g. "Moon orbits Sun")

---

## Evaluation Architecture (`api/semantic_evaluator.py`)

The evaluator exposes the internal mechanics of AR.I.S for every question:
1. **Tokens & Candidate Extraction**: Evaluates tokenization and concept/entity candidate detection.
2. **Semantic Representation**: Uses `ArisSemanticEncoder` coordinate embeddings ($d=64$, character n-gram hashing and normalization) to represent the concepts.
3. **Relational Binding**: Extracts subject, relation span, and object without hardcoded patterns.
4. **Graph Path Discovery**: Searches symbolic graph reachability over active facts using multi-hop BFS.
5. **Epistemic State & Inhabitation**: Correctly detects ungrounded queries (`unsupported`) and respects relational inhibition bounds (e.g. denied negative hypotheses).

### CLI Command
Evaluation is executed cleanly via Django management command:
```bash
python manage.py brain evaluate_protocol --facts aris_training_facts.json --questions aris_question_training.json --output aris_evaluation_results.json
```

Or within the interactive brain shell:
```
aris-brain> evaluate
```

---

## Benchmark Results (Baseline Audit)

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Total Questions** | 35 | Evaluated across 10 categories |
| **Entity Accuracy** | **68.6%** | High candidate recognition across proper nouns |
| **Relation Accuracy** | **25.7%** | Canonical relation mapping across open phrasing |
| **Direction Accuracy** | **34.3%** | Subject-Object orientation preservation |
| **Semantic Target Accuracy**| **22.9%** | Correct semantic alignment of target variable |
| **Direct Answer Accuracy** | **0.0%** | Answer matches without external LLM/advisor synthesis |
| **2-Hop Accuracy** | **0.0%** | Multi-hop chain grounding without structural advisor |
| **3-Hop Accuracy** | **0.0%** | Multi-hop chain grounding without structural advisor |
| **4-Hop Accuracy** | **0.0%** | Multi-hop chain grounding without structural advisor |
| **Unsupported Detection** | **100.0%** | Zero hallucinations on ungrounded/unknown queries |
| **Paraphrase Accuracy** | **0.0%** | Sensitivity to phrase variations |
| **Spelling Robustness** | **0.0%** | Exact token requirement in candidate extraction |
| **Grammar Robustness** | **0.0%** | Interrogative inversion parsing |
| **Generalization Accuracy** | **19.4%** | Unseen DEV/TEST split performance |

---

## Empirical Failure Analysis & Primary Training Target

The empirical failure distribution identified across the 35 benchmark questions:
1. **RELATION_FAILURE**: 12 questions (34.3%)
2. **ENTITY_FAILURE**: 9 questions (25.7%)
3. **PATH_DISCOVERY_FAILURE**: 5 questions (14.3%)
4. **SEMANTIC_NORMALIZATION_FAILURE**: 3 questions (8.6%)

### Primary Training Target:
> **Relational discovery & canonical phrase abstraction**

**Root Cause**: While entity recognition (`Entity Accuracy: 68.6%`) and ungrounded detection (`Unsupported Detection: 100%`) are robust, question interrogative forms (e.g. *"Where does Arun work?"*, *"Who is the manager of Arun?"*, *"Which city is Axinix located in?"*) place the interrogative anchor at the start (`"Where"`, `"Who"`), causing the non-lexical positional binding layer to extract the interrogative pronoun as the subject or confuse the relation with auxiliary verbs (`"does"`). 

Strengthening abstract relational phrase discovery and interrogative frame canonicalization is the single highest-leverage priority for next-generation AR.I.S cognitive training.
