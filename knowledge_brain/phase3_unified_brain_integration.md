# Phase 3: Unified Cognitive Brain Architecture

## Overview

**AR.I.S Phase 3** connects all five existing trained cognitive tensor banks into **ONE unified cognitive system** operating through the single canonical serialization and neural substrate boundary:

$$\text{Raw Language} \xrightarrow{\text{ArisCanonicalTokenizer}} \text{TokenizedSequence} \xrightarrow{\text{Phase 0 Interface}} \text{Trained Cognitive Substrates} \in \mathbb{R}^{64}$$

In strict compliance with the Core Directives (Rules 1, 4, and 7):
* **Zero New Models or Tokenizers**: Uses only the existing 5 trained models in `models/`.
* **Zero Parallel Pipelines**: All cognitive consumers natively consume `TokenizedSequence` and continuous $\mathbb{R}^{64}$ tensors.
* **Pure Console Logging**: No artificial marketing banners or test result orchestrations.

---

## The Single Connected Cognitive Path

```text
RAW NATURAL LANGUAGE INPUT
    ↓
PHASE 1 CANONICAL SERIALIZATION BOUNDARY
ArisCanonicalTokenizer.serialize()
    ↓
TokenizedSequence
    ├── input_ids       (PyTorch LongTensor)
    ├── attention_mask  (PyTorch LongTensor)
    ├── tokens          (discrete string tokens)
    └── token_spans     (exact character coordinates)
    ↓
PHASE 0 NEURAL SUBSTRATE INTERFACE
Continuous Sequence Coordinates in R^64
    ↓
EXISTING TRAINED COGNITIVE SUBSTRATES (All aligned in R^64)
    ├── 1. Universal Dependencies Substrate (models/aris_ud_tensor_bank.pt)
    │      UPOS prototypes, 76 dependency relations, 108 morphological categories, H ∈ R^(1 × L × 64)
    ├── 2. Pragmatics & Illocutionary Substrate (models/aris_pragmatic_tensor_bank.pt)
    │      Speech acts (DECLARATIVE, INTERROGATIVE, COMMAND), deictic roles, question intentions
    ├── 3. Relational Binding Substrate (ArisRelationalBinder & Abstraction)
    │      Structural candidate discovery, 64-dim mean-pooled relation tensors, prototype centroids
    ├── 4. Structural Frame Substrate (models/aris_frame_classifier.pt)
    │      Parameter open-variable (?) vs bound anchor slot classification
    ├── 5. Goal & Entity Substrate (models/aris_goal_reasoning_tensor_bank_v1.pt)
    │      24 Predicate clusters, 10 entity semantic types, 192-dim whole-path compatibility net
    └── 6. Affordance & Epistemic Substrate (models/aris_affordance_tensor_bank.pt)
           Capability affordances (command, browser, math, retrieval), 5 epistemic states
    ↓
DOWNSTREAM REASONING & MEMORY
    ├── ArisSymbolicGraph (grounded entity nodes and dynamically abstracted relations)
    ├── ArisRetriever (1-hop deterministic fact retrieval and provenance extraction)
    └── ArisBrain (central memory and reasoning orchestrator)
```

---

## Subsystem Connection Details & Exact Line Audits

### 1. `api/cognitive/pragmatic_tensor_engine.py` (`ArisPragmaticTensorEngine`)
- **Added Methods**:
  - `_pool_sequence_vector(cls, sequence, hidden_states)`: Extracts normalized continuous vector in $\mathbb{R}^{64}$.
  - `classify_speech_act_sequence(cls, sequence, hidden_states)`: Projects sequence representations over `_speech_matrices` in $\mathbb{R}^{64}$ to classify `DECLARATIVE`, `INTERROGATIVE`, `COMMAND`, etc.
  - `infer_interrogative_role_sequence(cls, sequence, hidden_states)`: Projects operator tokens over `_interrogative_matrices` to infer question intentions (`location`, `person`, `time`, `cause`).
  - `classify_predicate_role_sequence(cls, sequence, hidden_states)`: Classifies communicative predicate roles.
- **Backward Compatibility**: `classify_speech_act(utterance)` and `infer_interrogative_role(token)` serialize via `ArisCanonicalTokenizer` and delegate to the sequence methods.

### 2. `api/cognitive/goal_reasoning_tensor_engine.py` (`ArisGoalReasoningTensorEngine`)
- **Added Methods**:
  - `_pool_vector(cls, sequence, hidden_states)`: Extracts normalized continuous vector in $\mathbb{R}^{64}$.
  - `align_predicates_sequence(cls, seq_a, seq_b, hs_a, hs_b)`: Aligns relational predicates directly between canonical `TokenizedSequence` representations across 24 predicate cluster manifolds.
  - `predict_target_type_sequence(cls, sequence, hidden_states)`: Projects question tokens over `_target_type_matrices` to predict target semantic categories (`ORGANIZATION`, `PERSON`, `GPE_CITY`, etc.).
- **Backward Compatibility**: `align_predicates(rel_a, rel_b)` and `predict_target_type(query_text)` serialize via `ArisCanonicalTokenizer`.

### 3. `api/cognitive/affordance_tensor_engine.py` (`ArisAffordanceTensorEngine`)
- **Added Methods**:
  - `_pool_vector(cls, sequence, hidden_states)`: Extracts normalized continuous vector in $\mathbb{R}^{64}$.
  - `predict_epistemic_and_affordance_sequence(cls, sequence, hidden_states)`: Passes pooled sequence vectors directly into `ArisAffordanceNet` trunk ($\mathbb{R}^{64} \to \mathbb{R}^{64}$). Emits epistemic states (`KNOWN`, `DERIVABLE`, `ACQUIRABLE`, `UNSUPPORTED`) and capability affordances (`command_prompt`, `browser_access`, `math_computation`, `memory_retrieval`).
- **Backward Compatibility**: `predict_epistemic_and_affordance(query_text)` serializes via `ArisCanonicalTokenizer`.

### 4. `api/linguistic/neural_frame_classifier.py` (`ArisNeuralFrameLearner`)
- **Added Methods**:
  - `predict_from_relational_binding(cls, binding, sequence)`: Consumes structural relational bindings emitted by `ArisRelationalBinder.bind_tokenized_sequence(sequence)` to classify which slot is the open parameter variable (`?`) vs bound anchor.

### 5. `api/cognitive/brain.py` (`ArisBrain`)
- **Added Methods**:
  - `store_fact_sequence(cls, sequence, topic, tags, ...)`: Ingests facts directly from canonical `TokenizedSequence`.
  - `perceive_capability_affordances_sequence(cls, sequence)`: Evaluates dynamic capability affordances directly from canonical sequence input.

---

## Verification Summary

* **Combined Phase 0, 1, 2, and 3 Tests**:
  `pytest api/tests/test_phase0_substrate.py api/tests/test_phase1_language_boundary.py api/tests/test_phase2_relational_integration.py api/tests/test_phase3_brain_integration.py -v`
  **35 passed in 9.36s** (Zero failures, Zero xfails)
* **Broader Cognitive Pipeline Regression**:
  `pytest api/tests/test_cognitive_pipeline.py`
  **53 passed in 11.66s** (Zero regressions)
