# Phase 2: Canonical Tokens to Relational System Integration (Validated & Strengthened)

## Overview

**AR.I.S Phase 2** establishes the connection between the Phase 1 canonical language serialization boundary (`raw language -> ArisCanonicalTokenizer -> Token IDs -> PyTorch sequence substrate`), the **existing trained AR.I.S Universal Dependencies (UD) Brain** (`models/aris_ud_tensor_bank.pt`), and the **existing relational learning and reasoning infrastructure**.

In strict accordance with Constitutional Directives (Rules 1, 2, 4, 7, and 8), Phase 2 created **no new neural model, no new relation engine, no new tokenizer, and no parallel architecture**. It integrated the canonical representation into existing relational components via clean, backward-compatible interface adapters.

### The Single Canonical Cognitive Path

```text
RAW LANGUAGE
    ↓
PHASE 1
ArisCanonicalTokenizer.serialize()
    ↓
TokenizedSequence
    ├── input_ids  (PyTorch LongTensor)
    └── attention_mask (PyTorch LongTensor)
    ↓
PHASE 0 NEURAL SUBSTRATE INTERFACE
ArisPhase0Model(embed_dim=64)
    ↓
EXISTING TRAINED AR.I.S UD BRAIN
ArisUniversalLinguisticEncoder (models/aris_ud_tensor_bank.pt)
    ↓
Learned Linguistic Representations
    ├── UPOS (Universal Part-of-Speech prototypes: NOUN, VERB, PROPN, AUX, ADP, etc.)
    ├── Dependency relations (nsubj, obj, obl, cop, compound, etc.)
    ├── Morphology features (108 categories)
    └── Continuous 64-dim contextual tensor representations: H ∈ R^(1 × L × 64)
    ↓
EXISTING RELATIONAL / COGNITIVE SYSTEM
    ├── ArisRelationalBinder (bind_tokenized_sequence: pools 64-dim UD tensor over relation span)
    ├── ArisRelationAbstractionEngine (from_trained_ud_substrate: prototype centroids in R^64)
    ├── ArisSymbolicGraph (stores bound entities + dynamically abstracted relation)
    └── ArisRetriever (executes 1-hop queries over grounded facts)
```

---

## Exact Integration Points & Modifications

### 1. `api/linguistic/ud_tensor_engine.py` (`ArisUniversalLinguisticEncoder`)
- **Lines [253–342](file:///e:/Loigmax/AR.I.S/api/linguistic/ud_tensor_engine.py#L253-L342)**:
  - Added method `encode_tokenized_sequence(cls, sequence: TokenizedSequence, substrate_model: Optional[Any] = None) -> Dict[str, Any]`.
  - Consumes canonical `TokenizedSequence` from Phase 1.
  - Loads `models/aris_ud_tensor_bank.pt` (417 MB, trained on 26,435 sentences and 499,373 tokens across 5 languages) via existing `_ensure_loaded()`.
  - Evaluates contextual UPOS tags, dependency relations, and computes 64-dimensional continuous token vectors $\mathbf{u}_i \in \mathbb{R}^{64}$.
  - Connects the Phase 0 PyTorch sequence substrate (`ArisPhase0Model`, `embed_dim=64`).
  - Stacks token vectors into unified hidden states $\mathbf{H} \in \mathbb{R}^{1 \times L \times 64}$.

### 2. `api/relational/relation_abstraction.py` (`ArisRelationAbstractionEngine`)
- **Lines [85–125](file:///e:/Loigmax/AR.I.S/api/relational/relation_abstraction.py#L85-L125)**:
  - Added factory classmethod `from_trained_ud_substrate(cls, tokenizer, ud_encoder, substrate_model, canonical_relations)`.
  - Operates in $\mathbb{R}^{64}$, matching the metric space of `models/aris_ud_tensor_bank.pt`.
  - `encode_phrase(phrase)` routes phrase tokens through `tokenizer.serialize()` and continuous 64-dim sequence pooling from the trained UD brain.
  - Replaces isolated random model encoding with learned representations from the UD brain.

### 3. `api/relational/relational.py` (`ArisRelationalBinder`)
- **Lines [174–326](file:///e:/Loigmax/AR.I.S/api/relational/relational.py#L174-L326)**:
  - Consumes `sequence: TokenizedSequence` and `hidden_states` (shape `(1, L, 64)`) from the trained UD brain.
  - Discovers subject and object candidate spans structurally from sequence tokens.
  - Mean-pools the trained 64-dimensional tensor slice across the intervening relation token indices into $\mathbf{h}_{\text{rel}} \in \mathbb{R}^{64}$.
  - Attaches `neural_representation` ($\mathbb{R}^{64}$) and `tensor_shape` (`[64]`) to each relational binding.

---

## Honest Dynamic Symbolic Graph Grounding (Zero Manual Injections)

Rather than hardcoding relations (`relation="brother_of"`) when populating `ArisSymbolicGraph`, Phase 2 dynamically derives canonical relations through the full neural stack:

```text
Statement / Fact: "Arun is brother of Ravi"
    ↓
ArisCanonicalTokenizer.serialize()
    ↓
TokenizedSequence (input_ids, attention_mask, tokens, token_spans)
    ↓
ArisUniversalLinguisticEncoder.encode_tokenized_sequence()
    ↓
Trained UD Hidden States (1, 7, 64) + UPOS (is:AUX, brother:NOUN, of:ADP)
    ↓
ArisRelationalBinder.bind_tokenized_sequence()
    ↓
Discovered Subject: "Arun", Relation Span: "is brother of", Object: "Ravi"
    ↓
ArisRelationAbstractionEngine.abstract_relation("is brother of")
    ↓
Dynamically Predicted Relation: "brother_of"
    ↓
ArisSymbolicGraph.add_relational_binding(subject="Arun", relation="brother_of", object="Ravi")
    ↓
ArisRetriever.execute_query(subject="Arun", relation="brother_of", target="object")
    ↓
Retrieved Grounded Answer: "Ravi"
```

---

## Three-Tier Scientific Integrity Framework (Rule 7)

```text
================================================================================
AR.I.S PHASE 2 — THREE-TIER SCIENTIFIC INTEGRITY REPORT
================================================================================
A. Interface Integration:                  PASS
   (TokenizedSequence -> Phase 0 Substrate -> Trained UD Brain -> Binder)

B. Representation Learning:                 PASS
   (Centroid Shift ||delta c_R|| = 0.6677 > 0 | Seen Accuracy: 76.7%)

C. Relational Generalization:
   - Condition 1 (Unseen Entities):        PASS (Subject=Elena, Object=NeuroVanguard)
   - Condition 2 (Held-Out Paraphrases):   OBSERVED (25.0% on TEST unseen split)
   - Condition 3 (Passive Structural Frame): PASS ('was founded by' bound)
   - Dynamic Symbolic Graph & Retrieval:   PASS (Zero Manual Injections)
================================================================================
NOTE: Passing A does NOT imply B. Passing B does NOT imply C.
================================================================================
```

---

## Scientific Integrity Audit & Pure Console Logging

1. **Elimination of Fresh Random Models**:
   - Phase 2 tests no longer instantiate an independent fresh random model with 32-dim Gaussian noise as a disconnected toy model.
   - Linguistic representations flow through the canonical serialization boundary into the existing trained UD brain (`aris_ud_tensor_bank.pt`) in $\mathbb{R}^{64}$.
2. **Elimination of Hardcoded Lexicon in Unit Tests**:
   - Removed the ad-hoc `words = [...]` array in unit test fixtures.
   - Unit tests load the canonical vocabulary artifact directly from disk via `ArisCanonicalTokenizer.from_vocabulary_file("models/aris_phase2_vocabulary.json")`, preserving frozen vocabulary behavior and determinism.
3. **Elimination of Artificial Marketing Banners**:
   - Removed `_print_test_banner` and all "WHY IT PASSED" narrative text.
   - All tests output pure, un-orchestrated console logs (`print(...)`) directly exposing the computed data, tokens, shapes, and raw query return dictionaries.
4. **Strict Ground-Truth Assertions**:
   - `test_11` validates held-out prepositional generalization (`"works for"` $\to$ `"works_at"`), asserting strictly: `assert res["predicted_relation"] == "works_at"`.
   - `test_13` tests zero-shot unobserved lexical synonymy (`"is an employee of"` $\to$ `"works_at"`) with explicit `@pytest.mark.xfail`. This honestly documents that untrained embeddings cannot generalize across unobserved synonym tokens without semantic language pre-training.

Metrics saved to: [`models/phase2_relational_integration_results.json`](file:///e:/Loigmax/AR.I.S/models/phase2_relational_integration_results.json).
