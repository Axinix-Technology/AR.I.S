# Declarative Cognitive Training & Multi-Dimensional Understanding Dataset

## Overview & Architectural Motivation
In accordance with **Rule 1 (Zero Hard Patches)** and **Rule 4 (Generic Intelligence)**, AR.I.S does not parse or interpret declarative assertions through hardcoded keyword matches or rigid procedural heuristic scripts. Instead, assertions are mapped into continuous vector coordinate projections, token-level deictic and thematic frames, directional relational triples, and cross-projected interrogative queries.

The dataset [`data/declarative_cognitive_training.json`](file:///e:/Loigmax/AR.I.S/data/declarative_cognitive_training.json) establishes the standard ground-truth benchmark and supervision dataset for multi-dimensional understanding of declarative statements.

---

## Multi-Dimensional Representation Architecture
For every declarative statement, AR.I.S models 8 discrete dimensions of cognitive understanding:

1. **Token-Level Decomposition & Morphosyntactic Annotations**:
   - POS tags, named entity tags, grammatical functions, and subword n-gram fingerprints for character-level continuous vector embedding.
2. **Cognitive Semantic Frames**:
   - Explicit thematic roles (`Agent`, `Patient`, `Location`, `Service`, `Employer`, `Leader`) grounded in domain-agnostic semantic frames (`Organizational_Founding`, `Residence_And_Habitation`, `System_Network_Execution`, `Speaker_Identity_Assertion`, `Professional_Employment`).
3. **Relational Triples (Subject-Predicate-Object)**:
   - Canonical forward bindings, inverse relations, and confidence ratings.
4. **Syntactic & Paraphrastic Variations**:
   - Active, passive, cleft, nominalized, and conversational reformulations representing identical propositional meaning.
5. **Cross-Lingual Equivalent Expressions**:
   - Multi-lingual alignments (e.g., French, German, Tamil transliterated, Hindi transliterated).
6. **Logical Entailments & Deductions**:
   - Downstream cognitive assertions that logically follow from the given fact.
7. **Interrogative Cross-Projections (Wh-Question Bindings)**:
   - Exhaustive set of wh-interrogatives (`who`, `where`, `what`, `which`, `is/does`) that query the statement's variables and their expected answer bindings.
8. **Pragmatics & Discourse Grounding**:
   - Speech act classification, discourse salience ranking, and dynamic deictic/anaphoric antecedent bindings (`he` $\to$ male person, `she` $\to$ female person, `it` $\to$ inanimate entity/system).

---

## Grounded Declarative Corpus Entries
The dataset covers the canonical conversational knowledge corpus:
- `DECL_001`: *"Arun is the founder of Axinix."*
- `DECL_002`: *"Dhivya lives in Chennai."*
- `DECL_003`: *"The server is running on port 8000."*
- `DECL_004`: *"TCS is a technology consulting company."*
- `DECL_005`: *"My name is Arun."*
- `DECL_006`: *"She is working in TCS."*
- `DECL_007`: *"Workhub is an ERP software."*
- `DECL_008`: *"QuantumEngine was developed in Zurich."*
- `DECL_009`: *"It is located in Berlin."*
- `DECL_010`: *"She manages the team."*
- `DECL_011`: *"He is an engineer at Google."*
- `DECL_012`: *"The headquarters is located in California."*

---

## Integration with AR.I.S Tensor Engine
The dataset is loaded and compiled by [`ArisPragmaticTensorEngine`](file:///e:/Loigmax/AR.I.S/api/cognitive/pragmatic_tensor_engine.py) into in-memory normalized PyTorch metric tensors (`models/aris_pragmatic_tensor_bank.pt`), ensuring sub-millisecond vector similarity calculations at runtime with **zero disk I/O** during turn execution.
