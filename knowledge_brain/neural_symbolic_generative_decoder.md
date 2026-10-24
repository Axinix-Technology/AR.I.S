# Neural-Symbolic Generative Language Decoder (Fluid Text Synthesis)

**Classification**: Cognitive Layer / Phase 2 Natural Language Generation Substrate  
**Module Reference**: [generative_decoder.py](file:///e:/Loigmax/AR.I.S/api/cognitive/generative_decoder.py)  
**Cross-References**: [ArisBrain](file:///e:/Loigmax/AR.I.S/api/cognitive/brain.py), [Deontic & Affective Cognition](file:///e:/Loigmax/AR.I.S/knowledge_brain/deontic_affective_cognition.md), [Dialogue Context & Coreference](file:///e:/Loigmax/AR.I.S/knowledge_brain/dialogue_context_coreference_phase1.md)

---

## 1. Architectural Purpose & Overview

The **Neural-Symbolic Generative Language Decoder** (`ArisGenerativeDecoder` & `ArisGroundedGenerator`) provides AR.I.S with fluid, expressive, human-grade conversational natural language generation (NLG) while strictly enforcing **100% factual grounding** and **epistemic self-awareness**.

Unlike commercial black-box LLMs that hallucinate facts or stochastic parrots that generate ungrounded prose, AR.I.S separates **symbolic cognitive derivation** (relational graph traversal, mathematical AST proofs, epistemic deficit analysis) from **neural surface realization**.

```
+--------------------------------------------------------------------------------------------------+
|                            AR.I.S NEURAL-SYMBOLIC NLG SUBSTRATE                                  |
+--------------------------------------------------------------------------------------------------+
| 1. Symbolic Reasoning State    ──► [Path Provenance, Epistemic State, Deictic Coreferences]      |
| 2. Deontic & Affective Sensor  ──► [Ethical Invariants (O, F, P), Emotional Valence/Pragmatics]  |
| 3. Structural Discourse Planner──► [Discourse Role, Sentence Cadence, Salience Ordering]        |
| 4. Neural-Symbolic Realizer    ──► [Continuous Prototype Smoothing, Agreement, Fluid Syntax]     |
| 5. Grounded Factual Verifier   ──► [Zero-Hallucination Invariant: All Named Entities Grounded]    |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Core Capabilities & Invariants

### A. Grounded Single-Hop & Multi-Hop Narrative Synthesis
* **Single-Hop Triples**: Converts $(s, r, o)$ triples into rich grammatical sentences with dynamic active/passive inflection without hardcoded static templates.
* **Multi-Hop Explanations**: Chains graph provenance paths into explainable, step-by-step deductive narratives (*"According to verified knowledge, A is connected to B through [relation], and B leads to C. Therefore, [conclusion]."*).

### B. Epistemic Boundary Conditioning
* **`KNOWN`**: Synthesizes affirmative, grounded factual statements.
* **`UNSUPPORTED`**: Expresses honest epistemic boundaries (*"I do not currently have verified records regarding..."*) rather than fabricating answers.
* **`STORED`**: Generates natural confirmations when learning new facts into memory.
* **`ACQUIRABLE`**: Verbalizes tool-assisted reasoning (Math evaluations, web lookups) with transparent execution summaries.

### C. Zero-Hallucination Invariant
* Every named entity, quantitative value, and structural claim in the generated output is cross-verified against the symbolic inputs.
* Hallucinated entities or ungrounded assertions are mathematically rejected before emission.

### D. Zero-Shot Alien Generalization
* Operates seamlessly across synthetic/alien tokens (e.g. *"Zorblax flarps Krelik"*) without requiring english dictionary lookups.

---

## 3. Integration with ArisBrain

All generation is invoked through `ArisBrain`:
1. `ArisBrain.process_dialogue_turn()` receives the user's message.
2. Evaluates deontic safety and affective pragmatics.
3. Resolves coreferences and runs relational/epistemic reasoning.
4. Passes the grounded state to `ArisGroundedGenerator.generate()`.
5. Returns a rich, conversational, fully-audited response.
