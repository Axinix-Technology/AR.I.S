# Goal-Conditioned Relational Reasoning & Target Compatibility (Phase 17.1)

## Overview
Phase 17.1 establishes the foundational mathematical formulation and learning architecture for **goal-conditioned relational reasoning** in AR.I.S. Prior to this phase, graph reasoning suffered from a fundamental bias: graph search unconditionally favored terminal leaf nodes and penalized intermediate nodes, while relation matching was limited to character substring overlaps. This caused terminal leaf overshoot (selecting a distant city instead of an intermediate company) and failed on predicate semantic variations (`founder` ↔ `founded`, `establish` ↔ `founded`).

Phase 17.1 eliminates hardcoded graph topology heuristics and replaces them with a learned tensor substrate evaluating whole-path goal compatibility.

---

## 1. Architectural Correction: Graph Engine Neutrality
The graph traversal engine (`ArisSymbolicGraph`) is restored to its proper role as an unbiased candidate generator. It is completely neutral with respect to graph topology:
- No intermediate node penalty (the previous `confidence *= 0.5` is removed).
- No leaf node sorting priority (the previous `is_leaf` sorting priority is removed).
- Candidate paths across all depths ($1 \le k \le 4$) are exposed to the cognitive tensor layer without topological bias.

---

## 2. Whole-Path Learning Target
The learned compatibility score is formulated as:

$$S_\theta(c, p, G) = f_\theta(\text{goal}, \text{anchor}, \text{path}, \text{candidate}, \text{candidate-type}, \text{evidence})$$

Contrastive learning enforces:

$$S_\theta(c^+, p^+, G) > S_\theta(c^-, p^-, G)$$

Where $c^+$ is the semantically valid target node for question $Q$, and $c^-$ are **counterfactual negative candidates drawn from the very same graph**:
1. **Overshot targets**: intermediate nodes passed over to reach a terminal leaf.
2. **Undershot targets**: intermediate nodes before reaching the correct depth.
3. **Wrong target type**: candidates violating the semantic target type (e.g. `GPE_COUNTRY` instead of `GPE_STATE`).
4. **Wrong branch / wrong relation**: candidates on unrelated branches.
5. **Wrong direction**: passive/active inverse mismatches.
6. **Unsupported reachable nodes**: reachable nodes when the requested relation is absent from the evidence.

---

## 3. Four Observable Learning Subspaces
To ensure transparent diagnosis and avoid conflating distinct cognitive capabilities, the learned engine maintains 4 separate subspaces:

### Subspace A: Goal → Path / Candidate Compatibility ($S_\theta$)
Evaluates whether the sequence of edges $(r_1, \dots, r_k)$ in path $p$ compositionally satisfies the question's goal $G$, terminating at candidate $c$.

### Subspace B: Predicate Semantic Alignment
Continuous metric manifolds capturing:
- Derivational morphology: `founder` ↔ `founded`, `sponsor` ↔ `sponsored by`
- Lexical / semantic alignment: `establish` ↔ `founded`, `supervises` ↔ `oversees`
- Relational paraphrastic transforms: `receives power` ↔ `transmits energy`

### Subspace C: Entity & Target Type Representation
Continuous manifold projections for semantic categories:
`ORGANIZATION`, `PERSON`, `GPE_CITY`, `GPE_STATE`, `GPE_COUNTRY`, `PRODUCT_TECHNOLOGY`, `AWARD_HONOR`.

### Subspace D: Epistemic Support
Joint evidence-backed evaluation:
$$\text{Path supported by evidence} \land \text{Path satisfies goal} \land \text{Candidate satisfies target type} \implies \text{SUPPORTED}$$
If no candidate path satisfies this joint condition, the reasoner emits `UNSUPPORTED` (`matches_count = 0`).

---

## 4. Verification & Constitutional Compliance
- **Rule 1 (Zero Hard Patches)**: Pure metric learning over continuous tensor representations; zero procedural keyword if-clauses or manual synonym dictionaries.
- **Rule 4 (Generic Intelligence)**: Evaluates arbitrary novel entities and graphs without hardcoded templates.
