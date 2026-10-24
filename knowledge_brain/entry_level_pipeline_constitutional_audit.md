# Entry-Level Pipeline Constitutional Violation Audit & Failure Diagnosis

> **Constitutional Audit Date**: 2026-10-04  
> **Directives**: Rule 1 (Zero Hard Patches), Rule 4 (Generic Intelligence Over Code-Based Orchestration), Rule 7 (Prohibition of Surface Guards & Mandatory Exact-Line Audit)

---

## 1. Executive Context & Invariant

AR.I.S is fundamentally an **Artificial Intelligent System**, not a procedural Python orchestration pipeline. All linguistic, structural, and semantic classifications must emerge from trained neural tensor substrates (`CognitiveLinguisticFrame`, `ArisUniversalLinguisticEncoder`, `ArisCanonicalTokenizer`), never from procedural string methods, regular expressions, hardcoded tag lists, or hash functions.

This audit establishes the exact failure points of the live entry-level pipeline and maps out the root-cause transition plan.

---

## 2. The `iamarun` Failure Demonstration

When an utterance lacks whitespace boundaries (e.g. `iamarun`), an intelligent cognitive system must perform subword/character neural segmentation to deduce constituent concepts:
$$\text{"iamarun"} \xrightarrow{\text{Learned Neural Substrate}} [\text{"i"}, \text{"am"}, \text{"arun"}] \rightarrow \text{DECLARATIVE}(\text{subject}=\text{"i"}, \text{relation}=\text{"am"}, \text{object}=\text{"arun"})$$

In the current codebase, `api/linguistic/tokenizer.py` relies on `re.compile(r"(\w+(?:'\w+)?)|([^\w\s])")`. The pattern greedily matches `iamarun` as a single opaque token. Downstream candidate extraction fails, relational binding yields zero bindings, and MD5 semantic hashing fails to find any exemplar match.

---

## 3. Violation Summary Registry

Across 7 files in the live pipeline, **38 exact-line constitutional violations** were identified:

1. **[tokenizer.py](file:///e:/Loigmax/AR.I.S/api/linguistic/tokenizer.py)**: 13 violations. Word boundary splitting by regex `\w+`, hardcoded sentence terminators `{".", "!", "?"}`, character-level `.isdigit()`, `.isalnum()`, `.isupper()` checks, and string-slicing path guards.
2. **[pipeline.py](file:///e:/Loigmax/AR.I.S/api/linguistic/pipeline.py)**: 3 violations. Hardcoded UPOS exclusion sets (`_NON_ENTITY_UPOS`, `_NOMINAL_UPOS`) and procedural pure-number checks.
3. **[syntactic_normalizer.py](file:///e:/Loigmax/AR.I.S/api/linguistic/syntactic_normalizer.py)**: 10 violations. Explicit `.endswith(".")`, `.startswith("wh")`, substring `"by"`, `"of"`, and `"'s"` checks.
4. **[pragmatic_tensor_engine.py](file:///e:/Loigmax/AR.I.S/api/cognitive/pragmatic_tensor_engine.py)**: 3 violations. Hardcoded threshold `0.82` and raw Python string `.split()[0]`.
5. **[epistemic_state.py](file:///e:/Loigmax/AR.I.S/api/cognitive/epistemic_state.py)**: 5 violations. Punctuation stripping `.rstrip("?!.")`, equality checks `t.get("raw") == "="`, and `.isalpha()`.
6. **[brain.py](file:///e:/Loigmax/AR.I.S/api/cognitive/brain.py)**: 1 violation. String equality check against `"?"`.
7. **[semantic_representation.py](file:///e:/Loigmax/AR.I.S/api/semantic/semantic_representation.py)**: 3 violations. Character n-gram MD5 hash projection modulo 64 providing zero semantic intelligence.

---

## 4. Root-Cause Architectural Remedy

1. **Neural Segmentation**: Transition entry tokenization from procedural regex to the Phase 0 canonical vocabulary substrate with learned subword scoring.
2. **Learned Embedding Projection**: Deprecate MD5 hash projections in `ArisSemanticEncoder`; wire the Phase 0 PyTorch Transformer embeddings and self-attention tensor manifold into the live speech act classification flow.
3. **Dependency-Driven Normalization**: Replace string matching in `syntactic_normalizer.py` with Universal Dependencies graph relations (`nsubj`, `nsubj:pass`, `obj`, `nmod:poss`, `case`).
