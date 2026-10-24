# Contextual Token Representation Experiment (Phase 0)

## Objective
Answer the fundamental scientific question:
> Does AR.I.S produce a different representation for the same token when its surrounding language context changes?

This experiment isolates **contextual representation** only. It does not evaluate semantic understanding, intentionality, goal reasoning, or language generation.

---

## Experimental Setup & Directives
- **Substrate**: Existing `ArisPhase0Model` (`api/linguistic/phase0_substrate.py`) — stacked `TransformerBlock` encoder with multi-head self-attention, GELU feed-forward network, and LayerNorm.
- **Serialization Boundary**: Existing `ArisCanonicalTokenizer` (`api/linguistic/canonical_tokenizer.py`).
- **Processing Pipeline**:
  $$\text{Raw Text} \xrightarrow{\text{ArisCanonicalTokenizer}} \text{Token IDs} \xrightarrow{\text{ArisPhase0Model}} \text{Hidden States} \in \mathbb{R}^{B \times L \times D} \xrightarrow{\text{Position Slice}} \mathbf{h}_{\text{apple}} \in \mathbb{R}^{D}$$
- **Zero Python Magic / No Surface Heuristics**: Zero regex overrides, zero dictionary mappings, zero hard patches, zero test-time mocks.

---

## Measured Findings

### 1. Reproducibility (Experiment 1)
- Input A: `"I ate an apple."`
- Input B: `"I ate an apple."`
- Cosine Similarity: **1.000000**
- Euclidean Distance: **0.000000**
- *Finding*: Identical token sequences yield identical representations, verifying deterministic computation.

### 2. Context Sensitivity (Experiment 2)
- Input A: `"I ate an apple."` ($\text{pos}=4$, preceding verb: "ate")
- Input B: `"Apple released a phone."` ($\text{pos}=1$, preceding start token, subject position)
- Cosine Similarity: **0.091669**
- Euclidean Distance: **10.782634**
- *Finding*: The token representation changes substantially when surrounding context and positional coordinates change.

### 3. Pairwise Cross-Context Matrix (Experiment 3)
```text
          A       B       C       D       
A         1.0000  0.0917  0.5028  0.9903  
B         0.0917  1.0000 -0.0368  0.1066  
C         0.5028 -0.0368  1.0000  0.4805  
D         0.9903  0.1066  0.4805  1.0000  
```
- Sentences:
  - A: `"I ate an apple."`
  - B: `"Apple released a phone."`
  - C: `"The apple fell from the tree."`
  - D: `"She bought an apple."`
- Structurally similar contexts (A and D: `PRON VERB DET apple .`) yield high alignment ($\cos \approx 0.9903$).
- Inverted syntactic roles (B vs. A or C) yield orthogonal/slightly negative alignment ($\cos \approx 0.0917$ and $-0.0368$).

### 4. Positional Verification (Experiment 4)
- `"Apple released a phone."` $\to$ Token `"Apple"`, ID 10, Position 1, Norm 8.0000
- `"I ate an apple."` $\to$ Token `"apple"`, ID 8, Position 4, Norm 8.0000
- *Finding*: Dynamic token localization ensures the representation extracted belongs strictly to the targeted lexical occurrence.

### 5. Context Perturbation (Experiment 5)
Controlled perturbation of the preceding predicate:
- `"I ate an apple."` $\to \cos = 1.000000, d = 0.000000$ (Base)
- `"I bought an apple."` $\to \cos = 0.996106, d = 0.706025$
- `"I dropped an apple."` $\to \cos = 0.996113, d = 0.705345$
- `"I peeled an apple."` $\to \cos = 0.995752, d = 0.737353$
- *Finding*: Perturbing a single surrounding word produces a measurable geometric shift in the target token's representation.

---

## Conservative Scientific Interpretation
> **The existing Phase 0 representation is context-sensitive for the tested examples.**

*Cautionary Boundary*: This observation demonstrates that multi-head self-attention and positional embeddings in `ArisPhase0Model` propagate contextual information across the sequence into the token's coordinate vector. It does NOT demonstrate semantic understanding of words or abstract concepts.
