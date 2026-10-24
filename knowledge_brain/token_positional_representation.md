# Token & Positional Representation Primitive (AR.I.S)

## Architectural Principle: Representation Primitive, Not Semantic Reasoning

```text
                 AR.I.S
                   │
                   ▼
             ArisBrain
                   │
          ┌────────┴────────┐
          ▼                 ▼
     Memory Storage     Representation
          │                 │
       MongoDB        ArisTokenizer
                            │
                  ┌─────────┼─────────┐
                  ▼         ▼         ▼
                Token     Position   Span
                  │         │         │
                  └─────────┼─────────┘
                            ▼
                     Future Binding
                            │
                            ▼
                       Reasoning
```

### Critical Non-Negotiable Directives:
1. **Representation Layer Only**:
   - The token and positional layer is a **pure representation primitive**.
   - Positional proximity is strictly a lexical retrieval ranking heuristic.
   - Positional proximity must **never** be mistaken for semantic reasoning or relational inference (e.g. proximity between `brother` and `Arun` does not prove `brother_of(Arun, X)`).
2. **Deterministic Guarantee**:
   - For any identical input text, `ArisTokenizer` generates the exact same tokens, indices, spans, sentence boundaries, and normalized representations.
3. **No Premature Neural/Vector Abstractions**:
   - Zero embeddings, zero attention weights, zero black-box neural approximations at this stage. Pure transparent computational primitives:
     `TEXT -> TOKENS -> NORMALIZATION -> POSITION -> SPAN -> DISTANCE`.

---

## The 6 Acceptance Invariants

| # | Invariant | Specification | Verification Method |
|---|---|---|---|
| **1** | **Exact Lexical Preservation** | `raw == source[char_start:char_end]` for every single token without exception. | Unit test verifying substring slice against raw string. |
| **2** | **Deterministic Sequential Indexing** | `sequence_index: 0, 1, 2, ..., N-1` with zero gaps. | Monotonic zero-based progression check. |
| **3** | **Sentence Locality** | `sentence_index: 0 0 ..., 1 1 ..., 2 2 ...` tracking sentence boundaries across punctuation (`.`, `!`, `?`). | Sentence index increment test. |
| **4** | **Relative Position Coordinate** | $0.0 \le \text{relative\_pos} \le 1.0$ computed as $\frac{\text{sequence\_index}}{N - 1}$ (or $0.0$ if $N \le 1$). | Range and boundary check. |
| **5** | **Deterministic Distance Metric** | $\text{distance}(t_a, t_b) = \| \text{sequence\_index}_a - \text{sequence\_index}_b \|$. | Symmetric distance verification. |
| **6** | **Strict Backward Compatibility** | Legacy facts created prior to Phase 2 retrieve cleanly with default values and generate tokens on-demand during inspection. | Integration regression test. |

---

## Cognitive Inspection Microscope

AR.I.S provides a dedicated terminal inspection microscope to visually verify what the cognitive layer received:

```powershell
python manage.py brain inspect <fact_id>
```

### Output Example:
```text
AR.I.S TOKEN INSPECTOR
--------------------------------------------------------
Statement:
Arunbharathi founded Axinix Technologies.
Idx  Token           Type    Sentence  Char Span   Rel.Pos
---  -------------   ------  --------  ----------  -------
0    arunbharathi    word    0         0:12        0.00
1    founded         word    0         13:20       0.25
2    axinix          word    0         21:27       0.50
3    technologies    word    0         28:40       0.75
4    .               symbol  0         40:41       1.00

Token count: 5
```
