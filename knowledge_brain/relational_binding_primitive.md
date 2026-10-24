# Relational Binding Primitive (Tier 3)

## Overview

Tier 3 is the **Relational Binding Primitive** of AR.I.S.
It answers the structural question:

```
WHO -- RELATION --> WHOM?
```

It operates purely on structure — no dictionaries, no string matching, no hardcoded relation templates.

---

## Architectural Position

```
Tier 1 -- Atomic Tokens & Exact Positional Coordinates (FROZEN)
          |
Tier 2 -- Concept & Entity Candidates (FROZEN)
          |
Tier 3 -- Relational Binding Primitive (CURRENT)
          |  subject --> relation span --> object
          |
Tier 4 -- Reasoning & Training Ground (FUTURE)
```

---

## Module

`api/relational.py` — `ArisRelationalBinder`

### Method: `bind_relations(text, tokens, candidates) -> list[dict]`

**Algorithm (purely structural):**
1. Filter candidates to named anchors: `entity_candidate` + `ambiguous_capitalized`.
   `compound_word` is excluded — it describes a lexical concept, not a named referent.
2. Partition anchors by `sentence_index`.
3. Within each sentence, slide a window over consecutive pairs `(E_i, E_{i+1})`.
4. The relation span = `text[E_i.char_end : E_{i+1}.char_start]` stripped of whitespace.
5. Emit a directional binding dict per pair.

**Directional invariant:** Subject = `E_i`, Object = `E_{i+1}` — determined by sequence order only.
The relation text itself is never inspected.

**Sentence locality:** Bindings never cross sentence boundaries.

---

## Why `ambiguous_capitalized` participates in Tier 3

At Tier 2, sentence-initial single words are classified `ambiguous_capitalized` because
capitalization alone does not prove entityhood. However, at Tier 3 we discover structural
positions — not semantic types. A sentence-initial word like `"Arun"` in `"Arun is brother of Bala."`
is structurally positioned as the subject anchor before the relation span. Excluding it would prevent
any binding from forming for sentences where the subject is sentence-initial.

`compound_word` remains excluded because it names a lexical phenomenon (`co-founders`, `well-known`),
not a discrete referent that can anchor a relational direction.

---

## Binding Schema

```json
{
  "index": 0,
  "sentence_index": 0,
  "subject": {
    "candidate_index": 0,
    "text": "Divya",
    "token_indices": [0],
    "char_start": 0,
    "char_end": 5
  },
  "relation": {
    "text": "is sister of",
    "normalized": "is sister of",
    "token_indices": [1, 2, 3],
    "char_start": 6,
    "char_end": 18
  },
  "object": {
    "candidate_index": 1,
    "text": "Akila",
    "token_indices": [4],
    "char_start": 19,
    "char_end": 24
  }
}
```

---

## Integration

- `api/brain.py` — `store_fact` runs Tier 3 after Tier 2 and persists `relations` + `relation_count`
- `api/brain.py` — `inspect_fact` retrocomputes relations for legacy facts (backward compatible)
- `api/management/commands/brain.py` — Tier 3 table in the cognitive inspection microscope
- `api/serializers.py` — `relations` and `relation_count` exposed in `FactOutputSerializer`

---

## Invariants (Verified in `RelationalBindingTier3Tests`)

| Test | Invariant |
|------|-----------|
| `test_exact_span_preservation_invariant` | `source[rel.char_start:rel.char_end] == rel.text` for all bindings |
| `test_directional_asymmetry_invariant` | Swapped subject/object produces reversed binding (not equal) |
| `test_multi_proposition_locality_invariant` | 2-sentence input produces exactly 2 bindings, sentence-partitioned |
| `test_zero_relation_dictionary_invariant` | Novel unseen relation phrase binds structurally |
| `test_no_binding_for_single_entity` | Single anchor -> zero bindings |
| `test_brain_stores_and_returns_relations` | Brain's `store_fact` persists and returns real relations |
| `test_legacy_fact_backward_compatibility_with_relations` | `inspect_fact` retrocomputes relations for legacy facts |

---

## Live Microscope Examples

### `"Divya is sister of Akila."`
```
TIER 3: RELATIONAL BINDINGS (WHO -- RELATION --> WHOM)
Idx  Subject               Relation Span            Object
---  ----                  ----                     ----
0    Divya                 is sister of             Akila
```

### `"Maya Lin joined Quantum Dynamics in Berlin."`
```
TIER 3: RELATIONAL BINDINGS (WHO -- RELATION --> WHOM)
Idx  Subject               Relation Span            Object
---  ----                  ----                     ----
0    Maya Lin              joined                   Quantum Dynamics
1    Quantum Dynamics      in                       Berlin
```

---

## Status
- Phase: 3 (Active)
- Tests: 23/23 passing — all on real cognitive output, no mocked cognitive methods
- Frozen: No (Tier 3 is the active development layer)
