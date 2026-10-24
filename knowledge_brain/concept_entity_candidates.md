# Concept & Entity Candidate Layer (AR.I.S Tier 2)

## Architectural Role
The **Concept & Entity Candidate Layer** bridges the gap between atomic lexical tokens (Tier 1) and relational binding (Tier 3).
It forms candidate conceptual units while strictly adhering to the principle:
> **Representation must not pretend to know something that it hasn't actually established.**

---

## The Cognitive Hierarchy

```text
Tier 1: Atomic Tokens & Exact Positional Coordinates
        [axinix (0:6), technologies (7:19), co (30:32), - (32:33), founders (33:41)]
                   │
                   ▼
Tier 2: Concept & Entity Candidates (Compounds & Span Mappings)
        [Axinix Technologies] -> entity_candidate (0:19, tokens [0..1])
        [co-founders]         -> compound_word    (30:41, tokens [4..6])
        [Ajay]                -> entity_candidate (57:61, token [11])
        [Guru Prasath]        -> entity_candidate (73:85, tokens [15..16])
                   │
                   ▼
Tier 3: Relational Binding (Future Phase 3)
        is_cofounder_of(Guru Prasath, Axinix Technologies)
                   │
                   ▼
Tier 4: Reasoning & Learning Ground
```

---

## Structural Candidate Invariants (Zero Hard Patches)
To eliminate hardcoded dictionaries or common-word lists, candidate extraction relies purely on **structural and orthographic invariants**:

1. **Hyphenated Compounds (`compound_word`)**:
   - Generic structural pattern: `[word] [-] [word]` with contiguous character spans.
   - Preserves Tier 1 tokens (`co`, `-`, `founders`) while Tier 2 binds the compound span `co-founders` (`30:41`).
2. **Multi-Word Capitalization (`entity_candidate`)**:
   - Contiguous Title Case words within the same sentence (e.g. `Axinix Technologies`, `Guru Prasath`, `Maya Lin`, `John Smith`).
   - English grammar never forces multiple contiguous words to capitalize merely due to sentence position.
3. **Single Capitalized Words**:
   - **Inside sentence body**: Generates an `entity_candidate` (e.g. `Ajay`, `Boopalan`, `Berlin`).
   - **Sentence-initial single word**: Marked as `ambiguous_capitalized` (e.g. `The`, `Apple`) because sentence-start capitalization is structurally ambiguous without semantic context.

---

## Two-Tier Cognitive Inspection Microscope

```powershell
python manage.py brain inspect <fact_id>
```

Provides immediate visual inspection of both tiers:
1. **Tier 1**: Token index, normalized token, token type, sentence index, character span, and relative position.
2. **Tier 2**: Concept/candidate index, span text, candidate type (`compound_word`, `entity_candidate`, `ambiguous_capitalized`), constituent token indices, and exact character spans.
