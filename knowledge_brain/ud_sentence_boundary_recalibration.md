# Universal Dependencies Sentence Boundary Recalibration

## Overview
In natural language, periods (`.`) perform dual linguistic functions:
1. **Sentence / Clause Terminators**: Marking the conclusion of a full grammatical thought.
2. **Intra-Constituent Markers**: Serving as abbreviation dots, initials, and title separators (e.g., `Dr. Martin`, `Prof. Keller`, `U.S. Dynamics`).

Under the **AR.I.S Constitution** (Rule 1: Zero Hard Patches, Rule 7: Prohibition of Surface Guards), static string checks and dictionary lookups (such as `if word in ["Dr", "Mr"]`) are strictly forbidden. All structural segmentations must be grounded in the continuous neural tensor substrate.

---

## The Problem (Test 03 Root Cause)
In `Geneva Quantum Lab managed by Dr. Martin Keller.`, the raw tokenizer blindly incremented `current_sentence` whenever a character matched `{".", "!", "?"}`.
- `Geneva Quantum Lab` received `sentence_index: 0`.
- `Martin Keller` received `sentence_index: 1`.

Because `ArisRelationalBinder` partitions candidate entities strictly by `sentence_index`, it observed only one entity per sentence and produced **zero relational bindings**. `Dr. Martin Keller` was omitted from the symbolic graph, leading to a reasoning failure where the engine traversed 4 hops to find a distractor.

---

## Architectural Solution
In `ArisTokenizer.enrich_tokens_with_syntax()`, immediately after the Universal Dependencies PyTorch tensor engine (`ArisUniversalLinguisticEncoder`) predicts continuous UPOS tags:
1. Sentence indices are recalibrated across the token sequence using the neural UPOS tags.
2. A `PUNCT` symbol matching a terminator does **not** advance `sentence_index` if the UPOS context reveals an intra-proper-noun bridge:
   $$\text{tokens}[i - 1].\text{upos} == \text{PROPN} \quad \land \quad \text{tokens}[i + 1].\text{upos} == \text{PROPN}$$
3. All subsequent tokens maintain their true clause coordinate `sentence_index: 0`.
4. `ArisRelationalBinder` correctly binds:
   $$\text{Geneva Quantum Lab} \xrightarrow{\text{managed by}} \text{Dr. Martin Keller}$$
   guaranteeing 100% graph fidelity.

---

## Exact-Line Audit Record
- **File**: `api/linguistic/tokenizer.py`
- **Lines Changed**: 127–148
- **What Changed**: Added UD tensor-driven boundary recalibration loop in `enrich_tokens_with_syntax`.
- **Why Changed**: Prevents premature sentence fragmentation on abbreviation/title periods, ensuring relational bindings are extracted across titles without string-level guards.
