# Relational Retrieval 1-Hop Traversal (Tier 4)

## Overview

Tier 4 is the **Deterministic 1-Hop Relational Retrieval Engine** of AR.I.S.
It establishes the fundamental query retrieval baseline over Tier 3 structural bindings:

```text
               (Subject, Relation, ?)      ──► Object     [FORWARD]
               (?, Relation, Object)      ──► Subject    [REVERSE]
               (Subject, ?, Object)      ──► Relation   [LINK DISCOVERY]
               (Subject, Relation, Object) ──► Confirmed  [VERIFICATION]
```

---

## Zero Spoon-Feeding Architecture

> [!IMPORTANT]
> **No English Grammar Dictionaries**:
> There are zero hardcoded question words (`who`, `what`, `where`, `does`, `did`).
> AR.I.S does not orchestrate answers using Python grammatical heuristics.

When any sentence is passed to AR.I.S as a natural query:
1. AR.I.S executes its **native Tier 1 $\rightarrow$ Tier 2 $\rightarrow$ Tier 3 pipeline** on the query text directly.
2. If the query forms a structural relational binding $(S_Q, R_Q, O_Q)$, AR.I.S compares that binding against stored bindings $(S_M, R_M, O_M)$ in memory.
3. If $R_Q == R_M$ and $O_Q == O_M$, the differing anchor $S_M$ is emitted as the answer.
4. If $S_Q == S_M$ and $R_Q == R_M$, the differing anchor $O_M$ is emitted as the answer.
5. If $S_Q == S_M$ and $O_Q == O_M$, the differing relation $R_M$ is emitted as the answer.
6. If the query does not match memory or does not form a structural binding, AR.I.S returns **0 matches**.

Because of this zero-spoon-feeding design, query parsing is **language-agnostic**:
- `"Who leads Axinix Technologies Development Team?"` $\rightarrow$ maps to `[Who] ──[leads]──► [Axinix Technologies Development Team]` $\rightarrow$ returns `Ajay`.
- `"Yaar leads Axinix Technologies Development Team?"` $\rightarrow$ maps to `[Yaar] ──[leads]──► [Axinix Technologies Development Team]` $\rightarrow$ returns `Ajay`.

---

## Core Modules

### 1. `api/retrieval.py`
- **`ArisRelationalQuery`**:
  - Holds `(subject, relation, object, direction, target_variable)`.
  - Supports explicit slots via `from_params()`.
  - Runs native Tier 1–3 pipeline on text via `from_text()` with zero dictionaries.
- **`ArisRetriever`**:
  - `execute_query(query, facts=None, limit=10)`: Traverses bindings, evaluates structural constraints, deduplicates answers, and returns full provenance (answer, source statement, fact ID, confidence, matched binding).

### 2. `api/brain.py`
- `ArisBrain.query_relations(query_text=None, subject=None, relation=None, obj=None, direction=None, limit=10)`:
  Central gateway method delegating to `ArisRetriever`.

### 3. CLI Command (`api/management/commands/brain.py`)
- `python manage.py brain query [query_text] [--subject S] [--relation R] [--object O] [-v]`

### 4. REST API Endpoint (`api/views.py` & `api/urls.py`)
- `GET /api/brain/query/?subject=...&relation=...`
- Returns HTTP 200 with structured JSON answers.

---

## Verification & Audit Proof

1. **Forward Traversal**: `(Ajay, leads, ?)` $\rightarrow$ `Axinix Technologies Development Team`.
2. **Reverse Traversal**: `(?, leads, Axinix Technologies Development Team)` $\rightarrow$ `Ajay`.
3. **Reverse Family Traversal**: `(?, is brother of, Bharathikannan)` $\rightarrow$ `Arun`.
4. **Link Discovery**: `(Ajay, ?, Axinix Technologies Development Team)` $\rightarrow$ `leads`.
5. **No Match Constraint Integrity**: `(Ajay, manages, ?)` $\rightarrow$ 0 matches; `(Ajay, leads, Testing Team)` $\rightarrow$ 0 matches.
6. **Natural Question Parsing Without Dictionaries**: `"Who leads ...?"` and `"Yaar leads ...?"` both work purely via structural intersection.
7. **Cognitive Test Integrity**: All 32 automated tests in `api/tests.py` execute on real cognitive representations with zero cognitive method mocking.
