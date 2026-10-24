# Epistemic Cognition & Dynamic Capability Selection (Tier 6-A)

## 1. Overview & Philosophical Foundation
The core principle of **Tier 6-A** is that a Cognitive Brain must never use procedural keyword dispatch (e.g. `if "search" in query: browse()`).

Instead, the system operates as a closed cognitive loop:
$$\text{Perception (Tiers 1–3)} \longrightarrow \text{Goal Formation} \longrightarrow \text{Epistemic Evaluation} \longrightarrow \text{Capability Matching} \longrightarrow \text{Action} \longrightarrow \text{Memory Update}$$

---

## 2. The 5 Formal Epistemic States

| State | Definition | Operational Consequence |
| :--- | :--- | :--- |
| **`KNOWN`** | Memory directly satisfies the goal (exact entity/relation anchor present). | Produce answer directly from memory graph with **zero tool dispatches**. |
| **`DERIVABLE`** | Memory contains sufficient information or numbers, but requires execution/inference (e.g. `10 - 4` or birth/death years). | Dispatch local computation or inference capability to resolve deficit. |
| **`ACQUIRABLE`** | Required information is missing from memory, but matches an external intake contract (`browser_access`). | Dispatch external sensor; ingest fresh facts, tokens, and relations into memory. |
| **`UNSUPPORTED`** | Required information is absent and no registered capability can bridge the gap. | Clean report of inability to answer without hallucinations. |
| **`AMBIGUOUS`** | Goal or parameters cannot be uniquely grounded from input. | Formulate structural clarification requirement. |

---

## 3. Structural Goal Formation (No Trigger Words)
Goals are formulated strictly from structural invariants:
* **Mathematical Operations**: Detected by sequential numbers and primitive operators (`+`, `-`, `*`, `/`).
* **Relational Frames**: Derived from Tier 3 `Subject --Relation--> Object` bindings.
* **Concept & Entity Targets**: Derived from Tier 2 candidate spans (multi-token capitalization and compound nouns).
* **Zero Hardcoded Word Dictionaries**: There are no trigger lists (`search`, `browser`, `google`, `for`, `in`).

---

## 4. Capability Contracts (State Transformers)
Capabilities publish structural contracts:
1. **`ACCEPTS`**: Prerequisites (domain, required parameters).
2. **`PRODUCES`**: State transitions (e.g. `ACQUIRABLE -> KNOWN` or `DERIVABLE -> KNOWN`).

The reasoner calculates the **Epistemic Deficit** ($\Delta = \text{Goal} - \text{Memory}$) and matches the capability whose `PRODUCES` satisfies the deficit.

---

## 5. The Law of Memory Priority (Zero Premature Tool Dispatch)
A cognitive brain must **never** interrogate the outside world before checking its internal memory:
1. **Memory First**: Every inquiry (other than self-contained formal arithmetic) probes and traverses stored Tier 3 relational bindings and facts in MongoDB first.
2. **Deficit-Gated Acquisition**: External capabilities (such as `browser_access`) are triggered **only and strictly when memory yields an epistemic deficit** ($\Delta = 0$ matches found in internal memory).
3. **Closing the Loop**: When external acquisition occurs, acquired knowledge is parsed into Tier 1 tokens, Tier 2 candidates, and Tier 3 relational bindings and stored into memory. Subsequent queries for that entity or relation are immediately served as `KNOWN` directly from internal memory without external tool execution.

---

## 6. Verification & Audit Standards
- `python manage.py brain reason "..." -v` displays:
  `[Epistemic State]: State='...', Deficit='...', Target='...', Anchor='...'`
- Direct memory hits report `State='KNOWN'` with zero tool calls.
- Arithmetic expressions report `State='DERIVABLE'`.
- Missing external entities report `State='ACQUIRABLE'`.
- Fully decoupled from lexical syntax or keyword triggers.

