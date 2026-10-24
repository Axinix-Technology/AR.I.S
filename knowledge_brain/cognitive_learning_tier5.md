# Cognitive Learning & Operation Generalization (Tier 5)

## Overview

Tier 5 is the **Relational Operation Learning & Autonomous Generalization Engine** of AR.I.S.
It establishes the transition from programmed data retrieval to autonomous cognitive learning:

```text
Human Language Expression
         │
         ▼
Tier 1–3 Structural Representation (FROZEN ✅)
         │
         ▼
Tier 5: Learned Operational Query Intent (CURRENT)
         │
         ▼
Abstract Relational Operation (De-lexicalized Traversal Schema)
         │
         ▼
Tier 4 Deterministic Traversal Engine (FROZEN ✅)
         │
         ▼
Final Ground-Truth Answer
```

---

## The Core Principle: Generalization to Unseen Situations

> **"Training data is not the target. Training data is the teaching signal. The target is generalization to structurally valid but lexically unseen situations."**

### Why Associative Synonym Tables Were Banned
Building a key-value dictionary of synonyms (`leads ↔ heads`, `brother ↔ sibling`) is not learning—it is merely hardcoding a vocabulary table by another name. Tier 5 does **not** store vocabulary. It discovers the **abstract relational operation** (e.g. `reverse_traversal`, `forward_traversal`) described by language.

---

## 4-Level Generalization Hierarchy

| Level | Benchmark Type | Teaching Signal | Unseen Test Situation | Result |
|---|---|---|---|---|
| **Level 1** | Baseline Sanity | `Arun is brother of Bala.` $\to$ `Arun` | Re-evaluate teaching signal | Passes |
| **Level 2** | Unseen Entities | `Arun is brother of Bala.` $\to$ `Arun` | Memory: `Maya is brother of Qorin.`<br>Query: `Who is brother of Qorin?` | Emits `Maya` (Zero entity memorization) |
| **Level 3** | Unseen Alien Vocabulary | `Arun flarps Zeta.` $\to$ `Arun` | Memory: `Qorin daxels Veltra.`<br>Query: `Who daxels Veltra?` | Emits `Qorin` (Zero dictionary dependency) |
| **Level 4** | Unseen Composition | Multi-hop pattern | Multi-hop chain across unseen alien entities | Emits target entity |

---

## Modules

### 1. `api/cognitive_learner.py`
- **`ArisOperationTemplate`**: Stores de-lexicalized operation schema:
  - `operation_type`: `reverse_traversal` / `forward_traversal`
  - `source_role`: position of variable in question (`subject` or `object`)
  - `target_role`: position of answer in fact (`subject` or `object`)
  - `anchor_role`: position of shared constraint anchor
- **`ArisCognitiveLearner`**:
  - `learn_demonstration(fact, question, answer)`: Aligns answer slot in fact against question binding, de-lexicalizes the demonstration, and registers abstract operation template.
- **`ArisCognitiveReasoner`**:
  - `reason(question_text, facts=None, templates=None)`: Evaluates unseen question against learned operational templates and executes deterministic Tier 4 traversal over memory.

### 2. Terminal CLI Commands
- `python manage.py brain learn --fact "..." --question "..." --answer "..." -v`:
  Displays the **Operation Abstraction Microscope**.
- `python manage.py brain reason "..." -v`:
  Executes autonomous reasoning on unseen entities and displays reasoning trace.
