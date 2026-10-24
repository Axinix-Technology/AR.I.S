# Universal Dependencies Interrogative Target Discrimination

## Overview
Interrogative question utterances in natural language specify their target answer domain either via:
1. **Pronominal Interrogatives**: (e.g., *"Who presides...?"*, *"Where is...?"*) where the interrogative pronoun inherently denotes the broad domain (`Who` $\to$ `PERSON`, `Where` $\to$ `LOCATION`).
2. **Determiner-Governed Nominal Interrogatives**: (e.g., *"Which country is...?"*, *"Which company integrated...?"*, *"What facility does...?"*) where the interrogative determiner (`Which`, `What`) modifies and syntactically governs a nominal head (`country`, `company`, `facility`).

Under **Rule 1 (Zero Hard Patches)**, **Rule 4 (Generic Intelligence)**, and **Rule 7 (Prohibition of Surface Guards)**:
- Hardcoded entity names (e.g. `country = "india"` or `"belgium"`) are strictly forbidden.
- Sentence-level n-gram guessing that dilutes the syntactic head with declarative noise is eliminated.
- Target type prediction is grounded directly in the **Universal Dependencies (UD v2.18) syntactic dependency structure**.

---

## Architectural Mechanism

### 1. Syntactic Governor Discovery
In `ArisGoalFormer.form_goal()`:
1. The opening token is projected continuously onto the pragmatic interrogative manifold (`_interrogative_matrices`).
2. If the opening token exhibits interrogative illocutionary force ($\text{similarity} \ge 0.60$) and governs an immediate nominal head (`tokens[1].upos in ("NOUN", "PROPN")`):
   $$\text{Which} \xrightarrow{\text{UD DET}} \text{syntactic head: } \mathbf{country} \xrightarrow{\text{Subspace C}} \text{target: } \mathbf{GPE\_COUNTRY}$$
3. The syntactic head noun (`country`) is passed directly to `predict_target_type()`, eliminating whole-sentence noise dilution.

### 2. Subspace C Geopolitical Manifold Hierarchy
In `data/goal_conditioned_relational_training.json` and `models/aris_goal_reasoning_tensor_bank_v1.pt`:
- **`GPE_COUNTRY`** models top-level sovereign container entities that cities and districts are `"situated in"`, `"located in country"`, `"sovereign territory of"`, with generic geopolitical indicators (`"country"`, `"nation"`, `"republic"`, `"kingdom"`, `"federation"`, `"commonwealth"`, `"sovereign"`, `"principality"`, `"realm"`).
- **`GPE_STATE`** models subnational internal divisions reached by administrative relations (`"in state"`, `"is in province"`, `"province of"`, `"canton of"`).
- This relational distinction cleanly differentiates sovereign countries from provinces across any geographic entity (e.g., `Belgium`, `Switzerland`, `France`) with zero hardcoded country names.

---

## Exact-Line Audit Record
- **Files**:
  - `api/cognitive/epistemic_state.py`: Lines 164–188 (UD syntactic interrogative governor extraction).
  - `data/goal_conditioned_relational_training.json`: Lines 360–415 (`GPE_STATE` and `GPE_COUNTRY` generic relational and sovereign manifold alignment).
  - `models/aris_goal_reasoning_tensor_bank_v1.pt`: Recompiled from training data.
- **Verification**: `test_05_target_type_discrimination_country_vs_person` PASSED.
