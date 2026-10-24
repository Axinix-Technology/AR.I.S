# Dialogue Context Management & Dynamic Coreference (Phase 1)

**Classification**: Conversational Memory & Coreference Architecture  
**Module Path**: `api/cognitive/dialogue_context.py`, `api/cognitive/deictic_learner.py`, `api/cognitive/coreference_resolver.py`  
**Directives**: Strictly adheres to Rule 1 (Zero Hard Patches) and Rule 4 (Generic Cognitive Intelligence).

---

## 1. Overview & Architectural Motivation

To evolve AR.I.S toward frontier conversational intelligence (Claude/ChatGPT benchmark), the cognitive brain requires:
1. **Stateful multi-turn conversational context**: Tracking interaction turns, participant roles (`USER`, `AGENT`), and discourse history.
2. **Deictic Speaker Grounding**: Structurally mapping first-person and second-person communicative markers (e.g. English *"I/my"*, Tamil *"naan/ennudaiya"*, French *"je/mon"*, or alien tokens) into grounded participant entities without hardcoded keyword dictionaries.
3. **Dynamic Anaphora & Coreference Resolution**: Resolving pro-forms and anaphora (*"it"*, *"she"*, *"they"*, *"that project"*) to concrete antecedent entities across multiple turns using continuous prototype coordinate projections and recency-weighted salience ranking.

---

## 2. Core Components & Data Structures

### `ArisDeicticLearner` & Continuous Prototype Vectors
- Maintains continuous coordinate centroids for communicative roles in vector space (dimension = 64):
  - `ROLE_SPEAKER_SUBJECT`: 1st-person speaker subject role.
  - `ROLE_SPEAKER_POSSESSIVE`: 1st-person speaker possessive/genitive role.
  - `ROLE_ADDRESSEE`: 2nd-person addressee participant role.
  - `ROLE_ANAPHORA_INANIMATE`: Inanimate pro-form entity role.
  - `ROLE_ANAPHORA_ANIMATE`: Animate pro-form person role.
- Contains **ZERO hardcoded word lists or regex hacks**. Evaluates tokens dynamically via cosine similarity against learned prototype vectors.

### `ArisDialogueTurn`
Encapsulates single-turn metadata:
- `turn_id` (UUID)
- `turn_index` (int)
- `speaker_role` (`USER` | `AGENT`)
- `raw_text` (str)
- `tokens` & `extracted_entities`
- `relational_bindings`
- `resolved_coreferences`
- `epistemic_state` (`STORED`, `KNOWN`, `DERIVABLE`, `ACQUIRABLE`, `UNSUPPORTED`)
- `timestamp`

### `ArisDialogueContext` & Non-Bottleneck Sliding Window
- Configurable capacity (default $N=32$ turns) with dynamic working memory recency calculation:
  $$\text{Salience}(e) = \sum_{t \in \text{Turns}} w_{\text{mention}} \cdot \lambda^{\Delta t}$$
  where $\lambda = 0.85$ is the temporal recency decay factor and $\Delta t$ is distance from the current turn.

### `ArisCoreferenceResolver`
- Coordinates with `ArisDeicticLearner` to project tokens into prototype coordinate space.
- Substitutes anaphoric tokens with top-salience antecedent entities prior to relational reasoning.
- Grounds speaker identity bindings (`USER.name`) across turns.

---

## 3. End-to-End Interaction Flow

```
User: "My name is Alex."
  1. Relational Binder: (Subject: [Speaker_Marker], Predicate: name, Object: Alex)
  2. Deictic Learner: Marker projects onto ROLE_SPEAKER_POSSESSIVE (sim >= 0.50)
  3. Grounding: Context binds USER.name = "Alex"
  4. Response: "Understood. I have recorded that Alex name is Alex."

User: "I founded NovaTech."
  1. Deictic Learner: "I" projects onto ROLE_SPEAKER_SUBJECT -> resolves to "Alex"
  2. Relational Binder: (Subject: Alex, Predicate: founded, Object: NovaTech)
  3. Salience Graph: [NovaTech: 1.80, Alex: 1.53]
  4. Memory: Fact stored in MongoDB

User: "It is located in Berlin."
  1. Deictic Learner: "It" projects onto ROLE_ANAPHORA_INANIMATE
  2. Salience Resolver: Inanimate top candidate -> "NovaTech"
  3. Resolved Utterance: "NovaTech is located in Berlin."
  4. Memory: (Subject: NovaTech, Predicate: located_in, Object: Berlin) stored

User: "Where is NovaTech located?"
  1. Relational Retriever: (Subject: NovaTech, Predicate: located_in, Object: ?)
  2. Traversal: Returns "Berlin"
```

---

## 4. API Endpoints & Interfaces

- `POST /api/chat/`: Stateful multi-turn dialogue endpoint.
- `GET /api/chat/sessions/<session_id>/`: Retrieves context transcript and active salience graph.
- `DELETE /api/chat/sessions/<session_id>/`: Resets session context.
- `GET /chat/`: Interactive web interface with real-time cognitive trace and coreference inspection.
