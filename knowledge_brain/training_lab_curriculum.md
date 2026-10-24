# Generalization Training Lab & 14-Tier Cognitive Curriculum

## 1. Overview & Purpose
The **AR.I.S Generalization Training Lab** (`/training-lab/`) provides a continuous, automated benchmark for evaluating the zero-shot and few-shot generalization capabilities of AR.I.S across 14 distinct curriculum levels.

### Key Distinction: Seen Retention vs. Out-of-Distribution Generalization
In accordance with Rule 4 (Generic Cognitive Intelligence):
> *"Training data is not the target. Training data is the teaching signal. The target is generalization to structurally valid but lexically unseen situations."*

The lab measures two orthogonal metrics:
1. **Seen Knowledge Retention (In-Distribution)**: Recall and direct lookup over explicitly seeded facts.
2. **Zero-Shot Transfer (Out-of-Distribution)**: Structural reasoning over novel entities, unseen predicates, multi-hop transitive paths, algebraic unknowns, and noise distractors.

---

## 2. 14-Tier Curriculum Specification

| Tier | Name | Target Cognitive Capability |
|---|---|---|
| **Level 1** | Direct Fact Retrieval (1-Hop Seen) | Exact relational recall from sensory memory |
| **Level 2** | Unseen Entity Zero-Shot Transfer | Generalization to novel proper names never encountered |
| **Level 3** | Unseen Relation Induction | Dynamic binding over novel predicate verbs |
| **Level 4** | Reverse Directional Reasoning | Implicit inverse relation traversal without duplication |
| **Level 5** | 2-Hop Transitive Deduction | Graph traversal over intermediate bridge nodes |
| **Level 6** | 3-Hop Deep Relational Chain | Deep multi-step deductive reasoning |
| **Level 7** | Structural Variable Parameter Binding | Binding abstract symbolic variables into tool arguments |
| **Level 8** | Deterministic Arithmetic Substrate | Safe AST arithmetic computation across operators |
| **Level 9** | Algebraic Linear Equation Solving | Universal affine root finding ($v = -c/m$) |
| **Level 10** | Positional & Spatial Distance | Coordinate metrics across 1D/2D/N-D embeddings |
| **Level 11** | Composite Relational + Math Inference | Extracting numeric facts from memory graph and executing math |
| **Level 12** | Noise & Distractor Robustness | Maintaining entity binding amid conversational filler words |
| **Level 13** | Epistemic Boundary Rejection | Refusing ungrounded/unsupported queries without hallucination |
| **Level 14** | Conflict Resolution & Temporal Overwrite | Belief revision when past facts are superseded |

---

## 3. Architecture & Endpoints
* **Web Interface**: `/training-lab/` (renders `aris_training_lab.html` with real-time SVG generalization curve and level cards).
* **REST Execution Gateway**: `POST /api/training-lab/run/` (executes `ArisTrainingLab.run_benchmark()` and returns JSON summary).
* **Isolation Guarantee**: Each curriculum level runs in an isolated cognitive memory scope to prevent cross-level data leakage.
