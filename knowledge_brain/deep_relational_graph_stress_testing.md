# Deep Relational Graph Stress Testing (Phase 17.2)

## Architectural Purpose
Evaluates AR.I.S cognitive reasoning capacity under extreme multi-hop relational complexity:
- **Depth**: Traversal across 6 to 8 hops in complex knowledge graphs.
- **Branching Factor**: 5 distinct branching paths intersecting at multiple topological tiers.
- **Negative & Distractor Resistance**: Dead-end distractor branches (wind tunnel, environmental registry) and competing branch targets (facility, person, country).
- **Question-Conditioned Discriminability**: Differentiating 8-hop leaf targets from intermediate 1-hop or 3-hop targets without overshooting or leaf bias.
- **Epistemic Support Invariant**: Rejecting absent relations with zero hallucination even in densely connected graphs.

---

## Benchmark Scenario: "Project Chimera — Transcontinental Quantum Defense & Power Grid"

### Graph Nodes
- **`Dr. Elena Rostova`** (PERSON) — Anchor origin
- **`CryoCore Propulsion`** (PRODUCT_TECHNOLOGY) — Hop 1
- **`AeroSpace Dynamics`** (ORGANIZATION) — Hop 2
- **`Toulouse Advanced Hangar`** (FACILITY) — Hop 3
- **`South France Grid`** (FACILITY) — Hop 4
- **`TransEuropean Power Authority`** (ORGANIZATION) — Hop 5
- **`Brussels Regulatory Commission`** (ORGANIZATION) — Hop 6
- **`Belgian Federal Tribunal`** (ORGANIZATION) — Hop 7
- **`Chief Justice Henri Laurent`** (PERSON) — Hop 8 Target

### Competing Branches & Distractors
- **Branch 2 (Component Supplier to Quantum Lab)**:
  - `AeroSpace Dynamics` $\to$ `Valence Microelectronics` $\to$ `Lyon Data Center` $\to$ `Geneva Quantum Lab` $\to$ `Dr. Martin Keller` (6 hops)
- **Branch 3 (Country Target Discrimination)**:
  - `TransEuropean Power Authority` $\to$ `Brussels Central District` $\to$ `Belgium` (7 hops, `GPE_COUNTRY`)
- **Dead-End 1**: `CryoCore Propulsion` $\to$ `Bordeaux Wind Tunnel` (Hop 2 dead-end)
- **Dead-End 2**: `Brussels Regulatory Commission` $\to$ `EU Environmental Registry` (Hop 7 dead-end)

---

## Test Invariants

| Test Identifier | Question Focus | Expected Target | Hops | Key Invariant Checked |
|---|---|---|---|---|
| `test_01_8_hop_backbone_connectivity` | Topology verification | `Chief Justice Henri Laurent` | 8 | Graph discovers and constructs full 8-hop path |
| `test_02_8_hop_deep_relational_reasoning` | Deep goal reasoning | `Chief Justice Henri Laurent` | 8 | Discovers 8-hop target through 7 intermediate organizations |
| `test_03_6_hop_alternative_branch` | Branch discrimination | `Dr. Martin Keller` | 6 | Selects Branch 2 over 8-hop primary backbone |
| `test_04_intermediate_node_stopping` | Zero overshoot | `Toulouse Advanced Hangar` | 3 | Intermediate query stops at exact goal without overshooting to leaf |
| `test_05_target_type_discrimination` | Type manifold discrimination | `Belgium` | 7 | Selects `GPE_COUNTRY` target rather than person/org nodes |
| `test_06_dead_end_distractor_rejection` | Distractor rejection | `AeroSpace Dynamics` | 2 | Disregards dead-end test facility |
| `test_07_unsupported_query_epistemic` | Epistemic state | `None` (`matches_count = 0`) | N/A | Returns 0 matches for unasserted relation |

---

## Architectural Rules Compliance
- **Rule 1 (Zero Hard Patches)**: Tested via learned goal compatibility, relational path scoring, and continuous metric inference.
- **Rule 4 (Generic Intelligence)**: No node depth limits or rigid procedural routing; topology is explored neutrally by BFS and ranked by whole-path compatibility $S_\theta(c, p, G)$.

---

## Test Execution Commands

The stress test suite can be run seamlessly across all test harnesses:

1. **Django Test Runner**:
   ```bash
   python manage.py test api.tests.test_deep_graph_stress_reasoning -v 2
   ```

2. **Pytest Engine**:
   ```bash
   pytest api/tests/test_deep_graph_stress_reasoning.py -v
   ```

3. **Direct Python Standalone**:
   ```bash
   python api/tests/test_deep_graph_stress_reasoning.py
   ```

---

## Multi-Anchor Learned Reasoning & Manifold Architecture

### 1. Multi-Anchor Candidate Space (Zero Procedural Selection)
Queries with complex multi-clause dependencies contain multiple grounded entities (e.g., *"Who is the manager of Geneva Quantum Lab interconnected with AeroSpace Dynamics?"*). Instead of Python procedural heuristics selecting a single anchor (subject vs. object):
- All grounded query entities are collected as candidate exploration origins.
- The open-variable graph search explores plausible relational pathways originating from all candidate anchors.
- Every discovered candidate tuple $(G, a, p, c)$ (Goal, Anchor, Path, Candidate Entity) is projected into the joint tensor evaluation engine.
- Candidate selection is 100% learned and goal-conditioned: $S_\theta(c, p, G)$ evaluates predicate alignment, semantic target type matching, and topological decay to score and rank candidates.

### 2. Target Semantic Type Manifold Expansion (Subspace C)
- Added `FACILITY` to `data/goal_conditioned_relational_training.json` with domain context indicators (`hangar`, `facility`, `center`, `station`, `complex`, `site`) and incoming relations (`operates out of`, `receives power from`, `hosts servers at`).
- Integrated dynamic compilation check into `ArisGoalReasoningTensorEngine._ensure_loaded()` to automatically compile missing target type manifolds on load.

