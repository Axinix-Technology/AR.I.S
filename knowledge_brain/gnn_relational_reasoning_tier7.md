# GNN Relational Path Reasoning & Human Feedback Loop (Tier 7)

## 1. Architectural Purpose & Overview

The **Tier 7 GNN Relational Reasoning & Human Feedback Engine** provides AR.I.S with generic, non-procedural multi-hop hypothesis formation, path provenance derivation, and human-in-the-loop belief revision.

Unlike black-box LLMs or pre-trained Python machine learning models (e.g. BERT, spaCy, or transformers) that require expensive training pipelines and opaque parameter matrices, AR.I.S performs **dynamic cognitive reasoning directly over structured memory**:
1. It forms **plausible hypotheses** ("plausible guesses") across $1 \dots N$ hops when no direct 1-hop fact is recorded.
2. It outputs the exact **path provenance** (`Abdul Kalam --[born in]--> Rameswaram --[in]--> Tamil Nadu`).
3. It listens to **human feedback**: when the user denies a connection ("Abdul Kalam was born in Kerala? No"), AR.I.S persists an **inhibition record** setting confidence to `0.0`, permanently silencing that refuted assertion.
4. When the user provides a **correction** ("Dhivya is not brother of Arun; Dhivya is sister of Arun"), AR.I.S dynamically induces reciprocal and symmetric relational rules.

---

## 2. The 4 Pillars Architecture

### Pillar 1: Semantic Representation (`api/semantic_representation.py`)
- Maps candidates and spans to 64-dimensional geometric coordinate vectors using deterministic sub-token n-gram projections.
- Provides cosine similarity for semantic alignment without external API calls or neural dependencies.

### Pillar 2: Relational Binding Substrate (`api/relational.py`)
- Preserved foundational Tier 3 representation.
- Structurally extracts directional triples $(E_i, \text{relation span}, E_{i+1})$ using exact source character boundaries.
- **Zero hard patches**: no dictionary lookups or English part-of-speech parsers.

### Pillar 3: Symbolic Graph Reasoning (`api/symbolic_graph.py`)
- Instantiates an in-memory graph (`ArisSymbolicGraph`) constructed directly from MongoDB relational bindings.
- Uses cycle-free Breadth-First Search (BFS) up to $N$ hops (default max depth: 4).
- Formats path provenance strings into clear, human-understandable audit trails:
  `Subject --[relation_1]--> Intermediate --[relation_2]--> Target`.

### Pillar 4: GNN Positional Path Reasoner & Belief Revision (`api/gnn_reasoner.py` & `api/relational_beliefs.py`)
- **Positional Decay Scoring**: Multi-hop path confidence decays inversely with distance:
  $$\text{Confidence} = \frac{\prod_{i=1}^k c_i}{k}$$
  where $k$ is the hop count and $c_i$ is edge confidence.
- **Hypothesis Statuses**:
  - `plausible_hypothesis`: Connecting path found across $\ge 2$ hops.
  - `verified_symmetric`: Single reverse hop verified via learned symmetric rule.
  - `denied`: Assertion inhibited by explicit user denial.
  - `unconnected`: No graph paths link the entities.
- **Human Belief Revision**:
  - Collection: `brain_relational_beliefs` (denials, confirmations).
  - Collection: `brain_relational_rules` (reciprocal relations, symmetric relations).
  - Auxiliary normalization: Strips auxiliary prefixes (`is`, `was`, `are`, `were`) so `was born in` and `born in` share the exact same belief key.

---

## 3. Terminal & Interactive Commands

| Command | Syntax | Action |
| :--- | :--- | :--- |
| `reason` | `python manage.py brain reason "<query>"` | Evaluates direct memory, epistemic deficit, or forms GNN plausible hypothesis. |
| `hypothesize` | `python manage.py brain hypothesize "<subject>" "<target>" [relation]` | Probes graph connectivity and outputs path provenance. |
| `feedback` | `python manage.py brain feedback "<subj>" "<rel>" "<obj>" --action deny [--correction "..."]` | Updates belief matrix and induces relational rules. |
| Interactive | In `interactive` shell: `reason ...`, `deny <subj> <rel> <obj>`, `confirm ...` | Live human feedback loop. |

---

## 4. Verification Standards

1. **Domain Agnosticism**: Tested across family relations, geography (`Abdul Kalam`), and alien vocabularies (`X qevron Y`, `Y daxels Z`).
2. **Cognitive Integrity**: Zero hard patches (`VAR_MARKERS` banned).
3. **Audit Verification**:
   - `python manage.py test api.tests.FourPillarsGNNReasoningTier7Tests`
