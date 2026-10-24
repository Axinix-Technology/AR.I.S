# Benchmark Dataset Ingestion & Supervised QA Training

## Architectural Purpose
AR.I.S operates on a multi-tier cognitive architecture where knowledge is grounded as symbolic relational bindings, graph topologies, and abstract operation templates rather than procedural code or static dictionaries.

To empower AR.I.S with broad domain awareness, relational link discovery, and multi-hop hypothesis generation, the system supports benchmark dataset ingestion and supervised QA demonstration training via:
1. **Knowledge Fact Ingestion (`training Dataset.json`)**: Ingests structured triples across 13 domains (astronomy, geography, computing, software, biology, people, family, organizations, education, etc.) directly through Tier 1 tokens, Tier 2 candidates, and Tier 3 relational bindings into MongoDB `brain_facts`.
2. **Supervised Question-Answer Training (`training question dataset.json`)**: Trains AR.I.S on 965 QA demonstrations:
   - **Reciprocal Supervision**: Induces reciprocal relationship rules (`brother_of` ↔ `sister_of`, `father_of` ↔ `son_of`, `mother_of` ↔ `son_of`) in `brain_relational_beliefs`.
   - **Operation Template Learning**: Discovers and reinforces abstract forward and reverse traversal operations (`op_forward_traversal`, `op_reverse_traversal`) in `brain_operation_templates`.
   - **Multi-Hop Traversal Learning**: Learns multi-hop graph traversal operations across 2 to 4 hops with full path provenance.
   - **Unsupported Connection Discrimination**: Validates that disconnected concepts (e.g. `Moon` ↔ `DNA`, `Tiger` ↔ `Paris`) are correctly marked `unconnected` / `0.0 confidence` without hallucinations.

---

## Core Directives Adherence

1. **Zero Hard Patches (Rule 1)**:
   No hardcoded relation dictionaries or entity lookup tables. Triple conversion to natural statements preserves structural entity detection:
   - Concepts and entities maintain proper referent casing.
   - Relation spans are discovered purely as the intervening token sequence between entity anchors.
2. **Generic Intelligence Over Procedural Orchestration (Rule 4)**:
   The ingestion engine leverages native `ArisBrain.store_fact`, `ArisCognitiveLearner.learn_demonstration`, and `ArisRelationalBeliefManager.learn_reciprocal_rule`. Reasoning is driven by dynamic graph BFS and GNN positional path scoring.
3. **Alphabetically Indexed Knowledge Brain (Rule 2 & 3)**:
   Continuously documented and indexed under **T** in `knowledge_brain/INDEX.md`.
4. **Human-Understandable Explanations (Rule 6)**:
   Every discovered path returns plain-language provenance (e.g., `Moon --[orbits]--> Earth --[orbits]--> Sun`).

---

## CLI & Interactive Execution

### Terminal Command:
```bash
python manage.py brain train --facts "training Dataset.json" --qa "training question dataset.json" --verbose
```

### Interactive Terminal:
```text
aris-brain> train
```

### Programmatic API:
```python
from api.brain import ArisBrain

# Train knowledge facts
facts_result = ArisBrain.train_dataset("training Dataset.json")

# Train supervised QA & reciprocal rules
qa_result = ArisBrain.train_qa_dataset("training question dataset.json")

# Evaluate QA reasoning accuracy
eval_result = ArisBrain.evaluate_qa_dataset("training question dataset.json")
```

---

## Benchmark Test Specifications

The dataset defines 10 core verification benchmarks for multi-hop graph connectivity:
1. `Moon` → `Sun`: CONNECTED (2 hops: Moon → Earth → Sun)
2. `Phobos` → `Sun`: CONNECTED (2 hops: Phobos → Mars → Sun)
3. `Chennai` → `Earth`: CONNECTED (1–3 hops: Chennai → Earth)
4. `Arun` → `Earth`: CONNECTED (2–4 hops: Arun → Chennai → Earth)
5. `WorkHub` → `Python`: CONNECTED (2 hops: WorkHub → Axinix/Django → Python)
6. `Gene` → `Cell`: CONNECTED (2 hops: Gene → DNA → Cell)
7. `Moon` → `DNA`: UNSUPPORTED (0.0 confidence, cycle-free rejection)
8. `Tiger` → `Paris`: UNSUPPORTED (0.0 confidence, disjoint components)
9. `Arun` → `Axinix`: CONNECTED (1 hop: Arun works at Axinix)
10. `IIT Madras` → `Asia`: CONNECTED (3–4 hops: IIT Madras → Chennai → Earth / Asia)
