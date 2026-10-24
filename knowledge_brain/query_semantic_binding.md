# Question → Semantic Query Representation & Binding Primitive

## Overview
The **Question → Semantic Query Binding** cognitive layer transforms natural-language questions into canonical structured semantic queries without hardcoded dictionaries or special-case bypasses:

$$\text{Query}(\text{subject}=\text{entity} \mid ?, \; \text{relation}=\text{canonical\_rel}, \; \text{object}=\text{entity} \mid ?, \; \text{answer\_slot}=\text{subject} \mid \text{object} \mid \text{boolean})$$

This isolates and resolves the primary bottleneck where interrogative natural-language queries were previously confused with declarative statements.

---

## Strict Non-Negotiable Directives (Rules 1 & 4)

1. **Zero Hardcoded Word Lists**:
   - Strictly NO static `WH_OPERATORS` or `AUXILIARIES` lists.
   - Sentence-initial single capitalized words are categorized as `ambiguous_capitalized` by `ArisTokenizer`. In questions, sentence position 0 functions as an interrogative/modal clause operator, not an entity candidate.
2. **Pure Positional Invariants for Slot Assignment**:
   Variable slot detection is determined by entity arity and relative structural position:
   - **2 Bound Entities** (e.g. *"Does Arun work at Axinix?"*):
     $$\text{Subject} = \text{Arun}, \quad \text{Object} = \text{Axinix}, \quad \text{answer\_slot} = \text{boolean}$$
   - **1 Bound Entity**:
     - **Entity precedes predicate tokens** (e.g. *"Where does Arun work?"*, *"Where is Axinix located?"*):
       $$\text{Entity is Subject} \implies \text{Subject} = \text{Arun}, \quad \text{Object} = ?, \quad \text{answer\_slot} = \text{object}$$
     - **Entity follows predicate tokens** (e.g. *"Who works at Axinix?"*, *"Who is the brother of Dhivya?"*):
       $$\text{Entity is Object} \implies \text{Subject} = ?, \quad \text{Object} = \text{Axinix}, \quad \text{answer\_slot} = \text{subject}$$
3. **Decoupled Architecture**:
   $$\text{INPUT} \to \text{TOKENIZATION} \to \text{ENTITY CANDIDATES} \to \text{QUESTION STRUCTURE} \to \text{SEMANTIC QUERY} \to \text{GRAPH SEARCH} \to \text{ANSWER}$$
   Graph search never silently repairs an incorrect semantic query. If the query representation is flawed, it fails at the query stage.
4. **Coordinate Space Canonicalization**:
   Extracted surface predicate tokens are projected into continuous 64-dimensional coordinate space using `ArisSemanticEncoder`, computing cosine similarity against canonical relations registered in the Brain memory.

---

## Empirical Benchmark Results (`aris_query_binding_eval.json`)

Evaluated across 100 benchmark question instances in `aris_query_binding_dataset.json`:

| Cognitive Stage | Accuracy | Note |
| :--- | :---: | :--- |
| **Entity Extraction** | **100.0%** | Zero false entity pollution from Wh-words or auxiliaries |
| **Question Type** | **100.0%** | Morphological clause operator detection |
| **Relation Extraction** | **95.0%** | Non-lexical predicate span extraction |
| **Canonical Relation** | **58.0%** | Continuous coordinate similarity mapping |
| **Subject Binding** | **98.0%** | Positional invariant mapping |
| **Object Binding** | **91.0%** | Positional invariant mapping |
| **Answer Slot** | **91.0%** | Correct slot assignment (subject / object / boolean) |
| **Semantic Query Accuracy** | **56.0%** | All 4 components simultaneously exact |
| **Graph Path Accuracy** | **3.0%** | Pure graph reachability over active facts |
| **Final Answer Accuracy** | **33.0%** | Grounded retrieval matching ground truth |

---

## Failure Distribution

| Failure Category | Count | Diagnostic Rationale |
| :--- | :---: | :--- |
| **`ENTITY_FAILURE`** | **0** | Perfect entity extraction without word blacklists |
| **`DIRECTION_FAILURE`** | **0** | Structural positional invariant resolved direction inversion |
| **`QUESTION_STRUCTURE_FAILURE`** | **0** | 100% question format recognition |
| **`ANSWER_SLOT_FAILURE`** | **9** | Complex edge cases with possessive or multi-modifier syntax |
| **`RELATION_FAILURE`** | **35** | Distant surface synonyms (e.g. *satellite of* vs *orbits*, *continent located in*) requiring learned relational prototypes |

---

## Management CLI Command
To run the benchmark:
```powershell
python manage.py brain evaluate_query_binding --dataset aris_query_binding_dataset.json --facts aris_training_facts.json --output aris_query_binding_eval.json
```
