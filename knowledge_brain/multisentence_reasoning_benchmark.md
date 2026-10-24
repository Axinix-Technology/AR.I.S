# Multi-Sentence Complex Reasoning & Target Discrimination Benchmark

## Overview
The **Multi-Sentence Complex Reasoning & Target Discrimination Benchmark** (`data/multisentence_complex_reasoning_dataset.json`) is a high-difficulty cognitive training and evaluation dataset for AR.I.S. It addresses the frontier cognitive challenge: transitioning from unconstrained path reachability to **semantic goal satisfaction and answer-type discrimination**.

---

## Architectural Objectives (Zero Hard Patches)
1. **Semantic Goal Satisfaction Over Pure Reachability**:
   - In graph exploration, blind traversal often terminates at the farthest leaf node (e.g. returning a city when the user asked for a company).
   - This dataset models exact semantic type matching, ensuring the reasoner identifies which intermediate or terminal node satisfies the query.
2. **Answer-Type & Target Discrimination**:
   - Entities are annotated with semantic categories (`ORGANIZATION`, `PERSON`, `GPE_CITY`, `GPE_STATE`, `GPE_COUNTRY`, `PRODUCT_TECHNOLOGY`, `AWARD_HONOR`).
   - Questions evaluate the reasoner's capacity to discriminate between target categories along identical traversal chains.
3. **Derivational & Morphological Relation Matching**:
   - Pairs noun-verb derivational variants (e.g. noun `founder` $\leftrightarrow$ verb `founded`, `manager` $\leftrightarrow$ `manages`, `sponsor` $\leftrightarrow$ `sponsored by`, `commanded` $\leftrightarrow$ `captains`) to teach semantic alignment beyond raw character substring containment.
4. **Multi-Sentence Discourse & Non-Direct Hops**:
   - 3 to 6 connected sentences per scenario introducing multiple entities, coreferences, cross-sentence dependencies, and distractor paths.
   - Evaluates multi-hop traversals spanning 1 to 4 hops with explicit distractor nodes that must be bypassed.

---

## Dataset Schema
* `narrative_context`: Connected multi-sentence text.
* `entities`: Entity spans with semantic categories and discourse roles.
* `formal_triples`: Ground-truth relational graph bindings.
* `qa_pairs`:
  * `question`: Natural language inquiry.
  * `target_type`: Expected semantic entity category.
  * `expected_answer`: Ground truth entity string.
  * `reasoning_path`: Complete multi-hop chain.
  * `hop_count`: Traversal depth.
  * `derivational_morphology`: Predicate pairs tested.
  * `distractors`: Reachable nodes that must be rejected.
