# Open-Variable Relational Reasoning & Graph Hypothesis Discovery

## Overview
Open-variable relational reasoning enables AR.I.S to resolve natural language questions where one entity slot is grounded (the anchor entity) while the other is an open query variable (e.g. *who*, *where*, *what*). Unlike concrete relational assertions (which feature pairwise candidate bindings), open-variable queries often have empty binary candidate bindings and require graph hypothesis discovery across 1-hop and multi-hop associative networks.

---

## Architectural Principles (Zero Hard Patches)
1. **Generic Intelligence Over Code-Based Orchestration**:
   - No hardcoded string checks (`"who is"`, `"partner of"`, `"company of"`, etc.).
   - No positional or character-slicing fallbacks (`t0["raw"]`, `[char_start:char_end]`).
   - No hardcoded entity names (`"Arunbharathi"`, `"Alice"`).
2. **Unified Linguistic Substrate**:
   - Nominal entities (including lowercase proper nouns like `"arunbharathi"`) are extracted via Universal Dependencies UPOS tensor classification (`PROPN`/`NOUN`), informed by surrounding lexical and punctuation context.
3. **Structured Goal Formulation**:
   - `ArisGoalFormer` perceives structural cues and constructs an `ArisGoal`:
     $$\text{Goal} = \langle \text{anchor\_entity}, \text{target\_variable}, \text{relational\_constraint} \rangle$$
4. **Graph-Driven Positional Hypothesis Discovery**:
   - When pairwise relational bindings are absent, the reasoning engine does NOT prematurely abort.
   - It invokes `ArisGNNReasoner.evaluate_open_variable_hypothesis()` to traverse the memory graph starting from the grounded anchor entity.
   - Computes weighted semantic overlap and subword continuity between relation tokens and path edges.
   - Distinguishes terminal leaf entities from intermediate nodes using graph topology.
   - Applies GNN positional decay over multi-hop paths to assign grounded plausibility confidence with full step provenance.

---

## Verification
- `test_open_variable_partner_reasoning`: Resolves lowercase/spelling-variant relational question (`"who is patner of arunbharathi?"`) to `"Pradeepa"` across coreference and relational edges.
- `test_open_variable_transitive_company_location`: Resolves multi-hop transitive relation (`"where does company of Alice operate?"`) from `"Alice"` through `"DeepMind"` to `"London"`.
