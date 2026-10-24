# Discourse-Aware Category Reasoning & Autonomous Tool Chat

## Overview & Constitutional Intent
In human-agent multi-turn dialogue, understanding user queries relies on **universal syntactic clause structure** across all grammatical categories (`PRON`, `AUX`, `DET`, `ADJ`, `NOUN`, `VERB`, `PROPN`, etc.) rather than manually hardcoded dictionaries or keyword lists (Rules 1 & 4). 

When a user asks a follow-up query seeking category members (e.g., *"What are the other companies he leads?"*), the cognitive engine must respect **discourse salience and conversational pragmatics**: entities established as the topic of preceding turns (e.g. `SpaceX`) must not be redundantly repeated when the query asks about remaining category members.

---

## Key Principles & Mechanisms

### 1. Zero Word Dictionaries (Anti-Glossary Invariant)
- **Constraint**: AR.I.S strictly forbids building handbooks, word lists, or hardcoded synonym dictionaries (e.g., manual lists of differential modifiers).
- **Generic Solution**: Language understanding is grounded in the universal Universal Dependencies (UD) tensor manifold (`models/aris_ud_tensor_bank.pt`). The model identifies syntactic constituent roles:
  - Subject Argument: `Elon Musk` (resolved from pro-form `he`)
  - Predicate Relation: `leads` (`VERB`)
  - Target Nominal Category: `companies` (`NOUN`)
  - Modifiers / Restrictors: `other` (`ADJ` / `amod`)

### 2. Dialogue Context Discourse Salience
- **Mechanism**: The reasoner evaluates the active conversation context (`ArisDialogueContextManager`).
- **Exclusion of Grounded Topics**: Entities that were the central focus of prior turns (e.g., `SpaceX` in Turn 2) are placed into `excluded_entities`.
- **Reasoning**: In `ArisGNNReasoner.evaluate_open_variable_hypothesis`, candidate nodes matching `excluded_entities` are skipped, revealing the unmentioned category members (`Tesla`, `Neuralink`, `Twitter`).

### 3. Terminal-Configurable Offline Advisor
- **Mechanism**: In `ArisBrain.process_dialogue_turn()` and `ArisCognitiveReasoner.reason()`, `use_advisor` is controlled via the environment variable `ARIS_USE_ADVISOR` (defaulting to `false` for offline dialogue turns, `use_neural=true`).
- **Result**: Guarantees that turns querying internal memory make exactly **0 external HTTP calls**, while remaining fully configurable from the terminal for ablation studies or online modes.

---

## Architectural Verification Criteria
1. **Zero Dictionaries**: No hardcoded lists of words in training JSON or python modules.
2. **0 HTTP Calls on Memory Retrieval**: Offline turns querying stored knowledge must execute purely locally via neural and symbolic graph memory.
3. **Conversational Multi-Entity Answers**: Answers containing multiple valid hypotheses are realized in natural natural language (e.g., *"Elon Musk leads Tesla, Neuralink, and Twitter."*).
