# AR.I.S — Capability & Cognitive Structure Specification

> **Document Name**: `ARIS_CAPABILITY_AND_COGNITIVE_STRUCTURE.md`  
> **Repository**: `e:/Loigmax/AR.I.S`  
> **Master Index**: [`knowledge_brain/INDEX.md`](file:///e:/Loigmax/AR.I.S/knowledge_brain/INDEX.md)  
> **Purpose**: Serves as the authoritative, ground-truth architectural extraction of all capabilities, epistemic states, cognitive operations, and tensor manifolds in AR.I.S. This document will later be used to construct a PyTorch (`.pt`) training dataset. (No `.pt` generation is performed in this document).

---

## 1. System Identity

### 1.1 Purpose of AR.I.S
**AR.I.S (Artificial Intelligent System)** is an autonomous cognitive system engineered to perceive natural language, form structured open goals, assess internal epistemic states, bind abstract parameter slots, execute deterministic tools, and traverse relational knowledge graphs without hardcoded keywords, regex heuristics, or brittle orchestration pipelines.

### 1.2 Cognitive Architecture & Execution Pipeline
The AR.I.S cognitive substrate functions as a closed-loop inference cycle:
$$\text{Sensory Input} \longrightarrow \text{Tiers 1–3 Linguistic Parsing} \longrightarrow \text{Goal Formation} \longrightarrow \text{Epistemic State Evaluation} \longrightarrow \text{Capability Affordance Perception} \longrightarrow \text{Action / Tool Dispatch} \longrightarrow \text{Relational Memory Integration} \longrightarrow \text{NLG Synthesis}$$

The concrete pipeline comprises 8 observable stages:
1. **Linguistic Tokenization & Span Preservation**: Extracts tokens, atomic path/number literals, and character offsets (`api/linguistic/tokenizer.py`).
2. **Entity & Concept Candidate Induction**: Identifies multi-token nominal spans and capitalized candidate entities without fixed dictionaries (`api/linguistic/pipeline.py`).
3. **Syntactic Frame Normalization & UPOS Projection**: Maps mood, voice, and Universal Dependencies syntactic roles (`api/linguistic/syntactic_normalizer.py`, `api/linguistic/ud_tensor_engine.py`).
4. **Relational Binding**: Discovers directed `(Subject, Relation-Span, Object)` triples (`api/relational/relational.py`).
5. **Epistemic State Evaluation**: Probes relational memory and evaluates knowledge gaps (`KNOWN`, `DERIVABLE`, `ACQUIRABLE`, `UNSUPPORTED`) (`api/cognitive/epistemic_state.py`).
6. **Affordance Perception & Multi-Tool Planning**: Evaluates all registered capabilities in `ArisCapabilityRegistry` against the epistemic deficit via metric tensor projections (`api/semantic/executable_tools.py`, `api/cognitive/affordance_tensor_engine.py`, `api/cognitive/deliberative_planner.py`).
7. **Deterministic Execution / Graph Traversal**: Dispatches bound parameters to deterministic tools or executes multi-hop traversal over `ArisSymbolicGraph` and `ArisGnnHypothesisReasoner`.
8. **Neural-Symbolic Generative Decoding**: Renders grounded, human-grade conversational responses conditioned on provenance and epistemic state (`api/cognitive/generative_decoder.py`).

### 1.3 Major Cognitive Layers / Tiers
* **Layer 0 (Sensory Structural Advisor)**: Optional external sensory teacher providing parameter frames and query variable observations without accessing memory (`api/relational/structural_advisor.py`).
* **Tier 1 (Token & Positional Representation Primitive)**: Exact character offset preservation, atomic path/number token typing (`api/linguistic/tokenizer.py`).
* **Tier 2 (Concept & Entity Candidate Layer)**: Multi-token proper-noun spans and compound noun phrases (`api/linguistic/pipeline.py`).
* **Tier 3 (Relational Binding Primitive)**: Sentence-local triple binding without dictionaries (`api/relational/relational.py`).
* **Tier 4 (Deterministic Relational Retrieval)**: 1-hop forward/reverse traversal over stored fact graphs (`api/relational/retrieval.py`).
* **Tier 5 (Relational Operation Learning & Generalization)**: De-lexicalized operational templates (`ArisOperationTemplate`) generalizing across unseen entities and alien vocabulary (`api/cognitive/cognitive_learner.py`).
* **Tier 6-A (Epistemic Cognition & Dynamic Capability Selection)**: Deficit-gated capability matching based on structural contracts (`api/cognitive/epistemic_state.py`).
* **Tier 6-B (Multi-Hop Structural Decomposition & Graph Grounding)**: Multi-hop graph exploration without procedural loops (`api/relational/symbolic_graph.py`).
* **Tier 7 (GNN Relational Path Reasoning & Belief Revision)**: Multi-hop topological hypothesis formation, reciprocal inference, and human-in-the-loop belief updates (`api/relational/gnn_reasoner.py`, `api/cognitive/relational_beliefs.py`).

### 1.4 Generic Intelligence Rules
1. **Rule 1 (Zero Hard Patches)**: Address root causes at the deepest structural level. Never mask symptoms with temporary bypasses.
2. **Rule 2 (Living Knowledge Brain with Alphabetical Indexing)**: Continuously document all architectural decisions in `knowledge_brain/` with strict alphabetical indexing in `INDEX.md`.
3. **Rule 3 (Knowledge Brain Pre-Verification)**: Inspect existing records prior to implementing requirements.
4. **Rule 4 (Generic Intelligence Over Code-Based Orchestration)**: Rigid procedural scripts are banned; capabilities must strengthen generic, dynamic reasoning.
5. **Rule 5 (Mandatory Implementation & Audit Plan)**: Formulate verified plans with code-proof audits prior to execution.
6. **Rule 6 (Human-Understandable Language)**: Maintain clear, jargon-free explanations.
7. **Rule 7 (Prohibition of Surface Guards & Mandatory Exact-Line Audit)**: Ban character-level guards (`.endswith("?")`, `.startswith()`, substring checks, hardcoded stopwords). All classifications must stem from the neural tensor substrate.

---

## 2. Capabilities / Tools

### 2.1 Mathematical Computation (`math_computation`)
* **Capability ID**: `math_computation`
* **Aliases**: `["math", "calculator", "arithmetic"]`
* **Registry Class**: `ArisMathTool` ([`api/semantic/executable_tools.py`](file:///e:/Loigmax/AR.I.S/api/semantic/executable_tools.py#L96))
* **Nature**: Deterministic, actually executable. Contains zero English vocabulary.
* **Purpose**: Safe AST expression evaluation, linear equation root finding ($v = -c/m$), and positional coordinate distance metrics.
* **What AR.I.S Can Access / Control**: Internal CPU arithmetic operations via Python AST; does not access network or filesystem.
* **Input Parameters**:
  * `operation` (Optional[str]): `"evaluate"` | `"solve_linear"` | `"distance"`.
  * `expression` (Optional[str]): Contiguous arithmetic expression string (e.g. `"2015 - 1931"`).
  * `equation` (Optional[str | Dict[str, str]]): Linear equation (e.g. `{"left": "2*x + 4", "right": "10"}`).
  * `variable` (Optional[str]): Target variable name to solve for (defaults to `"x"`).
  * `param1` (Optional[float | int]): First operand.
  * `op_str` (Optional[str]): Operator symbol (`"+"`, `"-"`, `"*"`, `"/"`).
  * `param2` (Optional[float | int]): Second operand.
  * `coord_a` (Optional[float | List[float]]): First coordinate index.
  * `coord_b` (Optional[float | List[float]]): Second coordinate index.
* **Required Parameters**: At least one valid arithmetic payload (`expression`, `equation`, or `(param1, op_str, param2)`).
* **Output Structure**:
  ```python
  {
      "status": "success" | "error" | "solved",
      "value": float | int,            # Computed numerical result
      "expression": str,               # Evaluated expression
      "message": Optional[str]         # Present on error or unsolvable condition
  }
  ```
* **Execution Method**: `ArisMathTool.execute(**kwargs)` $\to$ routes to `evaluate_expression`, `solve_linear_equation`, or `calculate_distance`.
* **Limitations / Boundaries**: Rejects non-linear polynomials, trigonometric functions, and division by zero. Only supports primitive operator symbols (`+`, `-`, `*`, `/`).
* **Related Learned Operation / Template**: `op_executable_math_computation` with `operation_type="executable_capability"`.

---

### 2.2 Browser & Web Access (`browser_access`)
* **Capability ID**: `browser_access`
* **Aliases**: `["browser_search", "web_search", "internet_search"]`
* **Registry Class**: `ArisBrowserAccessTool` ([`api/semantic/executable_tools.py`](file:///e:/Loigmax/AR.I.S/api/semantic/executable_tools.py#L420))
* **Nature**: Deterministic external capability, actually executable.
* **Purpose**: External knowledge acquisition via HTTP browser simulation and web search to bridge missing fact deficits.
* **What AR.I.S Can Access / Control**: External public web endpoints via HTTP GET; parses textual search snippets and ingests them into memory.
* **Input Parameters**:
  * `query` (str): Search query string to look up or navigate in the browser.
* **Required Parameters**: `query` (non-empty string).
* **Output Structure**:
  ```python
  {
      "status": "success" | "error",
      "result": str,                     # Narrative result string
      "query": str,                      # Query executed
      "ingested_facts_count": int,       # Number of fresh facts parsed into memory
      "total_tokens": int,               # Tokens parsed from web response
      "total_relations": int,            # Relations bound from web response
      "detail": Optional[str]            # Error message if network fails
  }
  ```
* **Execution Method**: `ArisBrowserAccessTool.execute(query=...)` $\to$ HTTP GET via `requests` with 5-second timeout $\to$ linguistic tokenization $\to$ `ArisBrain.store_fact`.
* **Limitations / Boundaries**: Read-only web query access. Does not execute JavaScript, cannot authenticate behind login forms, subject to network timeout and rate limiting.
* **Related Learned Operation / Template**: `op_executable_browser_access` with `operation_type="executable_capability"`.

---

### 2.3 Command Prompt / Terminal Execution (`command_prompt`)
* **Capability ID**: `command_prompt`
* **Aliases**: `["terminal", "shell", "git_tool", "cli"]`
* **Registry Class**: `ArisCommandPromptTool` ([`api/semantic/executable_tools.py`](file:///e:/Loigmax/AR.I.S/api/semantic/executable_tools.py#L550))
* **Nature**: Deterministic execution capability, actually executable.
* **Purpose**: Executes shell and command prompt instructions within a designated working directory on the host operating system.
* **What AR.I.S Can Access / Control**: Host CLI process execution via `subprocess.run(shell=True, timeout=30)`. Can read `stdout`, `stderr`, and `returncode`.
* **Input Parameters**:
  * `command` (str): The terminal command to execute (e.g. `"git status"`, `"dir"`, `"git branch"`).
  * `working_dir` (Optional[str]): The directory / folder path where execution occurs (e.g. `"E:\\Loigmax\\tracker-v2"`).
* **Required Parameters**: `command` (non-empty string).
* **Output Structure**:
  ```python
  {
      "status": "success" | "error",
      "result": str,                     # Combined stdout or stderr
      "stdout": str,                     # Raw stdout output
      "stderr": str,                     # Raw stderr output
      "exit_code": int,                  # Process exit code (0 indicates success)
      "command": str,                    # Cleaned command executed
      "working_dir": Optional[str]       # Directory where command was run
  }
  ```
* **Execution Method**: `ArisCommandPromptTool.execute(command=..., working_dir=...)` $\to$ `subprocess.run(shell=True, timeout=30)`.
* **Limitations / Boundaries**: 30-second execution timeout; non-interactive execution only (cannot respond to interactive prompts like `[Y/N]`); restricted to host environment capabilities.
* **Related Learned Operation / Template**: `op_executable_command_prompt` with `operation_type="executable_capability"`.

---

### 2.4 Memory & Knowledge Retrieval (`memory_retrieval`)
* **Capability ID**: `memory_retrieval`
* **Registry Function / Class**: `ArisBrain.retrieve_facts` ([`api/cognitive/brain.py`](file:///e:/Loigmax/AR.I.S/api/cognitive/brain.py#L120)), `ArisRelationalRetriever` ([`api/relational/retrieval.py`](file:///e:/Loigmax/AR.I.S/api/relational/retrieval.py#L160))
* **Nature**: Cognitive reasoning and deterministic memory retrieval substrate.
* **Purpose**: Query internal relational facts and structured knowledge bindings stored in MongoDB or in-memory belief banks.
* **What AR.I.S Can Access / Control**: Reads and searches local MongoDB collections: `brain_facts`, `brain_relations`, `brain_beliefs`.
* **Input Parameters**:
  * `query` (Optional[str]): Full-text entity or statement query.
  * `subject` (Optional[str]): Subject anchor for directional lookup.
  * `relation` (Optional[str]): Relational predicate filter.
  * `object` (Optional[str]): Object anchor for reverse lookup.
  * `limit` (int): Maximum facts to return (defaults to 10).
* **Required Parameters**: At least one search anchor (`query`, `subject`, `relation`, or `object`).
* **Output Structure**:
  ```python
  {
      "matches_count": int,
      "matches": List[Dict[str, Any]],   # List of matching fact documents
      "direction": "forward" | "reverse" | "link" | "verify"
  }
  ```
* **Execution Method**: In-memory dictionary lookup across `cls._in_memory_beliefs` or MongoDB query via `db.brain_facts.find()`.
* **Limitations / Boundaries**: Read-only; cannot invent facts absent from storage.

---

### 2.5 GNN Positional Hypothesis Reasoner (`gnn_positional_hypothesis`)
* **Capability ID**: `gnn_positional_hypothesis`
* **Registry Class**: `ArisGnnHypothesisReasoner` ([`api/relational/gnn_reasoner.py`](file:///e:/Loigmax/AR.I.S/api/relational/gnn_reasoner.py#L15))
* **Nature**: Cognitive reasoning capability (reasoning-only, does not execute OS processes or external networks).
* **Purpose**: Multi-hop relational graph path inference, open-variable hypothesis discovery, and topological candidate scoring.
* **What AR.I.S Can Access / Control**: In-memory `ArisSymbolicGraph` graph structure and `ArisRelationalBeliefManager`.
* **Input Parameters**:
  * `anchor_entity` (str): Known starting entity node in graph.
  * `target_relation` (Optional[str]): Predicate constraint connecting anchor to target.
  * `graph` (`ArisSymbolicGraph`): Active graph instance.
  * `max_depth` (int): Maximum hop exploration depth (defaults to 4).
* **Required Parameters**: `anchor_entity`, `graph`.
* **Output Structure**:
  ```python
  {
      "status": "supported" | "hypothetical" | "unsupported",
      "target_entity": str,              # Deduced target entity
      "path": List[Dict[str, Any]],      # Sequence of graph edges traversed
      "score": float,                    # Topological confidence score
      "hop_count": int                   # Path length
  }
  ```
* **Execution Method**: Graph breadth-first/depth-first beam search with GNN positional score filtering.
* **Limitations / Boundaries**: Dependent on relational edges present in memory; cannot ground unlinked disconnected components.

---

## 3. Capability Affordances

Capability affordances in AR.I.S represent **abstract functional matchings** between goal structures and capability contracts—evaluated entirely through continuous metric tensor similarity, never through keyword lists.

```
+----------------------------------------------------------------------------------------------------+
|                                AR.I.S ABSTRACT AFFORDANCE MODEL                                    |
+----------------------------------------------------------------------------------------------------+
| Capability           | Satisfied Goal / Action           | Target Entity / Parameter Type          |
+----------------------+-----------------------------------+-----------------------------------------+
| math_computation     | numerical_derivation, root_find   | scalar_number, arithmetic_expression    |
| browser_access       | external_knowledge_acquisition    | entity_reference, web_query_text        |
| command_prompt       | system_terminal_execution         | filesystem_path, cli_command_string     |
| memory_retrieval     | internal_fact_verification        | stored_triple_anchor, relational_edge   |
| gnn_hypothesis       | multi_hop_path_inference          | graph_node_leaf, relational_constraint  |
+----------------------------------------------------------------------------------------------------+
```

### 3.1 Affordance Evaluation Contract (`ArisCapability.evaluate_affordance`)
Every capability evaluates affordance by projecting input representations against its schema:
$$\text{Relevance} = \begin{cases} 
\text{"yes"} & \text{if } \text{sim}(\mathbf{v}_{\text{goal}}, \mathbf{v}_{\text{domain}}) \ge 0.40 \text{ or bound params complete} \\
\text{"possible"} & \text{if structural parameters partially bound} \\
\text{"no"} & \text{otherwise}
\end{cases}$$

* **Constraints & Invariants**:
  * Tools must never inspect character strings for hardcoded trigger words (e.g. `if "browse" in query`).
  * If a capability is marked `IS_EXTERNAL = True`, it is gated by external acquisition policy (`allow_external=True`) and evaluated only when memory yields an epistemic deficit.

---

## 4. Cognitive Operations

Cognitive operations in AR.I.S are represented by `ArisOperationTemplate` ([`api/cognitive/cognitive_learner.py`](file:///e:/Loigmax/AR.I.S/api/cognitive/cognitive_learner.py#L34)). They represent de-lexicalized operational schemas learned from single-turn or multi-turn demonstrations.

### 4.1 Schema Fields of `ArisOperationTemplate`
| Field Name | Type | Description |
| :--- | :--- | :--- |
| `template_id` | `str` | Unique operation identifier (e.g. `op_executable_command_prompt`) |
| `name` | `str` | Human-understandable operation label |
| `operation_type` | `str` | `"reverse_traversal"` \| `"forward_traversal"` \| `"executable_capability"` |
| `source_role` | `str` | Position of query variable (`"subject"`, `"object"`, `"parameters"`) |
| `target_role` | `str` | Position of ground-truth answer (`"subject"`, `"object"`, `"numeric_result"`, `"output"`) |
| `anchor_role` | `str` | Position of constraint anchor (`"object"`, `"subject"`, `"relational_operator"`) |
| `demonstration_count` | `int` | Number of reinforcing demonstrations observed |
| `capability_name` | `Optional[str]`| Registered tool name dispatched by this operation |
| `parameter_mapping` | `Dict[str, Any]`| Maps query token slots to capability parameters schema |

### 4.2 Learned Operation Types
1. **`forward_traversal`**:
   * **Semantic Purpose**: Query supplies Subject anchor, asks for Object ($S \xrightarrow{R} ?$).
   * **Binding**: Variable in Object position; Anchor in Subject position.
   * **Execution**: Traverses forward edges in `ArisSymbolicGraph`.
2. **`reverse_traversal`**:
   * **Semantic Purpose**: Query supplies Object anchor, asks for Subject ($? \xrightarrow{R} O$).
   * **Binding**: Variable in Subject position; Anchor in Object position.
   * **Execution**: Traverses incoming/reverse edges in `ArisSymbolicGraph`.
3. **`executable_capability`**:
   * **Semantic Purpose**: Query supplies arguments to an external deterministic tool.
   * **Binding**: Tokens matching parameter slots are extracted and bound to tool schema.
   * **Execution**: Calls `ArisCapabilityRegistry.dispatch(template.capability_name, **bound_params)`.

---

## 5. Epistemic States

AR.I.S classifies every user goal into one of five formal epistemic states (`ArisEpistemicState`, [`api/cognitive/epistemic_state.py`](file:///e:/Loigmax/AR.I.S/api/cognitive/epistemic_state.py#L21)):

```
                                [ User Goal G ]
                                       │
                    Is G a self-contained computation?
                               ┌───────┴───────┐
                              Yes              No
                               │               │
                        [ DERIVABLE ]    Probe Memory
                                               │
                                      Facts Found in Memory?
                                       ┌───────┴───────┐
                                      Yes              No
                                       │               │
                                   [ KNOWN ]    External Sensor Available?
                                                 ┌─────┴─────┐
                                                Yes          No
                                                 │           │
                                          [ ACQUIRABLE ] [ UNSUPPORTED ]
```

### 5.1 Exact Evaluation Conditions

| Epistemic State | Operational Definition | Exact Code Condition in `ArisEpistemicEvaluator` | Action Taken |
| :--- | :--- | :--- | :--- |
| **`KNOWN`** | Memory graph directly satisfies the goal. Exact anchor and matching relation are present. | `matching_facts` contains facts where relational binding matches goal relation or entity anchor. | Return fact directly from memory graph. **Zero tool calls permitted**. |
| **`DERIVABLE`** | Goal requires mathematical execution OR internal memory graph contains multi-hop path connecting anchors. | `goal.target_type in ("numerical_value", "algebraic_equation")` OR `anchor_in_graph and target_in_graph`. | Dispatch `math_computation` or `gnn_positional_hypothesis`. |
| **`ACQUIRABLE`** | Required anchor/relation is absent from internal memory ($\Delta = \text{Goal} - \text{Memory} > 0$), but matches an external sensor contract. | `len(matching_facts) == 0` AND external capability (`IS_EXTERNAL = True`) is registered. | Dispatch external intake sensor (`browser_access`), ingest fresh facts, update memory. |
| **`UNSUPPORTED`** | Information is absent from memory and no registered capability can bridge the deficit. | `len(matching_facts) == 0` AND no capability contract matches the deficit. | Output transparent admission of ignorance without hallucination. |
| **`AMBIGUOUS`** | Input lacks sufficient structural grounding to formulate an actionable goal. | Input entities or anchors cannot be grounded into candidates. | Formulate clarification request. |

---

## 6. Goal / Intent Representation

The goal representation is instantiated by the `ArisGoal` dataclass ([`api/cognitive/epistemic_state.py`](file:///e:/Loigmax/AR.I.S/api/cognitive/epistemic_state.py#L30)):

```python
@dataclass
class ArisGoal:
    raw_query: str                          # Original user utterance
    target_variable: str                    # '?' | 'result' | 'entity' | 'fact'
    target_type: str                        # 'numerical_value' | 'algebraic_equation' | 'entity_reference' | 'knowledge_acquisition' | 'system_terminal_execution'
    anchor_entity: Optional[str] = None     # Grounded entity name (e.g. 'A.P.J. Abdul Kalam', 'tracker-v2')
    relational_constraint: Optional[str] = None # Predicate constraint (e.g. 'president of', 'status')
    parameters: Dict[str, Any] = field(default_factory=dict) # Bound slot arguments
    confidence: float = 1.0                 # Goal formation confidence
```

### Structural Derivation
* Goals are formed by `ArisGoalFormer.form_goal()`:
  * Uses token structural types (`type: "path"`, `type: "number"`).
  * Uses syntactic interrogative locus (`interrogative_head`, `wh_type`).
  * Zero keyword trigger lists (no hardcoded `"search"`, `"who"`, `"calculate"`).

---

## 7. Planning / Multi-Step Reasoning

Multi-step deliberation is governed by `ArisReasoningPlanner` ([`api/cognitive/deliberative_planner.py`](file:///e:/Loigmax/AR.I.S/api/cognitive/deliberative_planner.py#L54)) and `ArisCognitiveReflector` ([`api/cognitive/cognitive_reflector.py`](file:///e:/Loigmax/AR.I.S/api/cognitive/cognitive_reflector.py#L12)).

### 7.1 Sub-Goal Dependency Node (`SubGoalNode`)
```python
@dataclass
class SubGoalNode:
    goal_id: str                       # UUID string
    goal_type: str                     # 'relational_retrieval' | 'capability_execution' | 'math_evaluation' | 'verification'
    query_text: str                    # Sub-query clause text
    target_variable: str               # Target variable slot (e.g. '?', '$v1')
    anchor_entity: Optional[str]       # Anchor entity for this step
    relation: Optional[str]            # Relational constraint
    capability_name: Optional[str]     # Target tool dispatched
    parameters: Dict[str, Any]         # Arguments passed to tool
    depends_on: List[str]              # List of parent goal_ids required before execution
    output_variable: Optional[str]     # Variable name populated by this step
    resolved_value: Optional[Any]      # Result returned by step execution
    execution_status: str              # 'pending' | 'success' | 'failed' | 'backtracked'
```

### 7.2 Multi-Step Execution & Variable Propagation
1. **Clause & Intent Deconstruction**: `ArisReasoningPlanner.create_plan()` splits composite queries on coordination conjunctions and clause boundaries (`" and then "`, `" and "`, etc.).
2. **Variable Propagation**: If Step 1 returns an entity (e.g., `resolved_value = "1931"`), Step 2 receives that variable as an operand in its parameter binding:
   $$\text{Step 1: } \text{Birth Year of Kalam} \to 1931 \implies \text{Step 2: } 2015 - 1931 \to 84$$
3. **Metacognitive Self-Auditing (`ArisCognitiveReflector`)**:
   * **Circular Reasoning Check**: Verifies that an edge does not traverse $A \to B \to A$.
   * **Contradiction Detection**: Checks candidate facts against `ArisRelationalBeliefManager.is_fact_denied()`.
   * **Automated Backtracking**: If a path yields a dead-end, the engine reverts to the parent node and attempts an alternate branch.

---

## 8. Semantic / Linguistic Representation

### 8.1 Universal Dependencies (UD) Substrate
* **Tokens**: Stored as dictionaries with `sequence_index`, `raw`, `token`, `type` (`"word"`, `"number"`, `"path"`, `"punct"`), `position` (`{"char_start", "char_end"}`).
* **Morphosyntax**: `ArisUDTensorEngine` ([`api/linguistic/ud_tensor_engine.py`](file:///e:/Loigmax/AR.I.S/api/linguistic/ud_tensor_engine.py#L22)) projects token embeddings against in-memory PyTorch exemplar matrices to predict Universal POS (`UPOS`) and dependency relation (`deprel`).

### 8.2 Syntactic Frames (`SyntacticFrame`)
* Defined in `api/linguistic/syntactic_normalizer.py`:
  * `speech_act`: `SpeechActType.INTERROGATIVE` | `DECLARATIVE` | `IMPERATIVE`.
  * `subject`: Extracted nominal subject span.
  * `predicate`: Relational or verbal head.
  * `object`: Direct/indirect object span.
  * `answer_slot`: Expected target role (`"subject"` | `"object"` | `"boolean"`).

### 8.3 Semantic Relation Abstraction
* `ArisRelationAbstraction` ([`api/relational/relation_abstraction.py`](file:///e:/Loigmax/AR.I.S/api/relational/relation_abstraction.py#L18)):
  * Embeds relations into a continuous centroid prototype manifold using scale-free $(i, j)$ subword embeddings (`dim=128`).
  * Enables zero-shot generalization across morphological variants (`"founded"`, `"founder of"`, `"established by"`, `"created"`).

### 8.4 Symbolic Relational Graph (`ArisSymbolicGraph`)
* Adjacency list: `Dict[str, List[SymbolicEdge]]` where each edge stores:
  * `from_entity: str`
  * `relation: str`
  * `to_entity: str`
  * `direction: str` (`"forward"` | `"reverse"`)
  * `confidence: float`

---

## 9. Proposed PyTorch Training Dataset Structure

To train the comprehensive neural model for AR.I.S, the dataset must be structured into 7 distinct sample categories, corresponding directly to the repository's cognitive engines:

### 9.1 Linguistic Samples (`sample_type="linguistic"`)
* **Input Representation**: Tokenized character and subword sequences $\mathbf{X} \in \mathbb{R}^{T \times D}$.
* **Target Representation**: UPOS label indices and dependency head indices.
* **Features**: Token character length, capitalization bit, numeral bit, path bit, token position.
* **Expected Output**: UPOS class distribution ($C=17$) and dependency relation tag ($R=37$).

### 9.2 Semantic Frame Samples (`sample_type="semantic_frame"`)
* **Input Representation**: Sentence token vectors and candidate boundary masks.
* **Target Representation**: Speech Act (`PHATIC`, `DECLARATIVE`, `INTERROGATIVE`) and Argument Roles (`Subject`, `Predicate`, `Object`, `Answer Slot`).
* **Features**: 14 non-lexical positional invariants from `ArisNeuralFrameClassifier`.
* **Expected Output**: Frame role assignment vector for each candidate span.

### 9.3 Relational Binding Samples (`sample_type="relational_binding"`)
* **Input Representation**: Sentence embedding and pair of entity candidate vectors $(\mathbf{e}_1, \mathbf{e}_2)$.
* **Target Representation**: Canonical predicate class or continuous relation prototype centroid.
* **Features**: Scaled subword n-gram embedding difference vector $\mathbf{v}_{\text{rel}} = \mathbf{e}_2 - \mathbf{e}_1$.
* **Expected Output**: Relational cluster assignment and edge directionality (`forward` vs `reverse`).

### 9.4 Epistemic Evaluation Samples (`sample_type="epistemic_state"`)
* **Input Representation**: Concatenation of Goal Vector $\mathbf{g}$, Memory Density Vector $\mathbf{m}$, and Candidate Support Vector $\mathbf{s}$.
* **Target Representation**: Epistemic State class index ($0=\text{KNOWN}, 1=\text{DERIVABLE}, 2=\text{ACQUIRABLE}, 3=\text{UNSUPPORTED}$).
* **Features**: Memory match count, relation alignment score, graph topological connectivity score.
* **Expected Output**: Softmax probability over the 4 epistemic states.

### 9.5 Capability Affordance Samples (`sample_type="capability_affordance"`)
* **Input Representation**: Goal semantic vector $\mathbf{g} \in \mathbb{R}^D$ and Candidate Token Vectors $\mathbf{C} \in \mathbb{R}^{K \times D}$.
* **Target Representation**: One-hot capability class index ($0=\text{math}, 1=\text{browser}, 2=\text{command\_prompt}, 3=\text{memory\_retrieval}, 4=\dots$).
* **Features**: Domain cosine similarity, structural token match count (paths, numbers, queries).
* **Expected Output**: Capability relevance score vector ($V \in \mathbb{R}^{|\text{Capabilities}|}$).

### 9.6 Parameter Slot-Binding Samples (`sample_type="slot_binding"`)
* **Input Representation**: Candidate Span Vectors $\mathbf{C} \in \mathbb{R}^{K \times D}$ and Capability Parameter Slot Vectors $\mathbf{S} \in \mathbb{R}^{M \times D}$.
* **Target Representation**: Binary assignment matrix $\mathbf{A} \in \{0, 1\}^{K \times M}$ indicating which candidate binds to which parameter slot.
* **Features**: Continuous bilinear tensor product matrix $\mathbf{M} = \mathbf{C} \mathbf{W} \mathbf{S}^\top$.
* **Expected Output**: Softmax alignment over candidate tokens for each parameter slot.

### 9.7 Multi-Step Planning Samples (`sample_type="multi_step_plan"`)
* **Input Representation**: Composite multi-intent goal representation.
* **Target Representation**: Sequence of sub-goal nodes, their assigned capabilities, and variable propagation edges.
* **Features**: Clause delimiter boundary mask, dependency adjacency matrix $\mathbf{D} \in \{0, 1\}^{N \times N}$.
* **Expected Output**: Acyclic dependency graph over sub-goals with assigned capability labels.

---

## 10. Tensorization Requirements

For compilation into PyTorch `.pt` model files:

| Specification | Dimension / Value | Details |
| :--- | :--- | :--- |
| **Semantic Embedding Dimension ($D$)** | $128$ | Base coordinate embedding dimension produced by `ArisSemanticEncoder`. |
| **Max Sequence Length ($T$)** | $64$ | Maximum token sequence length for sentence-level inputs; zero-padded with boolean masks. |
| **Max Candidates ($K$)** | $16$ | Maximum entity/parameter candidates considered per utterance. |
| **Max Parameter Slots ($M$)** | $8$ | Maximum slots per capability schema. |
| **Max Sub-Goals ($N$)** | $8$ | Maximum steps in a multi-step deliberative plan. |
| **Continuous Values** | `torch.float32` | Metric manifolds, cosine projections, bilinear tensor matrices, confidence scores. |
| **Categorical Labels** | `torch.long` | State classes, UPOS tags, capability registry indices, dependency heads. |
| **Graph Adjacency Tensors** | `(B, N, N)` float | Directed dependency matrices for planning and multi-hop GNN reasoning. |
| **Attention / Padding Masks** | `torch.bool` | Explicit padding masks for variable-length token sequences and candidate lists. |
| **Batching Standard** | Fixed size ($B=32$ or $B=64$) | Stacked tensors with length masks; zero runtime dynamic graph reshaping. |

---

## 11. Source Traceability

| Extracted Component | Source Knowledge Brain Document | Source Code File & Class |
| :--- | :--- | :--- |
| **System Invariants & Rules** | [`knowledge_brain/aris_constitution.md`](file:///e:/Loigmax/AR.I.S/knowledge_brain/aris_constitution.md), [`knowledge_brain/generic_cognition.md`](file:///e:/Loigmax/AR.I.S/knowledge_brain/generic_cognition.md) | `AGENTS.md`, `GEMINI.md` |
| **Rule 7 (Surface Guards Ban)** | [`knowledge_brain/rule_surface_guards_prohibition_and_line_audit.md`](file:///e:/Loigmax/AR.I.S/knowledge_brain/rule_surface_guards_prohibition_and_line_audit.md) | Core system directives |
| **Math Capability** | [`knowledge_brain/mathematical_cognitive_substrate.md`](file:///e:/Loigmax/AR.I.S/knowledge_brain/mathematical_cognitive_substrate.md) | `api/semantic/executable_tools.py:ArisMathTool` |
| **Browser Capability** | [`knowledge_brain/browser_access_capability.md`](file:///e:/Loigmax/AR.I.S/knowledge_brain/browser_access_capability.md) | `api/semantic/executable_tools.py:ArisBrowserAccessTool` |
| **Command Prompt Capability** | [`knowledge_brain/command_prompt_capability_and_multi_tool_planner.md`](file:///e:/Loigmax/AR.I.S/knowledge_brain/command_prompt_capability_and_multi_tool_planner.md) | `api/semantic/executable_tools.py:ArisCommandPromptTool` |
| **Capability Registry** | [`knowledge_brain/executable_capability_parameter_binding.md`](file:///e:/Loigmax/AR.I.S/knowledge_brain/executable_capability_parameter_binding.md) | `api/semantic/executable_tools.py:ArisCapabilityRegistry` |
| **Affordance Tensor Engine** | [`knowledge_brain/epistemic_cognition_tier6a.md`](file:///e:/Loigmax/AR.I.S/knowledge_brain/epistemic_cognition_tier6a.md) | `api/cognitive/affordance_tensor_engine.py:ArisAffordanceTensorEngine` |
| **Epistemic States & Goal** | [`knowledge_brain/epistemic_cognition_tier6a.md`](file:///e:/Loigmax/AR.I.S/knowledge_brain/epistemic_cognition_tier6a.md) | `api/cognitive/epistemic_state.py:ArisEpistemicState, ArisGoal` |
| **Learned Operation Templates**| [`knowledge_brain/cognitive_learning_tier5.md`](file:///e:/Loigmax/AR.I.S/knowledge_brain/cognitive_learning_tier5.md) | `api/cognitive/cognitive_learner.py:ArisOperationTemplate` |
| **Deliberative Planner** | [`knowledge_brain/deliberative_reasoning_reflection_phase3.md`](file:///e:/Loigmax/AR.I.S/knowledge_brain/deliberative_reasoning_reflection_phase3.md) | `api/cognitive/deliberative_planner.py:SubGoalNode, ArisReasoningPlanner` |
| **Metacognitive Reflector** | [`knowledge_brain/deliberative_reasoning_reflection_phase3.md`](file:///e:/Loigmax/AR.I.S/knowledge_brain/deliberative_reasoning_reflection_phase3.md) | `api/cognitive/cognitive_reflector.py:ArisCognitiveReflector` |
| **Pragmatic Metric Tensor** | [`knowledge_brain/dialogue_context_coreference_phase1.md`](file:///e:/Loigmax/AR.I.S/knowledge_brain/dialogue_context_coreference_phase1.md) | `api/cognitive/pragmatic_tensor_engine.py:ArisPragmaticTensorEngine` |
| **Goal Reasoning Tensor** | [`knowledge_brain/goal_conditioned_relational_reasoning.md`](file:///e:/Loigmax/AR.I.S/knowledge_brain/goal_conditioned_relational_reasoning.md) | `api/cognitive/goal_reasoning_tensor_engine.py:ArisGoalReasoningTensorEngine` |
| **Universal Dependencies Engine**| [`knowledge_brain/universal_dependencies_linguistic_training.md`](file:///e:/Loigmax/AR.I.S/knowledge_brain/universal_dependencies_linguistic_training.md) | `api/linguistic/ud_tensor_engine.py:ArisUDTensorEngine` |
| **GNN Relational Reasoner** | [`knowledge_brain/gnn_relational_reasoning_tier7.md`](file:///e:/Loigmax/AR.I.S/knowledge_brain/gnn_relational_reasoning_tier7.md) | `api/relational/gnn_reasoner.py:ArisGnnHypothesisReasoner` |

---

## 12. UNRESOLVED_INFORMATION

The following items cannot be fully verified from the current repository code or Knowledge Brain files and require future specification before final PyTorch dataset generation:

1. **Optimal Affordance Training Dataset Scale**:
   * The exact number of supervised exemplars per capability slot needed to achieve $>95\%$ slot-binding accuracy across noisy, ungrammatical user utterances is not yet empirically benchmarked.
2. **Dynamic Capability Removal / Unregistration**:
   * Currently, `ArisCapabilityRegistry` supports dynamic registration via `_CAPABILITIES[name] = tool_cls`. An explicit unregistration and versioning lifecycle contract (`unregister()`, `get_version()`) is not yet formalized in `executable_tools.py`.
3. **Continuous Cross-Capability Chaining Protocols**:
   * While `ArisReasoningPlanner` creates multi-step plans with variable propagation for Git operations (`browser_access` $\to$ `command_prompt`), an arbitrary generic type-checking contract verifying output-type to input-type compatibility between two arbitrary unknown 3rd-party tools at compile-time is pending formalization.
4. **PyTorch Batch Training Loss Weighting**:
   * The precise multi-task loss balance between Slot-Binding Loss ($\mathcal{L}_{\text{slot}}$), Epistemic Classification Loss ($\mathcal{L}_{\text{epistemic}}$), and Relational Alignment Loss ($\mathcal{L}_{\text{rel}}$) is not yet fixed in `models/`.
