# Browser Access Capability (Executable Tools)

## Overview
The **Browser Access Capability** provides AR.I.S with deterministic web query search and retrieval functionality without hardcoding or modifying the underlying neural frame classifier.

---

## Architectural Principles

1. **Zero Model Modifications**:
   - The local neural frame classifier (`api/neural_frame_classifier.py`), network weights, and dataset remain completely unchanged.
   - The neural model continues to specialize strictly in syntax/structural perception (variable vs anchor slots).

2. **Dynamic Capability Discovery**:
   - The capability is registered in `ArisCapabilityRegistry` under `browser_access`.
   - `ArisCapabilityRegistry.get_capabilities_knowledge()` dynamically generates the capability descriptor and parameters schema.
   - The sensory advisor (`ArisStructuralAdvisor`) automatically injects this registered schema into its dynamic system prompt:
     ```text
     - Capability 'browser_access': Performs browser access for web query search and retrieval.
       Parameters: query: The search query string to look up or navigate in the browser
     ```

3. **Deterministic Execution & Generic Parameter Binding**:
   - `ArisBrowserAccessTool.execute(query="...")` receives the bound query parameter from AR.I.S and executes web query search safely via HTTP.
   - **Zero Hardcoded Parameters**: The cognitive learner (`ArisCognitiveLearner`) does not expect fixed parameter names like `param1`, `op_str`, or `param2`. All registered tools forward their dynamic schema parameters (`**params`) directly to `ArisCapabilityRegistry.dispatch()`.

4. **Cognitive Memory Ingestion (Tier 1 Tokens, Tier 2 Candidates, Tier 3 Relations)**:
   - Retrieved browser snippets are automatically parsed into atomic factual statements and stored via `ArisBrain.store_fact()`.
   - Each statement undergoes full cognitive ingestion:
     - **Tier 1**: Deterministic tokenization with positional coordinates and token types.
     - **Tier 2**: Concept and entity candidate spans extraction.
     - **Tier 3**: Relational binding (subject, relation, object) discovery.
   - Harvested knowledge is persisted in MongoDB (`brain_facts`), making it immediately queryable and traversable by AR.I.S relational memory.

---

## Verification & Audit
- `ArisCapabilityRegistry.get_capability("browser_access")` -> `ArisBrowserAccessTool`
- `ArisCapabilityRegistry.get_capability("browser_search")` -> `ArisBrowserAccessTool` (alias)
- Registered capabilities dynamically inform the LLM Teacher without hardcoding.
- Generic parameter dispatch works across mathematical, browser, and arbitrary future tools without touching cognitive core reasoning logic.
- Browser search results automatically convert into tokens, entity candidates, and relational bindings stored in MongoDB.


