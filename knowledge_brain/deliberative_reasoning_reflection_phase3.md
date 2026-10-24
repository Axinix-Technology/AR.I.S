# Deliberative Chain-of-Thought & Metacognitive Reflection (Phase 3)

**Classification**: Autonomous Cognitive Core / Phase 3 Reasoning & Planning Substrate  
**Module References**: [deliberative_planner.py](file:///e:/Loigmax/AR.I.S/api/cognitive/deliberative_planner.py), [cognitive_reflector.py](file:///e:/Loigmax/AR.I.S/api/cognitive/cognitive_reflector.py)  
**Cross-References**: [ArisBrain](file:///e:/Loigmax/AR.I.S/api/cognitive/brain.py), [Neural-Symbolic Generative Decoder](file:///e:/Loigmax/AR.I.S/knowledge_brain/neural_symbolic_generative_decoder.md), [Deontic & Affective Cognition](file:///e:/Loigmax/AR.I.S/knowledge_brain/deontic_affective_cognition.md)

---

## 1. Context & Architectural Paradigm Shift

Early reasoning systems relied on rigid "1-hop" vs "multi-hop" procedural pipelines and manual template definitions. 

In **AR.I.S Phase 3**, we move to **Goal-Directed Autonomous Deliberation**:
The Brain directly perceives the **user's intent, anchor entities, and final epistemic goal**. It dynamically navigates whatever depth and capability sequence is required (relational graph traversal, mathematical calculation, verification) without manual hop counters or brittle pipelines.

```
+--------------------------------------------------------------------------------------------------+
|                            AR.I.S DELIBERATIVE REASONING ENGINE                                  |
+--------------------------------------------------------------------------------------------------+
| 1. Complex User Prompt        ──► [Autonomous Intent & Goal Deconstruction]                     |
| 2. Deliberative Planner (CoT) ──► [Step 1 ──► Step 2 ──► Step 3 (Sub-traversals & Tools)]        |
| 3. Metacognitive Reflector    ──► [Self-Audit: Verify Contradictions, Circular Loops, Proof]     |
| 4. Self-Correction Loop       ──► [Backtrack / Revise Hypothesis if Deficit / Conflict Found]    |
| 5. Brain Decider Gateway      ──► [ArisBrain Commits Final Grounded Result & Thinking Trace]     |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Core Components

### A. Dynamic Goal Planner (`ArisReasoningPlanner`)
- **Sub-Goal Dependency Trees**: Deconstructs composite and multi-intent queries into an acyclic dependency graph.
- **Dynamic Variable Propagation**: Passes outputs from upstream steps (e.g., an entity retrieved from memory) as arguments to downstream tools (e.g., mathematical AST evaluation or secondary graph lookups).

### B. Metacognitive Reflector & Self-Auditing (`ArisCognitiveReflector`)
- **Circular Reasoning Detection**: Prevents infinite loops across symmetric or inverse graph relations ($A \to B \to A$).
- **Contradiction & Denial Inhibition**: Cross-checks hypotheses against recorded human feedback and belief matrices.
- **Automated Backtracking**: If an intermediate path fails or encounters a contradiction, the engine automatically backtracks to explore alternative paths.

### C. Deliberative Thinking Tracer (`ArisDeliberativeTracer`)
- Emits transparent, human-understandable reasoning traces ("Thinking Steps") alongside the grounded final answer.
- Integrates seamlessly with `ArisBrain.deliberate_reasoning()` and `ArisGroundedGenerator`.
