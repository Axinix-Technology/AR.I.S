# Executable Capability Registry & Functional Parameter Binding

## 1. Overview & Core Directive
AR.I.S (Artificial Intelligent System) is an autonomous cognitive system. Executable capabilities (such as arithmetic math computation, browser search, or external APIs) are deterministic tools, **NOT** cognitive reasoners.
- Tools do NOT decide intent, do NOT reason, and do NOT classify variables.
- Tools receive **BOUND** parameters directly from AR.I.S and execute computations or external queries.
- The Teacher / Structural Advisor is provided the knowledge of registered capabilities and their parameter schemas, observing query parameter frames.
- AR.I.S learns the executable operation template from demonstrations, binds parameters generically, and executes tools autonomously offline without ongoing supervision.

---

## 2. Architectural Invariants (Non-Negotiable)

1. **Zero Lexical Shortcuts / No Verbal Cheats**:
   - `ArisMathTool.OPERATORS` strictly accepts primitive math operator symbols: `+`, `-`, `*`, `/`.
   - English linguistic synonyms (`plus`, `minus`, `times`, `divided by`) are strictly forbidden in tool operator mappings.
2. **Generic Tool Schemas Provided to Teacher**:
   - The system prompt contains no hardcoded capability names or `e.g.` examples.
   - `ArisCapabilityRegistry.get_capabilities_knowledge()` dynamically extracts all registered capability schemas (name, description, parameters).
3. **Pure Parameterized Execution**:
   - The canonical demonstration:
     `"What is 1 + 1?"` (guided with Teacher) $\to$ AR.I.S learns the executable template and grounds execution $\to$ `2`.
   - The canonical offline killer benchmark:
     `"What is 10 - 4?"` (`--no-advisor`) $\to$ AR.I.S independently identifies parameters (`param1 = 10`, `op_str = -`, `param2 = 4`), binds parameters from the learned template, dispatches to `math_computation`, and returns `6` completely offline without retraining.
4. **Generalization Across Unseen Operations & Compositional Multi-Term Chains**:
   - Automatically executes unseen multiplication (`25 * 4 = 100`) and division (`100 / 5 = 20`) without any hardcoded if-statements or verbal shortcuts.
   - Fully supports **compositional multi-term expressions** (`1 + 1 + 1 = 3`, `10 - 4 - 2 = 4`, `2 + 3 * 4 = 14`). Parameter extraction binds the full contiguous parameter expression chain rather than arbitrarily stopping at two operands.


---

## 3. Cognitive Component Architecture

```
User Query: "What is 10 - 4?" (Offline Mode: --no-advisor)
   │
   ▼
[Tier 1 Tokenizer] ➔ Tokens: ['What', 'is', '10', '-', '4', '?']
   │
   ▼
[AR.I.S Cognitive Reasoner]
   │
   ├── Matches Learned Executable Template: 'op_executable_math_computation'
   ├── Parameter Extraction: param1='10', op_str='-', param2='4'
   └── Dynamic Parameter Binding
   │
   ▼
[ArisCapabilityRegistry.dispatch("math_computation", param1=10, op_str='-', param2=4)]
   │
   ▼
[ArisMathTool] ➔ Deterministic Execution: 10 - 4 = 6
   │
   ▼
Grounded Answer: 6
```

---

## 4. Verification & Audit Trail
- Automated tests in `api/tests.py` under `FunctionalParameterBindingCapabilityTests`:
  - `test_math_tool_primitive_operators_only`: Confirms only `+`, `-`, `*`, `/` exist.
  - `test_capability_registry_dynamic_knowledge`: Confirms dynamic knowledge generation.
  - `test_teacher_guided_demonstration_1_plus_1`: Confirms 1+1 grounding.
  - `test_canonical_killer_test_10_minus_4_offline`: Confirms 10-4 offline autonomous result `6`.
  - `test_offline_multiplication_and_division`: Confirms 25*4=100 and 100/5=20.
  - `test_local_neural_classifier_offline_prediction`: Confirms local neural classifier offline predictions.
