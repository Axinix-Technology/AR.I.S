# Sensory Structural Advisor Primitive (Layer 0)

> **Foundational Milestone Directive**:
> *"The advisor observes. AR.I.S learns, identifies patterns, reasons, allocates work, and answers."*

---

## 1. Architectural Philosophy & The Cognitive Boundary

AR.I.S must never be an API wrapper masquerading as an intelligence. External packages and models act strictly as **sensory receptors** (analogous to biological eyes/ears), observing structural syntactic slots in human language without accessing memory or answering questions.

```
                    HUMAN LANGUAGE
                           │
                           ▼
                  ┌─────────────────┐
                  │ Tier 1 Tokenizer │
                  └────────┬────────┘
                           ▼
                  ┌─────────────────┐
                  │ Tier 2 Concepts │
                  └────────┬────────┘
                           ▼
                  ┌─────────────────┐
                  │ Tier 3 Binding  │
                  │ S ── R ── O     │
                  └────────┬────────┘
                           │
                    structural frame
                           │
                           ▼
              ┌────────────────────────┐
              │ STRUCTURAL ADVISOR     │
              │                        │
              │ OBSERVATION ONLY       │
              │                        │
              │ variable_slot          │
              │ anchor_slot            │
              └───────────┬────────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ AR.I.S BRAIN       │
                │                    │
                │ learns             │
                │ identifies pattern │
                │ selects operation  │
                │ reasons            │
                │ traverses memory   │
                └─────────┬──────────┘
                          │
                          ▼
                    MONGODB MEMORY
                          │
                          ▼
                       ANSWER
```

---

## 2. The Strict Sensory Contract

1. **Input to Advisor**:
   The advisor receives only the structural frame produced by Tier 1–3:
   ```text
   subject: 'A', object: 'Who'
   ```
2. **Output from Advisor**:
   The advisor outputs strictly a structural JSON observation:
   ```json
   {
     "variable_slot": "object",
     "anchor_slot": "subject"
   }
   ```
3. **Strict Constraints**:
   - The advisor **never** receives facts stored in AR.I.S memory.
   - The advisor **never** answers the question.
   - The advisor **never** decides traversal direction (forward vs reverse).
   - The advisor **never** re-interprets the relation span.

---

## 3. Clean Failure Policy (Zero Hard Patches)

If the advisor is disabled (`use_advisor=False`), network fails, or credentials are unavailable:
$$\text{advisor\_status} = \text{"unavailable"} \longrightarrow \text{AR.I.S answers 0 matches}$$

No hardcoded vocabulary fallback (`if "who" in text:`) is permitted. This guarantees clean scientific boundaries and prevents hidden procedural bypasses.

---

## 4. Autonomous Cognitive Decision & Execution

Once the sensory observation is received:
1. **Template Matching**: AR.I.S inspects its de-lexicalized `brain_operation_templates`:
   - Observation `(variable=subject, anchor=object)` matches template where `anchor_role == "object"` $\to$ AR.I.S selects `reverse_traversal`.
   - Observation `(variable=object, anchor=subject)` matches template where `anchor_role == "subject"` $\to$ AR.I.S selects `forward_traversal`.
2. **Deterministic Memory Traversal**:
   - AR.I.S constructs `ArisRelationalQuery` and executes graph retrieval over MongoDB memory.
   - Grounds and returns the verified memory answer.

---

## 5. Control & Ablation Verification Matrix

| Experiment | Query | Advisor Role | AR.I.S Action | Result | Verification Significance |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Reverse Traversal** | `"Who daxels B."` | `var=subject, anc=object` | Reverse traversal `? --daxels--> B` | **`A`** | Verified reverse operation |
| **Forward Traversal** | `"A daxels Who."` | `var=object, anc=subject` | Forward traversal `A --daxels--> ?` | **`B`** | Verified forward operation |
| **Lowercase Robustness** | `"who daxels B."` / `"A daxels who."` | Observes slots | Traverses grounded memory | **`A` / `B`** | Robust to casual typing without dictionaries |
| **Control 1 (Nonexistent Fact)** | `"C daxels Who."` | `var=object, anc=subject` | Queries memory `C --daxels--> ?` | **`0`** | Proves advisor does NOT hallucinate answers |
| **Control 2 (Wrong Direction)** | `"Who daxels A."` | `var=subject, anc=object` | Queries memory `? --daxels--> A` | **`0`** | Proves memory grounding integrity |
| **Control 3 (Unseen Vocabulary)** | `"Who mivens D."` / `"C mivens Who."` | Observes slots | Traverses memory for `mivens` | **`C` / `D`** | Proves generalization across unseen words |
| **Control 4 (Alien Language)** | `"Who qevron Y."` / `"X qevron Who."` | Observes slots | Traverses memory for `qevron` | **`X` / `Y`** | Proves pure structural operation independence |
| **Control 5 (Ablation Test)** | `"A daxels Who." --no-advisor` | Disabled | Advisor unavailable $\to$ no answer | **`0`** | Proves absence of hidden procedural bypasses |
