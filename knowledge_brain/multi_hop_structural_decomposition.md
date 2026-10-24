# Multi-Hop Structural Decomposition & Graph Grounding (Tier 6-B)

## 1. Overview & Philosophical Foundation
In real-world cognitive inquiries, questions are often compositional, multi-step, and targeted across various semantic domains (e.g. *"Who is the brother of the leader of Axinix?"* vs *"Where does the brother of the leader of Axinix live?"*).

Under the AR.I.S Constitution:
- **Zero Python-Orchestrated Pipelines**: No hardcoded procedural pipelines like `if "brother of" in query:` or hardcoded dictionaries like `{"where": "location", "who": "entity"}`.
- **Generic Cognitive Boundary & Unconstrained Perception**:
  - The **Structural Advisor (Teacher / LLM)** acts as an **unconstrained perception sensor** (`observe_query`). It extracts all available structural dimensions in one unified pass: semantic target domain (`target_domain`), question placeholder (`target_variable`), concrete anchors (`anchors`), start entity (`start_entity`), and ordered relational hops (`hops`).
  - It contains zero domain-specific examples, zero query hardcodes, and does NOT limit the AI to narrow task subsets.
  - **Dynamic Vocabulary & Interrogative Learning**: AR.I.S autonomously learns the association between question tokens (`"Where"`, `"Who"`, `"When"`, or tokens in any language) and target domains (`location`, `entity`, `temporal`) into MongoDB collection `brain_interrogative_associations`.
  - **AR.I.S Retains 100% Cognitive Ownership**: AR.I.S feeds the extracted relational sequence into its deterministic Tier 4-C graph engine (`ArisRetriever.execute_multi_hop`), traverses stored MongoDB facts step-by-step, verifies graph provenance, and returns the grounded answer.

---

## 2. Architecture & Flow

```
Human Inquiry: "Where does the brother of the leader of Axinix live?"
                         │
                         ▼
        [Step 1: Direct 1-Hop Memory Probe]
                         │
                         ▼ (0 direct matches)
           [ArisStructuralAdvisor.observe_query]
     (Unified Unconstrained Perception Sensor)
                         │
                         ▼
       {
         "type": "relational_traversal",
         "target_domain": "location",
         "target_variable": "where",
         "anchors": ["Axinix"],
         "start_entity": "Axinix",
         "hops": [
           {"relation": "leads", "direction": "reverse"},
           {"relation": "is brother of", "direction": "forward"},
           {"relation": "lives in", "direction": "forward"}
         ]
       }
                         │
                         ▼
   [AR.I.S Cognitive Learning: Interrogative Token Association]
    Records: "where" <-> "location" into brain_interrogative_associations
                         │
                         ▼
       [ArisRetriever.execute_multi_hop]
       (Tier 4-C Deterministic Memory Traversal)
                         │
        Hop 1: Axinix --[leads (rev)]--> Arun
        Hop 2: Arun --[is brother of]--> Bala
        Hop 3: Bala --[lives in]--> Chennai
                         │
                         ▼
    Target Grounded Answer: "Chennai"
    Provenance: Axinix -> Arun -> Bala -> Chennai
```

---

## 3. Cognitive Grounding vs. Narrow Pipelines
| Narrow Pipeline Approach | AR.I.S Generic Cognitive Architecture |
| :--- | :--- |
| Hardcoded dictionary: `if query.startswith('where'): domain = 'location'`. | AR.I.S learns token-domain associations dynamically from demonstrations into MongoDB memory. |
| Separate methods and separate prompts for multi-hop vs single-hop. | Single, unified perception sensor (`observe_query`) providing all structural data without limiting the LLM. |
| Tailored prompt examples containing the user's specific test questions. | Strictly generic system prompt describing abstract roles; zero test question leakage. |
| Blind offline template selection that confuses Person answers with Locations. | Offline template selection verifies semantic target domain and relational grounding. |

---

## 4. Continuous Template Learning & Offline Execution
When a multi-hop traversal is demonstrated by the Advisor and successfully grounded against memory facts:
1. **Auto-Persistence**: AR.I.S calls `ArisCognitiveLearner.learn_multihop_demonstration` to save an `ArisOperationTemplate` into MongoDB (`brain_operation_templates`) tagged with `target_domain` and `hops`.
2. **Offline Re-Use (`--no-advisor`)**: In future inquiries:
   - AR.I.S checks its learned interrogative token associations to deduce the expected `target_domain` (e.g. `"where"` -> `"location"`).
   - Filters candidate templates by `target_domain` and relation presence in the query.
   - Binds the start entity anchor dynamically from question candidates.
   - Executes deterministic graph traversal completely offline without external API calls.

---

## 5. Verification & Audit Standards
- **Step 1 (Teacher Demonstration - 3-Hop Location)**:
  `python manage.py brain reason "Where does the brother of the leader of Axinix live?" -v`
  Learns and persists the 3-hop location template and records `"where"` $\to$ `"location"`.
- **Step 2 (Offline Multi-Hop - Location)**:
  `python manage.py brain reason "Where does the brother of the leader of Axinix live?" --no-advisor -v`
  Selects the 3-hop location template (rejecting 2-hop entity templates) and returns `Chennai`.
- **Step 3 (Offline Multi-Hop - Entity/Person)**:
  `python manage.py brain reason "Who is the brother of the leader of Axinix?" --no-advisor -v`
  Selects the 2-hop entity template and returns `Bala` and `Bharathikannan`.
