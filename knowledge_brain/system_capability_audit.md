# AR.I.S Whole-System Capability Audit

## 1. Audit Objective & Philosophy

The objective of this capability audit is to establish the **current real-world capability of AR.I.S as a complete, black-box system** when given natural inputs by a real user.

- We do NOT test individual components in isolation.
- We do NOT test Phase 0 in isolation.
- We do NOT orchestrate cognitive stages via Python scripts.
- We do NOT inject expected answers, mock memory, or backdoor relations.
- Every test is evaluated through the **real public user entry points** of AR.I.S.
- A failure is an essential scientific observation that accurately maps current architectural boundaries.

---

## 2. Real Public Entry Point Specification

AR.I.S is operated via the following public interfaces:

### Primary Terminal Command Interface: `manage.py brain`

```text
Real user entry point: Terminal CLI
Command:               python manage.py brain [store | reason | query | interactive]
Interpreter:           venv\Scripts\python.exe
Working Directory:     E:\Loigmax\AR.I.S
```

1. **Fact / Context Storage (`store`)**:
   - **Command**: `python manage.py brain store "<statement>" [--verbose]`
   - **Arguments**: `statement` (str), `--topic` (str, default 'general'), `--tags` (str), `--confidence` (float), `--verbose` (flag).
   - **Output Format**:
     ```text
     [AR.I.S Brain] Fact stored successfully!
       <statement>
       [ID: <fact_id> | Topic: <topic> | Tokens: <count> | Candidates: <count> | Relations: <count> | Tags: <tags>]
     ```

2. **Autonomous Reasoning & Question Answering (`reason`)**:
   - **Command**: `python manage.py brain reason "<question_text>" [--verbose] [--limit <N>]`
   - **Arguments**: `question_text` (str), `--limit` (int, default 10), `--verbose` (flag), `--no-advisor` (ablation flag).
   - **Output Format**:
     ```text
     [AR.I.S REASONING] Query      : <question_text>
     [AR.I.S REASONING] Advisor    : ON | Templates in memory: <N>
     [AR.I.S REASONING] Epistemic  : <KNOWN | DERIVABLE | ACQUIRABLE | UNSUPPORTED | EPISTEMIC_DEFICIT>
     [AR.I.S REASONING] Matched Cap: <capability>
     [AR.I.S REASONING] Goal Type  : <target_type> | Anchor: <anchor_entity>
     =================================================================
       AR.I.S AUTONOMOUS REASONING (TIER 5)
     =================================================================
     Question:       "<question_text>"
     Matches Found:  <count>
     -----------------------------------------------------------------
     Applied Template: <template_name> (<operation_type>)
     1. AUTONOMOUS ANSWER: <answer>
        Source:      "<source_statement>"
        Confidence:  <confidence>
     =================================================================
     ```

3. **1-Hop Relational Retrieval (`query`)**:
   - **Command**: `python manage.py brain query "<query_text>" [--verbose]`

4. **Multi-Hop Traversal (`traverse`)**:
   - **Command**: `python manage.py brain traverse "<start_entity>" --hops "<hops>" [--verbose]`

5. **Demonstration Learning (`learn`)**:
   - **Command**: `python manage.py brain learn --fact "<fact>" --question "<question>" --answer "<answer>" [--verbose]`

6. **Interactive Brain Console (`interactive`)**:
   - **Command**: `python manage.py brain interactive`
   - Prompt: `aris-brain> store <statement>` | `aris-brain> reason <question>` | `aris-brain> exit`

### Conversational Dialogue Agent Interface (`/api/chat/`)
- **HTTP REST Endpoint**: `POST http://localhost:8000/api/chat/`
- **Payload**: `{"session_id": "audit_session_1", "message": "<utterance>"}`
- **Web UI**: `http://localhost:8000/chat/ui/`

---

## 3. Test Code Instrument Specification

The test instrument is located at [audit_capability_instrument.py](file:///e:/Loigmax/AR.I.S/scripts/audit_capability_instrument.py).

### Integrity Guarantees
1. **Zero Cognitive Imports**: The script imports only standard Python libraries (`sys`, `os`, `subprocess`, `time`, `argparse`). It imports zero modules from `api.cognitive.*`, `api.relational.*`, or `api.linguistic.*`.
2. **Subprocess Boundary**: Commands are executed strictly through `subprocess.run` calling `venv\Scripts\python.exe manage.py brain ...`.
3. **Zero Answer/Knowledge Injection**: No facts or answers are injected into database or memory structures by the script.
4. **Zero Auto-Grading**: The script never declares PASS or FAIL. It only captures raw stdout, stderr, exit code, and execution time.
5. **Single-Run Execution**: The script operates one test at a time (`--test <ID>` or `--run "<CMD>"`) to prevent automated loops.

---

## 4. Test Verification Checklist (Section 25)

Before capability execution begins, the human auditor and agent verify:

- [x] **Does it call only the real AR.I.S public interface?** Yes (`manage.py brain` via subprocess).
- [x] **Does it avoid internal component orchestration?** Yes (zero internal components called).
- [x] **Does it avoid answer injection?** Yes (zero answer injection).
- [x] **Does it avoid knowledge injection?** Yes (no test mocks or memory backdoors).
- [x] **Does it avoid hardcoded expected answers in execution logic?** Yes (no `if answer == ...` in execution path).
- [x] **Does it avoid automatic retries?** Yes (executes once and records raw output).
- [x] **Does it avoid fallback answers?** Yes (zero fallback logic).
- [x] **Does it avoid calling tools on behalf of AR.I.S?** Yes (no manual tool triggers).
- [x] **Does it capture the raw result without transforming it?** Yes (stdout and stderr preserved verbatim).
- [x] **Can the entire capability audit be performed manually?** Yes (every command is executable directly in console).

---

## 5. Manual Execution Protocol

Each capability scenario must be executed manually.

### Execution Procedure:
1. Run the test command directly via terminal or using the observation instrument:
   ```bash
   python scripts/audit_capability_instrument.py --test <TEST_ID>
   ```
   or run the AR.I.S CLI directly:
   ```bash
   venv\Scripts\python.exe manage.py brain reason "<question>" --verbose
   ```
2. Copy the complete raw AR.I.S output into the test card.
3. Observe the behavior and record without altering the input.
4. Assign one allowed result:
   - `PASS`
   - `PARTIAL`
   - `FAIL`
   - `UNSUPPORTED`
   - `AMBIGUOUS`
   - `NOT_DERIVABLE`
   - `CONFLICT`

---

## 6. Capability Test Cards (A through R)

### CAPABILITY GROUP A — BASIC LANGUAGE UNDERSTANDING

#### TEST ID: A1
- **GROUP**: Group A: Basic Language Understanding
- **INPUT**: `Ravi works at Axinix.`
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Ravi works at Axinix." --verbose
  ```
- **EXPECTED BEHAVIOR**: AR.I.S represents and stores the fact through its normal architecture (tokens, candidates, relational bindings).
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: A2
- **GROUP**: Group A: Basic Language Understanding
- **INPUT**: `Anita manages the engineering team.`
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Anita manages the engineering team." --verbose
  ```
- **EXPECTED BEHAVIOR**: AR.I.S handles a different relation through normal architecture.
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: A3
- **GROUP**: Group A: Basic Language Understanding
- **INPUT**:
  ```text
  Context: Ravi works at Axinix.
  Context: Axinix is located in Coimbatore.
  Question: Where is Axinix located?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Ravi works at Axinix."
  venv\Scripts\python.exe manage.py brain store "Axinix is located in Coimbatore."
  venv\Scripts\python.exe manage.py brain reason "Where is Axinix located?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Coimbatore`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

### CAPABILITY GROUP B — QUESTION → RELEVANT RELATION → ANSWER

#### TEST ID: B1
- **GROUP**: Group B: Question -> Relevant Relation -> Answer
- **INPUT**:
  ```text
  Context: Ravi helped Anita.
  Question: Who helped Anita?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Ravi helped Anita."
  venv\Scripts\python.exe manage.py brain reason "Who helped Anita?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Ravi`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: B2
- **GROUP**: Group B: Question -> Relevant Relation -> Answer
- **INPUT**:
  ```text
  Context: Ravi helped Anita.
  Question: Who did Ravi help?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Ravi helped Anita."
  venv\Scripts\python.exe manage.py brain reason "Who did Ravi help?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Anita` (opposite end of the same relationship)
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: B3
- **GROUP**: Group B: Question -> Relevant Relation -> Answer
- **INPUT**:
  ```text
  Context: Ravi works at Axinix.
  Question: Where does Ravi work?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Ravi works at Axinix."
  venv\Scripts\python.exe manage.py brain reason "Where does Ravi work?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Axinix`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: B4
- **GROUP**: Group B: Question -> Relevant Relation -> Answer
- **INPUT**:
  ```text
  Context: Ravi works at Axinix.
  Question: Who works at Axinix?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Ravi works at Axinix."
  venv\Scripts\python.exe manage.py brain reason "Who works at Axinix?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Ravi`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

### CAPABILITY GROUP C — MULTIPLE IDENTICAL OCCURRENCES

#### TEST ID: C1
- **GROUP**: Group C: Multiple Identical Occurrences
- **INPUT**:
  ```text
  Context: Ravi bought an apple and ate the apple.
  Question: What did Ravi buy?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Ravi bought an apple and ate the apple."
  venv\Scripts\python.exe manage.py brain reason "What did Ravi buy?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `apple` (with relational evidence confirming the `buy` relation)
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: C2
- **GROUP**: Group C: Multiple Identical Occurrences
- **INPUT**:
  ```text
  Context: Ravi bought an apple and ate the apple.
  Question: What did Ravi eat?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Ravi bought an apple and ate the apple."
  venv\Scripts\python.exe manage.py brain reason "What did Ravi eat?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `apple` (with relational evidence confirming the `eat` relation rather than `buy`)
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: C3
- **GROUP**: Group C: Multiple Identical Occurrences
- **INPUT**:
  ```text
  Context: Ravi told Anita that Anita should call Arun.
  Question: Who should call Arun?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Ravi told Anita that Anita should call Arun."
  venv\Scripts\python.exe manage.py brain reason "Who should call Arun?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Anita` (correctly resolving the second occurrence in embedded clause)
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

### CAPABILITY GROUP D — SUBJECT / OBJECT

#### TEST ID: D1
- **GROUP**: Group D: Subject / Object
- **INPUT**:
  ```text
  Context: Ravi helped Anita.
  Question: Who helped Anita?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Ravi helped Anita."
  venv\Scripts\python.exe manage.py brain reason "Who helped Anita?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Ravi`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: D2
- **GROUP**: Group D: Subject / Object
- **INPUT**:
  ```text
  Context: Ravi helped Anita.
  Question: Who did Ravi help?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Ravi helped Anita."
  venv\Scripts\python.exe manage.py brain reason "Who did Ravi help?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Anita`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: D3
- **GROUP**: Group D: Subject / Object
- **INPUT**:
  ```text
  Context: Anita helped Ravi.
  Question: Who helped Ravi?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Anita helped Ravi."
  venv\Scripts\python.exe manage.py brain reason "Who helped Ravi?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Anita`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: D4
- **GROUP**: Group D: Subject / Object
- **INPUT**:
  ```text
  Context: Anita helped Ravi.
  Question: Who did Anita help?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Anita helped Ravi."
  venv\Scripts\python.exe manage.py brain reason "Who did Anita help?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Ravi`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

### CAPABILITY GROUP E — ACTIVE / PASSIVE

#### TEST ID: E1
- **GROUP**: Group E: Active / Passive
- **INPUT**:
  ```text
  Context: Ravi helped Anita.
  Question: Who helped Anita?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Ravi helped Anita."
  venv\Scripts\python.exe manage.py brain reason "Who helped Anita?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Ravi`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: E2
- **GROUP**: Group E: Active / Passive
- **INPUT**:
  ```text
  Context: Anita was helped by Ravi.
  Question: Who helped Anita?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Anita was helped by Ravi."
  venv\Scripts\python.exe manage.py brain reason "Who helped Anita?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Ravi`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: E3
- **GROUP**: Group E: Active / Passive
- **INPUT**:
  ```text
  Context: Anita was helped by Ravi.
  Question: Who received help from Ravi?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Anita was helped by Ravi."
  venv\Scripts\python.exe manage.py brain reason "Who received help from Ravi?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Anita`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

### CAPABILITY GROUP F — SAME ENTITY, DIFFERENT QUESTION

#### TEST ID: F1
- **GROUP**: Group F: Same Entity, Different Question
- **INPUT**:
  ```text
  Context: Google was founded in 1998 by Larry Page and Sergey Brin.
  Question: What is Google?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Google was founded in 1998 by Larry Page and Sergey Brin."
  venv\Scripts\python.exe manage.py brain reason "What is Google?" --verbose
  ```
- **EXPECTED BEHAVIOR**: Entity definition or UNSUPPORTED/ACQUIRABLE if not derivable from context.
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: F2
- **GROUP**: Group F: Same Entity, Different Question
- **INPUT**:
  ```text
  Context: Google was founded in 1998 by Larry Page and Sergey Brin.
  Question: When was Google founded?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Google was founded in 1998 by Larry Page and Sergey Brin."
  venv\Scripts\python.exe manage.py brain reason "When was Google founded?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `1998`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: F3
- **GROUP**: Group F: Same Entity, Different Question
- **INPUT**:
  ```text
  Context: Google was founded in 1998 by Larry Page and Sergey Brin.
  Question: Who founded Google?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Google was founded in 1998 by Larry Page and Sergey Brin."
  venv\Scripts\python.exe manage.py brain reason "Who founded Google?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Larry Page and Sergey Brin`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: F4
- **GROUP**: Group F: Same Entity, Different Question
- **INPUT**:
  ```text
  Context: Google was founded in 1998 by Larry Page and Sergey Brin.
  Question: Where was Google founded?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Google was founded in 1998 by Larry Page and Sergey Brin."
  venv\Scripts\python.exe manage.py brain reason "Where was Google founded?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `NOT_DERIVABLE` / `UNSUPPORTED` (information not in context).
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

### CAPABILITY GROUP G — MULTI-HOP REASONING

#### TEST ID: G1
- **GROUP**: Group G: Multi-Hop Reasoning
- **INPUT**:
  ```text
  Context:
    Ravi works at Axinix.
    Axinix is located in Coimbatore.
    Coimbatore is in Tamil Nadu.
  Question: Which state is the company Ravi works at located in?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Ravi works at Axinix."
  venv\Scripts\python.exe manage.py brain store "Axinix is located in Coimbatore."
  venv\Scripts\python.exe manage.py brain store "Coimbatore is in Tamil Nadu."
  venv\Scripts\python.exe manage.py brain reason "Which state is the company Ravi works at located in?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Tamil Nadu` via path: Ravi -> works_at -> Axinix -> located_in -> Coimbatore -> located_in -> Tamil Nadu.
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

### CAPABILITY GROUP H — PARAPHRASE

#### TEST ID: H1
- **GROUP**: Group H: Paraphrase
- **INPUT**:
  ```text
  Context: Ravi works at Axinix.
  Question: Where does Ravi work?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Ravi works at Axinix."
  venv\Scripts\python.exe manage.py brain reason "Where does Ravi work?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Axinix`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: H2
- **GROUP**: Group H: Paraphrase
- **INPUT**:
  ```text
  Context: Ravi works at Axinix.
  Question: Which company does Ravi work for?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Ravi works at Axinix."
  venv\Scripts\python.exe manage.py brain reason "Which company does Ravi work for?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Axinix`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: H3
- **GROUP**: Group H: Paraphrase
- **INPUT**:
  ```text
  Context: Ravi works at Axinix.
  Question: What company employs Ravi?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Ravi works at Axinix."
  venv\Scripts\python.exe manage.py brain reason "What company employs Ravi?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Axinix`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: H4
- **GROUP**: Group H: Paraphrase
- **INPUT**:
  ```text
  Context: Ravi works at Axinix.
  Question: Ravi is employed by which company?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Ravi works at Axinix."
  venv\Scripts\python.exe manage.py brain reason "Ravi is employed by which company?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Axinix`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

### CAPABILITY GROUP I — UNSEEN ENTITIES

#### TEST ID: I1
- **GROUP**: Group I: Unseen Entities
- **INPUT**:
  ```text
  Context: Kiran helped Meena.
  Question: Who did Kiran help?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Kiran helped Meena."
  venv\Scripts\python.exe manage.py brain reason "Who did Kiran help?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Meena`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: I2
- **GROUP**: Group I: Unseen Entities
- **INPUT**:
  ```text
  Context: Devika helped Arjun.
  Question: Who did Devika help?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Devika helped Arjun."
  venv\Scripts\python.exe manage.py brain reason "Who did Devika help?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Arjun`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

### CAPABILITY GROUP J — MULTI-RELATION CONTEXT

#### TEST ID: J1
- **GROUP**: Group J: Multi-Relation Context
- **INPUT**:
  ```text
  Context:
    Ravi works at Axinix.
    Axinix is located in Coimbatore.
    Ravi manages the engineering team.
    Anita works with Ravi.
  Question: Where does Ravi work?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Ravi works at Axinix."
  venv\Scripts\python.exe manage.py brain store "Axinix is located in Coimbatore."
  venv\Scripts\python.exe manage.py brain store "Ravi manages the engineering team."
  venv\Scripts\python.exe manage.py brain store "Anita works with Ravi."
  venv\Scripts\python.exe manage.py brain reason "Where does Ravi work?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Axinix`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: J2
- **GROUP**: Group J: Multi-Relation Context
- **INPUT**: `Question: Where is Axinix located?`
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain reason "Where is Axinix located?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Coimbatore`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: J3
- **GROUP**: Group J: Multi-Relation Context
- **INPUT**: `Question: Who manages the engineering team?`
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain reason "Who manages the engineering team?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Ravi`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: J4
- **GROUP**: Group J: Multi-Relation Context
- **INPUT**: `Question: Who works with Ravi?`
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain reason "Who works with Ravi?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Anita`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

### CAPABILITY GROUP K — AMBIGUITY

#### TEST ID: K1
- **GROUP**: Group K: Ambiguity
- **INPUT**:
  ```text
  Context: Ravi told Arun that he should leave.
  Question: Who should leave?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Ravi told Arun that he should leave."
  venv\Scripts\python.exe manage.py brain reason "Who should leave?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `AMBIGUOUS` (AR.I.S should not arbitrarily select Ravi or Arun without evidence).
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

### CAPABILITY GROUP L — MISSING INFORMATION

#### TEST ID: L1
- **GROUP**: Group L: Missing Information
- **INPUT**:
  ```text
  Context: Ravi works at Axinix.
  Question: What is the population of Coimbatore?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Ravi works at Axinix."
  venv\Scripts\python.exe manage.py brain reason "What is the population of Coimbatore?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `NOT_DERIVABLE FROM PROVIDED CONTEXT` / `UNSUPPORTED`. If external acquisition triggered, record exact behavior.
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

### CAPABILITY GROUP M — CONFLICTING INFORMATION

#### TEST ID: M1
- **GROUP**: Group M: Conflicting Information
- **INPUT**:
  ```text
  Context:
    Ravi works at Axinix.
    Ravi works at NovaTech.
  Question: Where does Ravi work?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Ravi works at Axinix."
  venv\Scripts\python.exe manage.py brain store "Ravi works at NovaTech."
  venv\Scripts\python.exe manage.py brain reason "Where does Ravi work?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `CONFLICT / MULTIPLE CANDIDATES / AMBIGUOUS` (not arbitrarily forcing one).
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

### CAPABILITY GROUP N — GENERALIZATION

#### TEST ID: N1
- **GROUP**: Group N: Generalization
- **INPUT**:
  ```text
  Context: Devika is employed by Orion Systems.
  Question: Which organization employs Devika?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Devika is employed by Orion Systems."
  venv\Scripts\python.exe manage.py brain reason "Which organization employs Devika?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Orion Systems`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: N2
- **GROUP**: Group N: Generalization
- **INPUT**:
  ```text
  Context: Karthik is employed by BluePeak.
  Question: Where does Karthik work?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Karthik is employed by BluePeak."
  venv\Scripts\python.exe manage.py brain reason "Where does Karthik work?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `BluePeak`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

### CAPABILITY GROUP O — EXTERNAL KNOWLEDGE ACQUISITION

#### TEST ID: O1
- **GROUP**: Group O: External Knowledge Acquisition
- **INPUT**: `What is Google?`
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain reason "What is Google?" --verbose
  ```
- **EXPECTED BEHAVIOR**:
  ```text
  Goal identified:
  Capability selected:
  External acquisition performed:
  Raw acquired result:
  Final AR.I.S answer:
  ```
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

### CAPABILITY GROUP P — PROCEDURAL / ACTION CAPABILITY

#### TEST ID: P1
- **GROUP**: Group P: Procedural / Action Capability
- **INPUT**: `Create a text file named aris_test.txt containing the text "ARIS test".`
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain reason "Create a text file named aris_test.txt containing the text \"ARIS test\"." --verbose
  ```
- **EXPECTED BEHAVIOR**:
  ```text
  Goal:
  Capability selected:
  Action executed:
  Observed result:
  Verification:
  ```
  If not supported via CLI interface, record `NOT DEMONSTRATED`.
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

### CAPABILITY GROUP Q — MEMORY

#### TEST ID: Q1
- **GROUP**: Group Q: Memory
- **INPUT**:
  ```text
  Teach: Arun's favorite development language is Python.
  Ask: What is Arun's favorite development language?
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Arun's favorite development language is Python."
  venv\Scripts\python.exe manage.py brain reason "What is Arun's favorite development language?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Python`
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

#### TEST ID: Q2
- **GROUP**: Group Q: Memory
- **INPUT**: `Question: Which programming language does Arun prefer?`
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain reason "Which programming language does Arun prefer?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Python` (memory retrieves same underlying information under paraphrase)
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

### CAPABILITY GROUP R — LEARNING / REUSE

#### TEST ID: R1
- **GROUP**: Group R: Learning / Reuse
- **INPUT**:
  ```text
  Teach: "Arun is brother of Bala." -> Question: "Who is brother of Bala?" -> Answer: "Arun"
  Test: "Who is brother of Qorin?" (with stored fact "Kiran is brother of Qorin.")
  ```
- **AR.I.S COMMAND**:
  ```bash
  venv\Scripts\python.exe manage.py brain store "Kiran is brother of Qorin."
  venv\Scripts\python.exe manage.py brain learn --fact "Arun is brother of Bala." --question "Who is brother of Bala?" --answer "Arun" --verbose
  venv\Scripts\python.exe manage.py brain reason "Who is brother of Qorin?" --verbose
  ```
- **EXPECTED BEHAVIOR**: `Kiran` (reusing the learned reverse traversal template)
- **RAW AR.I.S OUTPUT**: `[Pending Manual Execution]`
- **OBSERVED BEHAVIOR**: `[Pending Manual Execution]`
- **RESULT**: `[Pending Manual Execution]`

---

## 7. Final Capability Report (Section 26)

*To be populated upon completion of manual audit execution.*

| Capability | Status |
| :--- | :--- |
| Basic language understanding | `[Pending Manual Execution]` |
| Entity identification | `[Pending Manual Execution]` |
| Relation identification | `[Pending Manual Execution]` |
| Subject/object understanding | `[Pending Manual Execution]` |
| Multiple identical occurrence resolution | `[Pending Manual Execution]` |
| Question understanding | `[Pending Manual Execution]` |
| Question → relevant relation → answer | `[Pending Manual Execution]` |
| Multi-hop reasoning | `[Pending Manual Execution]` |
| Active/passive understanding | `[Pending Manual Execution]` |
| Paraphrase handling | `[Pending Manual Execution]` |
| Unseen entity generalization | `[Pending Manual Execution]` |
| Relation generalization | `[Pending Manual Execution]` |
| Ambiguity handling | `[Pending Manual Execution]` |
| Conflict handling | `[Pending Manual Execution]` |
| Missing-information handling | `[Pending Manual Execution]` |
| External knowledge acquisition | `[Pending Manual Execution]` |
| Tool execution | `[Pending Manual Execution]` |
| Memory retrieval | `[Pending Manual Execution]` |
| Learning/reuse | `[Pending Manual Execution]` |

*Allowed statuses: `DEMONSTRATED`, `PARTIALLY DEMONSTRATED`, `NOT DEMONSTRATED`, `FAILS`.*

---

## 8. Final AR.I.S Capability Statement (Section 27)

*To be written based strictly on the manual audit results.*

```text
AR.I.S CURRENTLY CAN:
- [To be filled from manual audit results]

AR.I.S CURRENTLY CAN PARTIALLY:
- [To be filled from manual audit results]

AR.I.S CURRENTLY CANNOT YET DEMONSTRATE:
- [To be filled from manual audit results]

AR.I.S WAS NOT TESTED FOR:
- [To be filled from manual audit results]
```
