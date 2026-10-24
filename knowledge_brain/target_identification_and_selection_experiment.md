# Target Identification & Contextual Selection Experiment

## Objective
Empirically determine what the current AR.I.S linguistic substrate and cognitive pipeline can and cannot identify from text without external positional spoon-feeding:
1. Locate requested lexical tokens.
2. Distinguish between multiple occurrences of identical tokens.
3. Identify targets based on grammatical/relational roles.
4. Identify targets required by natural-language questions.

---

## Architectural Boundary Audit
The experiment evaluates the existing pipeline:
$$\text{Raw Language} \xrightarrow{\text{ArisCanonicalTokenizer}} \text{Token IDs} \xrightarrow{\text{ArisPhase0Model}} \text{Hidden States} \in \mathbb{R}^{B \times L \times D} \xrightarrow{\text{Relational / Cognitive Tier}} \text{Target Extraction}$$

- **Zero Test Orchestration**: No hardcoded expected positions supplied to models.
- **Zero Hacks**: Evaluated against real production implementations (`ArisCanonicalTokenizer`, `ArisPhase0Model`, `ArisRelationalBinder`, `ArisCognitiveReasoner`, `ArisGoalReasoningTensorEngine`).

---

## Measured Findings (Experiments A – J)

### Experiment A: Explicit Lexical Target
- Target: `apple`
- Inputs: `"I ate an apple."`, `"Apple released a phone."`, `"The apple fell from the tree."`, `"Yesterday I bought an apple."`, `"The farmer planted an apple tree near the house."`
- **Finding**: The lexical target is localized in the token sequence via `ArisCanonicalTokenizer` token spans (positions 4, 1, 2, 5, 5). However, **`ArisPhase0Model` does not identify the target token**—it computes contextual representations for all positions simultaneously. Target extraction relies on lexical localization.

### Experiment B: Multiple Occurrences
- Inputs:
  - `"I bought an apple and ate the apple."` $\to$ Occurrences at pos 4 and pos 8 ($\cos = 0.7542, d = 5.6086$).
  - `"The apple is beside another apple."` $\to$ Occurrences at pos 2 and pos 6 ($\cos = 0.5738, d = 7.3862$).
  - `"I saw an apple, picked the apple, and carried it home."` $\to$ Occurrences at pos 4 and pos 8.
- **Finding**: **AMBIGUOUS**. The architecture detects all candidate positions but possesses no internal mechanism to prefer one occurrence over another without external task conditioning.

### Experiment C: Grammatical Role Target
- `"Ravi helped Anita."` / `"Who did Ravi help?"` $\to$ `ArisRelationalBinder` binds `Subject: Ravi`, `Relation: helped`, `Object: Anita`.
- `"I ate an apple."` / `"What did I eat?"` $\to$ Non-capitalized common nouns are unextracted by default candidate filters without exemplar templates.
- **Finding**: Relational binding identifies syntactic roles for extracted entity candidates, but question-conditioned target extraction halts without an active `ArisOperationTemplate`.

### Experiment D: Same Entity, Different Requested Target
- Sentence: `"Google was founded in 1998 by Larry Page and Sergey Brin."`
- `ArisGoalReasoningTensorEngine` differentiates requested semantic dimensions:
  - `"Who founded Google?"` $\to$ `PERSON`
  - `"When was Google founded?"` $\to$ `TIME / EVENT`
  - `"What is Google?"` $\to$ `ENTITY / IDENTITY`
- **Finding**: The tensor substrate identifies the requested information dimension, but does not extract answer spans without a pre-compiled relational template.

### Experiment E: Same Entity, Different Relations
- Sentence: `"Ravi works at Axinix. Axinix is located in Coimbatore."`
- `"Where does Ravi work?"` $\to$ `Anchor: Ravi`, `Goal Type: GPE_CITY`
- `"Where is Axinix located?"` $\to$ `Anchor: Axinix`, `Goal Type: GPE_CITY`
- `"Who works at Axinix?"` $\to$ `Anchor: Axinix`, `Goal Type: PERSON`
- **Finding**: Preserves distinct relation anchors and semantic goal types in memory graphs, but requires learned template traversal for multi-hop graph resolution.

### Experiment F: Subject/Object Minimal Pairs
- F1: `"Ravi helped Anita."` $\to$ `Subject: Ravi`, `Object: Anita`
- F2: `"Anita helped Ravi."` $\to$ `Subject: Anita`, `Object: Ravi`
- **Finding**: **PASS**. Directional syntax is fully preserved: `Subject -> Relation -> Object` is structurally decoupled from `Object <- Relation <- Subject`.

### Experiment G: Unsupported Questions
- Context: `"I ate an apple."` / Q: `"What is the capital of France?"`
- Context: `"The dog is sleeping."` / Q: `"When was the dog born?"`
- **Finding**: **PASS**. Epistemic state evaluates to `ACQUIRABLE / UNSUPPORTED`. The system dispatches to external capability without hallucinating local entities.

### Experiment H: Ambiguous Pronoun Target
- Sentence: `"Ravi told Arun that he should leave."` / Q: `"Who should leave?"`
- **Finding**: **AMBIGUOUS**. Unconstrained deictic binding does not arbitrarily force either candidate.

### Experiment I: Position Independence
- Active: `"Ravi helped Anita."` (Ravi at pos 0)
- Passive: `"Anita was helped by Ravi."` (Ravi at pos 4)
- **Finding**: **PASS**. Relational binder tracks the participant across syntactic shifts.

---

## Capability Matrix

| Capability | Result |
| :--- | :--- |
| Token localization | **LIMITED** (Lexical scan only; not performed by Phase 0 neural substrate) |
| Position-independent localization | **PASS** (Locates tokens across arbitrary positions) |
| Multiple-occurrence handling | **AMBIGUOUS** (Detects all occurrences; cannot select without context) |
| Subject identification | **LIMITED** (Passes on proper-noun candidates; limited on common nouns) |
| Object identification | **LIMITED** (Passes on proper-noun candidates; limited on common nouns) |
| Relation-direction identification | **PASS** (Distinguishes directional argument roles) |
| Question-target identification | **LIMITED** (Neural goal type recognized; requires template to extract span) |
| Requested information dimension | **PASS** (Differentiates `PERSON`, `GPE_CITY`, `TIME`, etc.) |
| Contextual disambiguation | **AMBIGUOUS** (Does not force resolution on ambiguous pronouns) |
| Unsupported-question detection | **PASS** (Accurately flags non-derivable questions without hallucination) |

---

## Architecture Audit (Q1 – Q7)

- **Q1: Does `ArisPhase0Model` itself identify a target token?**
  **NO.** Phase 0 computes contextual representations for all positions. It contains no query head or pointer mechanism.
- **Q2: Does the current AR.I.S pipeline have a component that identifies the requested target?**
  **LIMITED.** Split across `ArisCanonicalTokenizer` (lexical spans), `ArisGoalReasoningTensorEngine` (semantic type), and `ArisRelationalBinder` (roles).
- **Q3: Does target identification depend on external lexical lookup?**
  **YES** for token-level localization.
- **Q4: Can the system distinguish multiple occurrences of the same word?**
  **NO / AMBIGUOUS.**
- **Q5: Can the system distinguish subject from object?**
  **YES / LIMITED.**
- **Q6: Can the system determine what information a question is requesting?**
  **YES.**
- **Q7: Can the system preserve the requested relation rather than collapsing to entity name?**
  **YES.**

---

## Conservative Scientific Conclusion
> **The current AR.I.S substrate produces contextual representations and can localize lexical tokens, but target selection is externally specified. The existing relational pipeline distinguishes subject and object roles for extracted candidates and identifies question target dimensions, but question-conditioned target-span extraction remains dependent on learned operation templates.**
