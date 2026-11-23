# Foundational Data Purification & Uncapped Continuous Streaming Protocol

## 1. Overview & Directive Alignment
In strict accordance with the user's non-negotiable directive:
> *"First train the basic then add tools. Before able to speak, asking to sing is not valid."*

This document formalizes the complete elimination of artificial boilerplate templates, the restoration of pure natural language and human reasoning trajectories, and the establishment of uncapped continuous streaming across all foundational training corpora for AR.I.S **Gen 3** (63.59M parameters, 1,024 context window, Tokenizer V3).

---

## 2. Root-Cause Remediation: Elimination of Canned Boilerplate (Rule 1 & Rule 4)

Prior pipeline iterations suffered from procedural template injection where artificial, static prefixes were prepended to dataset targets to artificially force `<|thought|>` token emission. This caused parameter capacity waste and induced repetitive attractor loops.

### Audit of Purified Corpora
1. **GSM8K (Mathematical Reasoning)**:
   - *Previous Hardcode*: `resp = f"<|thought|>\nAnalyzing mathematical problem, variables, and computation steps:\n{a}\nFinal Calculation: {a.split('####')[-1].strip()}"`
   - *Root Cause Fix*: Stripped internal calculator evaluation tags (`<<48/2=24>>`) using regex `re.sub(r"<<.*?>>", "", a).strip()`, delivering pure human step-by-step mathematical reasoning terminating cleanly with `#### <number>`.
   - *Direct Benefit*: Model learns authentic arithmetic deduction without parrot boilerplate.

2. **MBPP (Algorithmic Logic & Python Syntax)**:
   - *Previous Hardcode*: `resp = f"<|thought|>\nFormulating algorithmic logic, edge cases, and unit tests.\n```python\n{code}\n```\n# Verification Assertions:\n{tests}"`
   - *Root Cause Fix*: Assistant response yields clean, executable Python functions: `resp = f"```python\n{code}\n```"`.
   - *Direct Benefit*: Teaches valid Python syntax and function signatures directly.

3. **Universal Dependencies (Grammatical Syntax & UPOS)**:
   - *Previous Hardcode*: `p = f"Analyze grammatical syntax and parts of speech for: {sent.text}"` and `resp = f"Universal POS breakdown: {upos_str}"`
   - *Root Cause Fix*: Neutral linguistic query: `p = f"Sentence: {sent.text}\nProvide the Universal POS tags for each word."` and `resp = upos_str`.

4. **Multi-Turn Dialogue (SmolTalk & UltraChat)**:
   - Streams raw natural conversational turns with zero wrapper modifications.

5. **Extractive Reading & Instruction (SQuAD v2 & Dolly-15k)**:
   - Streams pure contextual paragraphs, questions, instructions, and human answers.

---

## 3. Uncapped Continuous Streaming Protocol (Elimination of Step 7,000 Starvation)

### The Step 7,000 Defect
In Gen 2, small high-density corpora had hard 3-epoch limits:
- MBPP (374 samples) was exhausted by Step ~250.
- GSM8K (7,473 samples) was exhausted by Step ~4,850.
- Dolly-15k was exhausted by Step ~6,800.

Consequently, by Step 7,000, 100% of the active training stream consisted solely of context-copy QA (SQuAD) and chat, causing catastrophic forgetting of coding syntax and arithmetic deduction right as learning rates cooled.

### The Uncapped Streaming Protocol
- `epoch_caps` defaults to `None` / `{}`.
- Every corpus iterator executes:
  ```python
  while cap is None or epoch < cap:
      epoch += 1
      for f in files:
          rows = tbl.to_pylist()
          random.shuffle(rows) # Every epoch permutes pairings
          for r in rows:
              yield sample
  ```
- All 8 foundational streams (dialogue, instruction, QA, math, code, grammar) cycle continuously throughout the entire 15× training run (~954M tokens, ~29,114 optimizer steps).

---

## 4. Exact-Line Code Audit (Rule 7)

- **`aris/data/pipeline_v2.py`**:
  - `Line 14`: Added `import re`.
  - `Lines 61-62`: Replaced default 3-epoch dict with `self.epoch_caps = epoch_caps if epoch_caps is not None else {}` (uncapped streaming).
  - `Lines 206-224`: Replaced canned `"Analyzing mathematical problem..."` with `re.sub(r"<<.*?>>", "", a).strip()` and `while cap is None or epoch < cap:`.
  - `Lines 232-254`: Replaced canned `"Formulating algorithmic logic..."` with `resp = f"```python\n{code}\n```"` and `while cap is None or epoch < cap:`.
  - `Lines 323-325`: Replaced Dolly cap check with `while cap is None or epoch < cap:`.
  - `Lines 422-424`: Replaced canned `"Universal POS breakdown:"` with `Sentence: {text}\nProvide the Universal POS tags for each word.`
- **`aris/data/dataset.py`**:
  - `Line 11`: Added `import re`.
  - `Lines 211-305`: Replaced legacy boilerplate across QA-SRL, UD, GSM8K, MBPP, and Vision.

---

## 5. Verification & Code-Proof
- **Automated Unit Tests**: 24 of 24 tests passed cleanly in 48.91 seconds (`test_data_pipeline_v2.py`, `test_eval_harness_v2.py`, `test_loss_alignment.py`, `test_tokenizer_v2.py`, `test_tokenizer_v3.py`, `test_train_clean.py`).
