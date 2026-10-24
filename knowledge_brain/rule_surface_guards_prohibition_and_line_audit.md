# Rule 7: Prohibition of Surface Guards & Mandatory Exact-Line Audit

## Directives

1. **Absolute Prohibition of Surface/Character Guards**:
   - Under no circumstances may code use character-level punctuation checks (e.g. `.endswith("?")`, `.startswith()`), string-slicing guards, or hardcoded stopword/keyword lists.
   - All grammatical and speech-act properties (questions, commands, declaratives, nominals) must be derived from the neural tensor substrate (`CognitiveLinguisticFrame` via `ArisPragmaticTensorEngine` and `ArisUniversalLinguisticEncoder`).

2. **Mandatory Exact-Line Recording and Verification**:
   - For every single code modification, the exact target file and line numbers must be documented.
   - The rationale—what was changed and why it was changed—must be explicitly logged to guarantee context-window fidelity and prevent regressions or hot-patch reinjection across compaction boundaries.
