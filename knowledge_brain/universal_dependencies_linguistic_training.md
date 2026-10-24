# Universal Dependencies (UD v2.18) Linguistic Training Substrate

## Overview & Architecture
In accordance with **Rule 1 (Zero Hard Patches)** and **Rule 4 (Generic Intelligence)**, AR.I.S does not parse or interpret syntactic structures through static word dictionaries or language-specific regex rules. Instead, universal linguistic representations are learned from standard **Universal Dependencies (UD v2.18)** multilingual treebanks and compiled into in-memory normalized PyTorch vector matrices.

---

## The Three Cognitive Substrate Layers

```text
                                  AR.I.S BRAIN
                                       │
      ┌────────────────────────────────┼────────────────────────────────┐
      ▼                                ▼                                ▼
1. Syntax & Grammar           2. Pragmatics & Dialogue        3. Structural Reasoning
[aris_ud_tensor_bank.pt]    [aris_pragmatic_tensor_bank.pt]   [aris_frame_classifier.pt]
   (models/aris_ud...)             (models/aris_pragmatic...)        (models/aris_frame...)
  Universal Dependencies       Speech acts, pronouns          Neural network predicting
  UPOS tags & relational       ('I', 'she', 'it') and         variable (?) vs bound
  dependencies.                wh-question targets.           anchor slots.
```

---

## 1. Streaming Treebank Ingestion (`ArisUDIngestionEngine`)
- **File:** [`api/linguistic/ud_ingestion_engine.py`](file:///e:/Loigmax/AR.I.S/api/linguistic/ud_ingestion_engine.py)
- Reads standard 10-column CoNLL-U format streamingly directly from compressed `Universal Dependencies 2.18/ud-treebanks-v2.18.tgz`.
- Extracts:
  - Token `FORM`, `LEMMA`, `UPOS`, `XPOS`, `FEATS`, `HEAD`, `DEPREL`.
  - Neutral dataclass `UDSentence` with words, lemmas, UPOS tags, and head-to-dependent relation links.

---

## 2. In-Memory Neural Metric Tensor Engine (`ArisUniversalLinguisticEncoder`)
- **File:** [`api/linguistic/ud_tensor_engine.py`](file:///e:/Loigmax/AR.I.S/api/linguistic/ud_tensor_engine.py)
- **Model:** [`models/aris_ud_tensor_bank.pt`](file:///e:/Loigmax/AR.I.S/models/aris_ud_tensor_bank.pt)
- Compiles continuous vector matrices across:
  - **17 UPOS Tags**: `PROPN`, `NOUN`, `VERB`, `AUX`, `PRON`, `ADP`, `ADJ`, `DET`, `ADV`, `NUM`, etc.
  - **60+ Dependency Relations**: `nsubj`, `obj`, `obl`, `cop`, `amod`, `compound`, `root`, `nmod:poss`, `appos`, `parataxis`.
  - **90+ Morphological Feature Representations**: `Person=1`, `Poss=Yes`, `Gender=Fem`, `Number=Sing`, `Tense=Past`.

---

## 3. Training CLI Command
Train the brain directly from terminal:
```bash
python manage.py brain train-ud --languages English Tamil French Hindi German --max-sentences 2000
```
- Operates 100% locally.
- Zero external runtime dependencies.
- Sub-millisecond vector similarity in memory.
