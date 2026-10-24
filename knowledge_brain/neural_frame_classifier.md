# Neural Frame Classifier (Local De-Lexicalized PyTorch Engine)

## 1. Overview & Core Directive
AR.I.S incorporates a local, offline neural frame classifier (`ArisNeuralFrameLearner`) built on PyTorch.
- Consumes pure structural invariants (relative token position, normalized index, character length, casing, punctuation symbols).
- Learns to identify open question variable slots (`?`, `_`, `Who`, `What`, `Q`, `Z`, `X`) versus bound entity anchors (`A`, `B`, `Veltra`, `Zeta`).
- **Strict Invariant: ZERO ARITHMETIC IN TRAINING**: The curriculum generator (`ArisStructuralCurriculumGenerator`) contains zero numbers, zero arithmetic operators (`+`, `-`, `*`, `/`), and zero math formulas. It trains purely on arbitrary alien relational frames (`A daxels B`, `X qevron Y`).

---

## 2. Structural Vectorizer Specification (14 Invariants)
The input tensor consists of 14 non-lexical structural features extracted from `(subject, relation, object)`:
- `[0]`: Subject relative position in sentence (0.0 – 1.0)
- `[1]`: Subject token index normalized
- `[2]`: Subject length normalized
- `[3]`: Subject uppercase flag (1.0 / 0.0)
- `[4]`: Subject titlecase flag (1.0 / 0.0)
- `[5]`: Subject variable placeholder symbol flag (?, _, $, #)
- `[6]`: Subject single-character flag (1.0 / 0.0)
- `[7]`: Object relative position in sentence (0.0 – 1.0)
- `[8]`: Object token index normalized
- `[9]`: Object length normalized
- `[10]`: Object uppercase flag (1.0 / 0.0)
- `[11]`: Object titlecase flag (1.0 / 0.0)
- `[12]`: Object variable placeholder symbol flag (?, _, $, #)
- `[13]`: Object single-character flag (1.0 / 0.0)

---

## 3. Network Architecture & Performance
- **Framework**: PyTorch (`torch.nn.Module`).
- **Topology**: Feedforward network: `14 -> Linear(32) -> ReLU -> Linear(16) -> ReLU -> Linear(2)`.
- **Inference Speed**: Microseconds (< 1 ms), fully offline on CPU.
- **Model Persistence**: Saved to `models/aris_frame_classifier.pt`.
