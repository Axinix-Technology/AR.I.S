# Cognitive Benchmark Suite

## 1. Overview & Purpose
To prevent capability regression and quantify neural learning over decades, AR.I.S uses an objective Stage-by-Stage Developmental Milestone Benchmark accessible via `python run.py eval`. Rather than evaluating static database lookups, it directly probes the neural model's language formation and cognitive reasoning across its 5 developmental stages.

## 2. The 5 Developmental Milestone Stages

### Stage 1: Lexical Grounding & Vocabulary Formation
- Evaluates whether the neural model generates recognizable language words rather than raw sub-word babbling.
- **Metrics**: 
  - **Word Validity Rate**: Percentage of generated tokens forming recognized words.
  - **Perplexity (PPL)**: Exponential of validation cross-entropy loss ($\text{PPL} = e^L$).

### Stage 2: Universal Grammatical Syntax (Universal Dependencies)
- Evaluates whether the model has internalized the grammatical structure of human language.
- Probes: Part-of-speech tagging (`NOUN`, `VERB`, `ADJ`, `PRON`, `PUNCT`, etc.) against held-out Universal Dependencies treebanks.
- **Metric**: **UPOS Alignment Rate**.

### Stage 3: Semantic Causal Attribution (QA-SRL 2020)
- Evaluates word-level causal reasoning answering *Who, What, Where, When, Why, How* for predicate actions.
- **Metrics**: **Span Recall** and **Token F1**.

### Stage 4: Multi-Hop Deduction (HotpotQA)
- Evaluates whether the model can connect supporting facts across multiple paragraphs to deduce answers.
- **Metrics**: **Exact Match (EM)** and **Token F1**.

### Stage 5: Live Voice Qualitative Preview
- Probes the model with standard conversational inputs (e.g. *"Hello, who are you?"*) and prints the unedited output, enabling human observation of voice evolution.

## 3. Usage
```bash
python run.py eval --samples 10
```
Outputs a standardized stage-by-stage scorecard tracking the child's linguistic progression from infancy to maturity.
