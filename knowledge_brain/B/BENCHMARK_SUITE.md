# Cognitive Benchmark Suite

## 1. Overview & Purpose
To prevent capability regression and quantify neural learning over decades, AR.I.S uses an objective Stage-by-Stage Developmental Milestone Benchmark accessible via `python run.py eval`. Rather than evaluating static database lookups or simulated mocks, it directly executes live neural forward passes across 7 developmental milestone stages.

All datasets (Universal Dependencies archives, HotpotQA parquets, QA-SRL files, GSM8K, and MBPP) are dynamically auto-discovered on disk without hardcoded path dependencies.

## 2. The 7 Developmental Milestone Stages

### Stage 1: Lexical Grounding & Vocabulary Formation
- Evaluates whether the neural model generates recognizable language words rather than raw sub-word babbling.
- **Metrics**: 
  - **Word Validity Rate**: Percentage of generated tokens forming phonotactically valid words.
  - **Status Classification**: INFANT (Babbling), TODDLER (Emerging Words), MATURE (Fluent Lexicon).

### Stage 2: Universal Grammatical Syntax (Universal Dependencies)
- Evaluates whether the model has internalized the grammatical structure of human language.
- Probes: Part-of-speech tagging (`NOUN`, `VERB`, `ADJ`, `PRON`, `PUNCT`, etc.) streamed directly from Universal Dependencies CoNLL-U treebanks (`ud-treebanks-*.tgz`).
- **Metric**: **UPOS Alignment Rate**.

### Stage 3: Semantic Causal Attribution (QA-SRL 2020)
- Evaluates word-level causal reasoning answering *Who, What, Where, When, Why, How* for target predicate verbs.
- **Metrics**: **Causal Span F1** against ground-truth semantic spans.

### Stage 4: Multi-Hop Deduction (HotpotQA)
- Evaluates whether the model can connect cross-paragraph supporting facts to deduce final answers.
- **Metrics**: **Exact Match (EM)** and **Token F1**.

### Stage 5: Mathematical Problem Solving (GSM8K)
- Evaluates step-by-step arithmetic deduction and calculation chains on Grade School Math word problems.
- **Metrics**: **Solution Token F1** and **Final Numeric Accuracy**.

### Stage 6: Algorithmic Code Synthesis (MBPP)
- Evaluates foundational Python programming, algorithmic formulation, and syntax validity on Grade 1 MBPP problems.
- **Metrics**: **Code Token F1** and **Abstract Syntax Tree (AST) Validity Rate**.

### Stage 7: Live Qualitative Voice Probes
- Probes the model with live conversational, reasoning, coding, and persona inputs:
  - `"hello"` (Conversational greeting)
  - `"Who are you?"` (Cognitive persona & identity)
  - `"What is 15 + 27?"` (Arithmetic calculation)
  - `"Write a Python function to check if a number is even."` (Algorithmic synthesis)
  - `"What is your purpose?"` (Autonomous cognitive mission)
- Prints unedited live outputs, enabling direct human observation of voice evolution.

## 3. Usage & Execution
```bash
python run.py eval --samples 5
```
Options:
- `--samples <int>`: Number of samples to evaluate per stage (default: 10).
- `--checkpoint <path>`: Explicit model checkpoint (defaults to `models/aris_lm_checkpoint.pt`).
- `--tokenizer <path>`: Explicit tokenizer file (defaults to `models/aris_bpe_tokenizer.json`).
- Automatically locates `ud-treebanks-*.tgz`, `data/hotpotqa/**/*.parquet`, `data/qasrl_gs/**/*.jsonl.gz`, `data/gsm8k/**/*.parquet`, and `data/mbpp/**/*.parquet`.
