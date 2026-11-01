# Multi-Corpus Cognitive Architecture

## 1. Concept & Rationale
AR.I.S operates on the core principle of generic neural intelligence: avoiding all hardcoded procedural rules, keyword dictionaries, and toy handcrafted datasets. To develop robust language, syntax, semantic causal attribution, multi-sentence deductions, algorithmic thinking, and visual perception, AR.I.S streams across **7 world-standard foundational corpora** on-the-fly.

## 2. Seven Foundational Pillars
1. **UltraChat 200k**:
   - Location: `data/ultrachat_200k/data/*.parquet`
   - Role: Teaches multi-turn dialogue flow, conversational persona, user engagement, and contextual responsiveness.
2. **Universal Dependencies v2.18**:
   - Location: `ud-treebanks-v2.18.tgz` (712 treebanks)
   - Role: Provides cross-linguistic grammatical structure, universal part-of-speech (UPOS) tagging, and syntactic dependencies.
3. **QA-SRL 2020 (Question-Answer Semantic Role Labeling)**:
   - Location: `data/qasrl_gs/qasrl-gs/*.jsonl.gz`
   - Role: Teaches word-level causal reasoning by grounding each predicate/verb in answerable questions (*Who, What, Where, When, Why, How*).
4. **HotpotQA**:
   - Location: `data/hotpotqa/distractor/*.parquet`
   - Role: Teaches multi-sentence, multi-hop reasoning. The system learns to identify supporting facts across distinct paragraphs and synthesize evidence to deduce the final answer.
5. **GSM8K (Grade School Math 8.5K)**:
   - Location: `data/gsm8k/train.parquet`
   - Role: Multi-step mathematical problem solving, intermediate calculation chains, and numeric deduction.
6. **MBPP (Mostly Basic Python Problems - Google Research)**:
   - Location: `data/mbpp/*.parquet`
   - Role: Foundational Grade 1 coding knowledge: functions, loops, control structures, and test-driven assertions (`assert`).
7. **LLaVA-Instruct (Multimodal Visual Reasoning)**:
   - Location: `data/vision/detail_23k.json`
   - Role: Visual scene comprehension, spatial relationships, lighting/color attributes, and image descriptions.

## 3. Dynamic Cognitive Interleaving Ratio
When `--focus all` is selected, the streaming dataset interleaves batches in a balanced cognitive ratio:
- 2 turns UltraChat (Conversational Dialogue)
- 1 sentence Universal Dependencies (Grammatical Syntax)
- 1 multi-hop query HotpotQA (Multi-Sentence Deduction)
- 1 causal query QA-SRL (Semantic Roles)
- 2 problems GSM8K (Mathematical Problem Solving)
- 2 tasks MBPP (Coding Knowledge & Assertions)
- 2 scenes LLaVA-Instruct (Vision & Spatial Understanding)

## 4. Modular Cognitive Focus Options
The training engine and dataset support modular domain specialization via `--focus`:
- `--focus all`: Full 7-pillar cognitive mixture.
- `--focus language`: UltraChat + Universal Dependencies.
- `--focus reasoning`: HotpotQA + QA-SRL.
- `--focus problem_solving`: GSM8K (100% focused).
- `--focus coding`: MBPP (100% focused).
- `--focus vision`: LLaVA-Instruct (100% focused).
