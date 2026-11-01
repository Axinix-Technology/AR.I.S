# Multi-Corpus Cognitive Architecture

## 1. Concept & Rationale
AR.I.S operates on the core principle of generic neural intelligence: avoiding all hardcoded procedural rules, keyword dictionaries, and toy handcrafted datasets. To develop robust language, syntax, semantic causal attribution, and multi-sentence deductions, AR.I.S trains across 4 world-standard corpora simultaneously.

## 2. Four Foundational Pillars
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

## 3. Dynamic Cognitive Interleaving Ratio
The streaming dataset interleaves batches in a 3:2:2:1 ratio:
- 3 turns UltraChat
- 2 multi-hop queries HotpotQA
- 2 causal queries QA-SRL
- 1 syntactic sentence Universal Dependencies

