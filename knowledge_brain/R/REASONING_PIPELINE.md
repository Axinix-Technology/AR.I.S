# Reasoning Pipeline

## 1. Multi-Sentence & Multi-Hop Reasoning
Rather than producing immediate answers without internal deliberation, AR.I.S learns explicit reasoning trajectories via `<|thought|>` blocks.

### Mechanics in HotpotQA:
1. **Evidence Identification**:
   - For any question requiring multi-sentence comprehension, the supporting facts are isolated by title and sentence ID.
   - Example:
     ```
     Evidence 1: Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia.
     Evidence 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.
     ```
2. **Latent Synthesis**:
   - The model learns to link the dates and contexts across both supporting facts.
3. **Target Resolution**:
   - Final conclusion is generated conditioned on the verified supporting evidence.

## 2. Word-Level Semantic Causal Attribution
Using QA-SRL 2020, every sentence's actions are decomposed into functional semantic roles:
- **Predicate**: Target verb/action in the sentence.
- **Questions**: Structural questions exploring the action:
  - *Who performed it?* (Agent)
  - *What was affected?* (Patient/Theme)
  - *Where did it occur?* (Locative)
  - *When did it occur?* (Temporal)
  - *Why did it happen?* (Causal)
  - *How was it executed?* (Manner/Instrument)
- **Answers**: Continuous token spans directly grounded in the input sentence.

This avoids rigid symbol extractors and enables the neural model to internally compute causal relationships.

