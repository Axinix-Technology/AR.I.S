# Problem Solving & Mathematical Reasoning

## 1. Overview & Purpose
General intelligence requires the capacity to decompose complex, multi-step queries into intermediate logical deductions rather than relying on shallow associative recall. In AR.I.S, the problem-solving substrate is powered by the **GSM8K** (Grade School Math 8.5K) corpus, created by OpenAI.

GSM8K focuses on linguistically articulated multi-step word problems requiring 2 to 8 operational steps, intermediate calculation verification, and final numerical extraction.

---

## 2. Dataset Mechanics & Sourcing

- **Source**: `openai/gsm8k` (`main/train-00000-of-00001.parquet`)
- **Location on Disk**: `data/gsm8k/train.parquet` (2.20 MB, 7,473 examples)
- **Structure**:
  - `question`: Natural language mathematical problem statement.
  - `answer`: Step-by-step reasoning chain with arithmetic annotations (`<<48/2=24>>`) concluded by the final target answer marked with `#### <value>`.

---

## 3. Cognitive Integration & Neural Tokenization

The data is streamed directly into ChatML format via `_iter_gsm8k()` in `EnterpriseArisDataset`:

```
<|im_start|>user
Solve the following mathematical problem step by step:
{question}
<|im_end|>
<|im_start|>assistant
<|thought|>
{step_by_step_reasoning}
Final Calculation: {final_answer}
<|im_end|>
```

### Key Substrate Benefits:
1. **Explicit Latent Reasoning (`<|thought|>`)**: Forces the model to allocate token capacity to intermediate deductive steps before emitting the conclusion.
2. **Numeric Precision Grounding**: Prevents hallucinated short-cuts by reinforcing causal arithmetic dependencies.
3. **Modular Cognitive Focus**: Activated independently via `--focus problem_solving`.

