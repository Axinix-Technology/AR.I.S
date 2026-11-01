# Coding Knowledge & Test-Driven Substrate

## 1. Overview & Purpose
Algorithmic reasoning and procedural formalization are essential pillars of autonomous intelligence. To ground AR.I.S with foundational "Grade 1" programming knowledge, we integrate Google Research's **MBPP** (Mostly Basic Python Problems) dataset.

MBPP introduces fundamental software engineering concepts: function definition, control flow, iterative loops, string manipulation, list comprehension, and test-driven validation.

---

## 2. Dataset Mechanics & Sourcing

- **Source**: `google-research-datasets/mbpp` (`full/train-00000-of-00001.parquet`, `prompt.parquet`, `test.parquet`)
- **Location on Disk**: `data/mbpp/*.parquet` (~964 total programming tasks)
- **Structure**:
  - `task_id`: Unique integer problem identifier.
  - `text`: Natural language programming task description.
  - `code`: Reference canonical Python implementation.
  - `test_list`: Formal unit test assertions (`assert func(input) == expected`).

---

## 3. Cognitive Integration & Neural Tokenization

The data is streamed directly into ChatML format via `_iter_mbpp()` in `EnterpriseArisDataset`:

```
<|im_start|>user
Write a Python program to solve this task:
{text}
<|im_end|>
<|im_start|>assistant
<|thought|>
Formulating algorithmic logic, edge cases, and unit tests.
```python
{code}
```
# Verification Assertions:
{assert_tests}
<|im_end|>
```

### Key Substrate Benefits:
1. **Syntax & AST Induction**: Directly grounds valid Python grammar, indentation, and function definitions without hardcoded syntax checkers.
2. **Test-Driven Self-Verification**: The presence of `assert` assertions trains the model to anticipate edge cases and output verifiable contracts.
3. **Modular Cognitive Focus**: Activated independently via `--focus coding`.

