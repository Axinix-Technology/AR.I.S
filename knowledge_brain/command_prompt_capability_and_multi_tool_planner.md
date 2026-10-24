# Command Prompt Capability & Deliberative Multi-Tool Planner

## Overview
Provides a decoupled execution tool (`command_prompt`) and an autonomous 3-step deliberative reasoning decomposition for multi-tool problem solving without hardcoded orchestration pipelines.

---

## Architectural Principles & Invariants

### 1. Decoupled Tool Execution (`ArisCommandPromptTool`)
- **Domain**: `system_terminal_execution`
- **Location**: `api/semantic/executable_tools.py`
- **Invariants**:
  - The tool is deterministic: it runs commands within a targeted working directory (`cwd`) using `subprocess.run()`.
  - The tool contains zero reasoning or domain heuristics.
  - Returns structured output: `{status, stdout, stderr, exit_code, command, working_dir}`.

### 2. Autonomous Multi-Tool Planning (`ArisReasoningPlanner`)
- When presented with a task requiring command execution on a target directory (e.g. `E:\Loigmax\tracker-v2`):
  - **Step 1 (`browser_access`)**: Research command syntax or operational contracts.
  - **Step 2 (`command_prompt`)**: Target and validate the working directory path.
  - **Step 3 (`command_prompt`)**: Execute the requested domain command (`git status`) and capture ground-truth system output.

### 3. Transparent Planner Output
- The deliberative planner prints every step of the decomposed plan, the selected capability for each step, and step execution status cleanly to stdout, guaranteeing complete transparency.

---

## Generic Executable Capability Template Transfer (Branch 1-B)

### Problem Solved (Checkpoint 12)
Learned `executable_capability` templates (e.g. `command_prompt`) were previously only reused when `capability_name == "math_computation"`. All other learned capabilities were invisible to the autonomous reasoning path.

### Fix Applied — 3 Surgical Edits to `api/cognitive/cognitive_learner.py`

| Edit | Location | What Changed |
|------|----------|--------------|
| Case C | `bind_executable_parameters` (~line 199) | Adds generic path extraction for schemas with `working_dir`. Extracts Windows/Unix paths from query, leaves unresolved params as `None`. |
| Branch 1-B | `reason()` (~line 752) | Scans ALL `executable_capability` templates before relational memory. On partial bind → returns `EPISTEMIC_DEFICIT`. On full bind → dispatches capability. |
| Branch 4 gate | `reason()` (~line 1387) | Widened from `ACQUIRABLE` only to also allow `DERIVABLE` so command_prompt templates fire via fallback path. |

### Turn 1 / Turn 2 Learning Pattern (Now Verified)
**Turn 1** (advisor active):
- Teacher shows AR.I.S: `command_prompt(command='dir /b /ad', working_dir='E:\Loigmax\AR.I.S')`
- AR.I.S executes, grounds the result, and persists template `op_executable_command_prompt`

**Turn 2** (advisor = False):
- Query: "What is the main branch status of this folder git: E:\Loigmax\tracker-v2?"
- Branch 1-B matches `op_executable_command_prompt`
- Binds `working_dir = E:\Loigmax\tracker-v2` ✅, detects `command = None` (EPISTEMIC_DEFICIT)
- Falls through to Browser Acquisition (Branch 4) to learn "git status / git branch" syntax
- Returns acquired knowledge — demonstrating autonomous knowledge gap detection + acquisition

### Test
- File: `api/tests/test_command_prompt_learning_and_transfer.py`
- Status: **PASSING** (1 passed in ~9s)
