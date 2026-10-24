# Root Cause Engineering (Zero Hard Patches)

## Definition
A **hard patch** is any code change that treats the symptoms of an issue without addressing the fundamental reason the issue exists (e.g., catching an exception and silently swallowing it, adding an ad-hoc flag to bypass broken logic, or hardcoding return values for failing edge cases).

## Core Directives
1. **Never suppress without resolving**: Do not mask exceptions or errors with broad try/catch blocks that hide architectural flaws.
2. **Deep inquiry**: Determine why the state or contract was violated in the first place.
3. **Clean refactoring**: Fix the underlying schema, contract, or component lifecycle cleanly.
4. **Permanent robustness**: The system must be left stronger, simpler, and less prone to future failure.
