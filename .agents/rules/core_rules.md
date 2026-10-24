# Core Rules & System Directives

These rules are non-negotiable and apply to every agent action in this workspace.

## 1. No Hard Patches
- Do not apply surface-level patches, quick hacks, or temporary bypasses.
- Identify and resolve the root cause cleanly and permanently.

## 2. Living Knowledge Brain with Alphabetical Indexing
- Every update, component, concept, or fix must be documented in `knowledge_brain/`.
- Maintain strict alphabetical order in `knowledge_brain/INDEX.md` for fast scanning and retrieval.

## 3. Knowledge Brain Pre-Verification
- Before implementing any requirement or fix, inspect `knowledge_brain/INDEX.md` and corresponding entries.
- Verify existing patterns, lessons learned, and system specifications.

## 4. Generic Cognitive Architecture (No Code-Based Orchestration)
- AR.I.S is an Artificial Intelligent System.
- Do not build hardcoded procedural orchestration pipelines or rigid code paths.
- Design solutions that empower generic, autonomous reasoning and flexible cognitive models.

## 5. Mandatory Implementation & Audit Plan
- Do not write or modify functional code without an implementation plan and verification/audit plan.
- Ensure all changes are code-proofed and validated.

## 6. Human-Understandable Language
- Keep all explanations, documentation, and comments crystal clear, descriptive, and human-readable.

## 7. Prohibition of Surface Guards & Mandatory Exact-Line Audit
- Never use character-level punctuation checks (e.g., `.endswith("?")`, `.startswith()`), string-slicing guards, or hardcoded stopword/keyword lists.
- All linguistic decisions must come from the neural tensor substrate (`CognitiveLinguisticFrame`).
- For every code edit, explicitly record and verify the exact file, exact line numbers, what changed, and why it changed.
