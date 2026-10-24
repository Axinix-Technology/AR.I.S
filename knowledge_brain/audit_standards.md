# Audit & Code-Proof Standards

## Purpose
Ensures that no code is written, changed, or deleted without a verified blueprint and an audit plan that proves safety, reliability, and correctness.

## Required Plan Structure Before Modification

Every update or bug fix must specify:
1. **Root-Cause Analysis**: What is the root cause? (No surface-level assumptions).
2. **Implementation Plan**: Step-by-step description of changes to be made.
3. **Generalization Check**: Does this make the system more generic and cognitive, or is it adding hardcoded procedural logic?
4. **Audit & Verification Steps**: How will this change be tested and audited to prove it introduces zero regressions and solves the issue permanently?

## Quality Bar
- Any code that fails the audit plan or introduces brittle assumptions must be rejected and redesigned.
