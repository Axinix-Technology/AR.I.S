# Deliberation Engine

## 1. Overview & Architectural Motivation
Traditional chatbots generate output through immediate, unverified auto-regressive decoding, leading to hallucinations and inability to reason across disparate facts. The AR.I.S Autonomous Deliberation Engine operates as a cognitive control loop:
1. **Deconstruction**: The system inspects incoming user queries and decomposes them into latent sub-queries or core entity references inside `<|thought|>` blocks.
2. **Self-Directed Retrieval**: AR.I.S accesses its internal Graph RAG memory to extract multi-hop paths and relational evidence.
3. **Evidence Synthesis**: Grounded facts from memory are injected into the neural attention context.
4. **Autonomous Response**: The language model decodes the answer with direct causal justification.
5. **Memory Consolidation**: Dialogue entities and verified conclusions are written back into the Graph Store, enabling cumulative lifelong memory.

## 2. Key Components
- `AutonomousDeliberationEngine` in `aris/core/deliberation.py`
- Neural inference with vectorized KV Caching
- Direct interface with `HybridGraphRetriever`

