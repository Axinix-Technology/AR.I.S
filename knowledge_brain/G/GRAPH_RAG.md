# Graph RAG Memory

## 1. Overview & Architectural Motivation
Standard vector retrieval (naive RAG) computes cosine similarity over flat text chunks, completely blind to relationships, causal chains, and multi-hop connections. 

AR.I.S employs a **Hybrid Neuro-Symbolic Graph RAG** architecture combining:
1. **Symbolic Relational Graph**: Explicit knowledge triplets `(Source Entity, Predicate, Target Entity)` indexed for multi-hop graph path exploration up to 3 hops without cycle trapping.
2. **Dense Neural Embeddings**: Latent vectors derived from `ArisLM` representations, capturing semantic nuance and lexical variation.
3. **Reciprocal Rank Fusion (RRF)**: Merges dense similarity rankings with symbolic graph centrality and distance rankings using standard rank smoothing ($k = 60$):
   $$\text{RRF Score}(d) = \sum_{m \in M} \frac{w_m}{k + \text{rank}_m(d)}$$

## 2. Storage & Traversal Mechanics
- **Backend**: SQLite with compound indexes on `(source_id)`, `(target_id)`, and `(predicate)`.
- **Traversal**: BFS exploration with visited node sets guaranteeing sub-millisecond execution and no circular loops.
- **Context Generation**: Graph subgraphs and paths are serialized into structured factual sentences and supporting background context.

