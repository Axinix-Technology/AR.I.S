"""
AR.I.S — Hybrid Neuro-Symbolic Graph Retriever (2026 Production Grade)
Pairs dense vector similarity with indexed multi-hop graph path exploration
using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations
import math
from typing import List, Dict, Any, Optional, Tuple
import torch

from .schema import EntityNode, RelationEdge, GraphPath
from .graph_store import ArisGraphStore
from ..tokenizer.bpe_tokenizer import ArisTokenizer
from ..models.aris_lm import ArisLM


class HybridGraphRetriever:
    """
    Hybrid Retriever integrating:
      1. Dense neural embeddings from ArisLM representations.
      2. Symbolic knowledge graph path exploration.
      3. Reciprocal Rank Fusion (RRF) for robust, un-hallucinated ranking.
    """

    def __init__(
        self,
        graph_store: ArisGraphStore,
        tokenizer: Optional[ArisTokenizer] = None,
        model: Optional[ArisLM] = None,
        device: Optional[torch.device] = None,
        rrf_k: int = 60,
    ):
        self.graph_store = graph_store
        self.tokenizer = tokenizer
        self.model = model
        self.device = device or torch.device("cpu")
        self.rrf_k = rrf_k

    def encode_text(self, text: str) -> Optional[List[float]]:
        """
        Derives dense embedding representation from model's token embeddings.
        """
        if not self.tokenizer or not self.model or not text.strip():
            return None

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            return None

        with torch.no_grad():
            token_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
            # Use model's token embedding table (tok_emb)
            embeds = self.model.tok_emb(token_tensor)  # (1, seq_len, embed_dim)
            # Mean pooling over tokens
            pooled = embeds.mean(dim=1).squeeze(0)  # (embed_dim,)
            # Normalize
            norm = torch.norm(pooled, p=2)
            if norm > 0:
                pooled = pooled / norm
            return pooled.cpu().tolist()

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        if len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 1.0,
        symbolic_weight: float = 1.0,
    ) -> List[Tuple[EntityNode, List[GraphPath], float]]:
        """
        Executes hybrid retrieval:
          1. Computes dense similarity ranking.
          2. Computes symbolic matching and centrality ranking.
          3. Fuses ranks via Reciprocal Rank Fusion (RRF).
          4. Returns top entities alongside their active multi-hop reasoning paths.
        """
        all_entities = self.graph_store.get_all_entities()
        if not all_entities:
            return []

        query_emb = self.encode_text(query)
        q_tokens = set(self.tokenizer.encode(query, add_special_tokens=False)) if self.tokenizer else set()

        # 1. Dense Scoring
        dense_scores: List[Tuple[str, float]] = []
        if query_emb:
            for ent in all_entities:
                if ent.embedding:
                    sim = self._cosine_similarity(query_emb, ent.embedding)
                else:
                    sim = 0.0
                dense_scores.append((ent.id, sim))
            dense_scores.sort(key=lambda x: x[1], reverse=True)

        dense_rank: Dict[str, int] = {e_id: r for r, (e_id, _) in enumerate(dense_scores)}

        # 2. Symbolic Scoring (Token overlap & in-degree centrality)
        symbolic_scores: List[Tuple[str, float]] = []
        for ent in all_entities:
            ent_tokens = set(self.tokenizer.encode(ent.name + " " + ent.description, add_special_tokens=False)) if self.tokenizer else set()
            overlap = len(q_tokens.intersection(ent_tokens)) / max(1, len(ent_tokens)) if ent_tokens else 0.0
            centrality = math.log1p(ent.access_count + len(self.graph_store.get_out_edges(ent.id)))
            sym_score = overlap * 2.0 + centrality * 0.1
            symbolic_scores.append((ent.id, sym_score))
        symbolic_scores.sort(key=lambda x: x[1], reverse=True)

        symbolic_rank: Dict[str, int] = {e_id: r for r, (e_id, _) in enumerate(symbolic_scores)}

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        entity_map = {e.id: e for e in all_entities}

        for ent_id in entity_map:
            score = 0.0
            if ent_id in dense_rank:
                score += dense_weight * (1.0 / (self.rrf_k + dense_rank[ent_id]))
            if ent_id in symbolic_rank:
                score += symbolic_weight * (1.0 / (self.rrf_k + symbolic_rank[ent_id]))
            rrf_scores[ent_id] = score

        # Top-K Entities
        sorted_ids = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:top_k]

        results = []
        for ent_id in sorted_ids:
            ent = entity_map[ent_id]
            # Explore multi-hop paths around top entities
            paths = self.graph_store.explore_subgraph([ent_id], max_hops=2, max_paths=3)
            results.append((ent, paths, rrf_scores[ent_id]))

        return results

    def build_context_narrative(self, query: str, top_k: int = 3) -> str:
        """
        Builds a structured factual context narrative from hybrid retrieval
        suitable for feeding directly into ArisLM's prompt context.
        """
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return ""

        context_lines: List[str] = []
        for ent, paths, _ in results:
            ent_summary = f"- [{ent.name}]: {ent.description}".strip() if ent.description else f"- [{ent.name}]"
            context_lines.append(ent_summary)
            for path in paths:
                p_text = path.format_factual_narrative()
                if p_text:
                    context_lines.append(f"  * Relational path: {p_text}")

        return "\n".join(context_lines)

