"""
AR.I.S — Semantic Relation Abstraction Engine

Learns and infers mappings from diverse surface-language expressions to canonical
semantic relations using continuous prototype-centroid coordinate embeddings.

Strict Non-Negotiable Directives:
  - ZERO hardcoded phrase dictionaries (Rule 1 & Rule 4).
  - Prototype-Centroid Learning: Each canonical relation is represented as a normalized
    64-dimensional coordinate vector formed by the centroid of its base concept and
    demonstrated surface exemplars.
  - Multi-Scale Subword Projections: ArisSemanticEncoder captures character unigrams,
    bigrams, trigrams, and full words, enabling robust generalization to inflections,
    subword variations, and spelling noise.
  - Contextual Collision Prevention: Full predicate n-grams (e.g. 'works at' vs 'works with')
    prevent relational collisions between distinct relations sharing a common lexical stem.
  - Decoupled & Inspectable: Emits ranked candidate relations and confidence scores.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import math

try:
    from api.semantic.semantic_representation import ArisSemanticEncoder
except ImportError:
    from ..semantic.semantic_representation import ArisSemanticEncoder


class ArisRelationAbstractionEngine:
    """
    Continuous Coordinate Space Relation Abstraction & Generalization Engine.
    Supports both legacy semantic hashing and Phase 1 canonical neural substrate encoding.
    """

    def __init__(
        self,
        canonical_relations: Optional[List[str]] = None,
        encoder_fn: Optional[Callable[[str], List[float]]] = None,
        dimension: Optional[int] = None,
    ):
        self.encoder_fn = encoder_fn
        self.dimension = dimension or (64 if encoder_fn is None else None)
        self.relations: List[str] = []
        self.prototypes: Dict[str, List[float]] = {}
        self.prototype_counts: Dict[str, int] = {}
        self.graph_pair_priors: Dict[Tuple[str, str], str] = {}

        if canonical_relations:
            for rel in canonical_relations:
                self.register_relation(rel)

    @classmethod
    def from_neural_substrate(
        cls,
        tokenizer: Any,
        substrate_model: Any,
        canonical_relations: Optional[List[str]] = None,
    ) -> "ArisRelationAbstractionEngine":
        """
        Phase 2: Factory constructing an ArisRelationAbstractionEngine backed by
        the Phase 1 ArisCanonicalTokenizer and Phase 0 ArisPhase0Model.
        """
        import torch

        def neural_encode(phrase: str) -> List[float]:
            cleaned = phrase.strip().lower()
            if not cleaned:
                return [0.0] * substrate_model.embed_dim
            seq = tokenizer.serialize(cleaned, source="relation_phrase", add_special_tokens=False)
            with torch.no_grad():
                out = substrate_model(seq.input_ids, attention_mask=seq.attention_mask)
                hs = out["hidden_states"]  # (1, L, D)
                pooled = hs.squeeze(0).mean(dim=0)
                norm = torch.norm(pooled, p=2)
                if norm > 1e-8:
                    pooled = pooled / norm
                return pooled.cpu().tolist()

        return cls(
            canonical_relations=canonical_relations,
            encoder_fn=neural_encode,
            dimension=substrate_model.embed_dim,
        )

    @classmethod
    def from_trained_ud_substrate(
        cls,
        tokenizer: Any,
        ud_encoder: Optional[Any] = None,
        substrate_model: Optional[Any] = None,
        canonical_relations: Optional[List[str]] = None,
    ) -> "ArisRelationAbstractionEngine":
        """
        Constructs an ArisRelationAbstractionEngine backed by the Phase 1 canonical tokenizer,
        Phase 0 neural substrate interface, and the trained UD Brain (models/aris_ud_tensor_bank.pt).
        Produces learned continuous linguistic coordinates in R^64.
        """
        import torch
        if ud_encoder is None:
            try:
                from api.linguistic.ud_tensor_engine import ArisUniversalLinguisticEncoder
                ud_encoder = ArisUniversalLinguisticEncoder
            except ImportError:
                from ..linguistic.ud_tensor_engine import ArisUniversalLinguisticEncoder
                ud_encoder = ArisUniversalLinguisticEncoder

        def trained_ud_encode(phrase: str) -> List[float]:
            cleaned = phrase.strip().lower()
            if not cleaned:
                return [0.0] * 64
            seq = tokenizer.serialize(cleaned, source="relation_phrase", add_special_tokens=False)
            res = ud_encoder.encode_tokenized_sequence(seq, substrate_model=substrate_model)
            hs = res["hidden_states"]  # (1, L, 64)
            pooled = hs.squeeze(0).mean(dim=0)
            norm = torch.norm(pooled, p=2)
            if norm > 1e-8:
                pooled = pooled / norm
            return pooled.cpu().tolist()

        return cls(
            canonical_relations=canonical_relations,
            encoder_fn=trained_ud_encode,
            dimension=64,
        )

    def encode_phrase(self, phrase: str) -> List[float]:
        """
        Encodes a phrase into a continuous coordinate vector.
        Delegates to neural encoder_fn when present; falls back to ArisSemanticEncoder.
        """
        if self.encoder_fn is not None:
            vec = self.encoder_fn(phrase)
            if self.dimension is None:
                self.dimension = len(vec)
            return vec
        return ArisSemanticEncoder.encode(phrase)

    def register_relation(self, rel: str):
        """Dynamically registers a canonical relation and initializes its prototype centroid."""
        if rel and rel not in self.relations:
            self.relations.append(rel)
            natural_phrase = rel.replace("_", " ")
            vec = self.encode_phrase(natural_phrase)
            if self.dimension is None:
                self.dimension = len(vec)
            self.prototypes[rel] = vec
            self.prototype_counts[rel] = 1

    def learn_from_graph(self, facts: List[Dict[str, Any]]):
        """
        Learns relational argument grounding directly from the knowledge graph facts
        without hardcoded phrase lists or dictionaries.
        """
        for f in facts:
            s = f.get("subject", "").strip().lower()
            r = f.get("relation", "").strip()
            o = f.get("object", "").strip().lower()

            if r in self.relations and s and o:
                self.graph_pair_priors[(s, o)] = r
                self.graph_pair_priors[(o, s)] = r

    def train_exemplars(self, training_pairs: List[Tuple[str, str]], learning_rate: float = 1.0):
        """
        Incrementally updates continuous relation prototype centroids in O(1) time
        using observed training expressions.
        """
        dim = self.dimension or ArisSemanticEncoder.DIMENSION
        for phrase, rel in training_pairs:
            if not rel or not phrase:
                continue
            self.register_relation(rel)
            vec = self.encode_phrase(phrase)
            if self.dimension is None:
                self.dimension = len(vec)
                dim = self.dimension
            self.prototype_counts[rel] += 1
            eta = learning_rate / self.prototype_counts[rel]
            for i in range(dim):
                self.prototypes[rel][i] = (1.0 - eta) * self.prototypes[rel][i] + eta * vec[i]

        # Re-normalize centroids to unit sphere
        for rel in self.relations:
            norm = math.sqrt(sum(x * x for x in self.prototypes[rel]))
            if norm > 1e-9:
                self.prototypes[rel] = [x / norm for x in self.prototypes[rel]]

    def abstract_relation(
        self,
        surface_phrase: str,
        question_text: str = "",
        entities: Optional[List[str]] = None,
        target_role: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Infers the canonical relation in strictly O(K) constant time via continuous
        coordinate projection against canonical prototype centroids.
        Zero hardcoded word checks, zero relation-specific if/else.
        """
        cleaned = surface_phrase.strip().lower()
        if not cleaned or not self.relations:
            return {
                "predicted_relation": self.relations[0] if self.relations else "",
                "confidence": 0.0,
                "candidate_relations": [(r, 0.0) for r in self.relations],
            }

        # Encode surface phrase into continuous coordinate vector
        query_vec = self.encode_phrase(cleaned)

        # O(K) prototype similarity scoring
        candidates = []
        ents_lower = [e.strip().lower() for e in (entities or [])]
        grounded_pair_rel = None
        if len(ents_lower) >= 2:
            grounded_pair_rel = self.graph_pair_priors.get((ents_lower[0], ents_lower[1]))

        for rel in self.relations:
            proto_vec = self.prototypes.get(rel)
            if not proto_vec:
                continue

            # Pure continuous cosine similarity against relation centroid (O(1))
            sim = sum(a * b for a, b in zip(query_vec, proto_vec))

            # Graph-grounded entity pair prior (pure graph topology, zero string rules)
            if grounded_pair_rel and rel == grounded_pair_rel:
                sim += 0.30

            candidates.append((rel, sim))

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_rel, best_score = candidates[0]

        return {
            "predicted_relation": best_rel,
            "confidence": round(float(best_score), 4),
            "candidate_relations": [(r, round(float(s), 4)) for r, s in candidates],
        }
