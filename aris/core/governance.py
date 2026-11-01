"""
AR.I.S — Epistemic Authority & Governance Engine (2026 Production Grade)
Enforces the 4-tier deciding authority hierarchy:
  - Tier 0: Sovereign User / Constitutional Root
  - Tier 1: Epistemic Grounding (Truth Verification)
  - Tier 2: Intrinsic Teleological Discovery (Knowledge Gap Detection)
  - Tier 3: Deontic Action Authorization (Safety Gating)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import math

from ..memory.schema import EntityNode, RelationEdge
from ..memory.graph_store import ArisGraphStore


class EpistemicGovernor:
    """
    Epistemic Authority and Governance Engine:
    Decides what knowledge is admitted to memory, what goals to pursue next,
    and what actions are permissible.
    """

    def __init__(
        self,
        graph_store: Optional[ArisGraphStore] = None,
        verification_threshold: float = 0.7,
    ):
        self.graph_store = graph_store
        self.verification_threshold = verification_threshold

    def evaluate_candidate_memory(
        self,
        name: str,
        description: str,
        source: str = "dialogue",
        user_confirmed: bool = False,
    ) -> Dict[str, Any]:
        """
        Tier 1: Evaluates candidate entity/fact before admission to permanent memory.
        Calculates epistemic confidence based on authority origin and corroboration.
        """
        # Tier 0 Override: Direct user confirmation has maximum authority
        if user_confirmed or source in ("user_direct", "constitutional_root"):
            return {
                "permitted": True,
                "status": "verified",
                "confidence": 1.0,
                "reason": "Direct Tier 0 Sovereign User / Root origin",
            }

        # For dialogue extractions without explicit confirmation:
        # Check if already present in graph store
        existing = self.graph_store.find_entity_by_name(name) if self.graph_store else None

        if existing:
            # Corroboration increases confidence with access count
            corroboration_score = min(0.9, 0.4 + (existing.access_count * 0.1))
            return {
                "permitted": True,
                "status": "verified" if corroboration_score >= self.verification_threshold else "hypothesis",
                "confidence": corroboration_score,
                "reason": f"Corroborated across {existing.access_count + 1} observations",
            }

        # Single uncorroborated mention starts as a hypothesis
        return {
            "permitted": True,
            "status": "hypothesis",
            "confidence": 0.45,
            "reason": "Single observation without corroboration (admitted as hypothesis)",
        }

    def analyze_knowledge_gaps(self, max_gaps: int = 5) -> List[Dict[str, Any]]:
        """
        Tier 2: Intrinsic Teleological Authority (Auto-Discovery).
        Identifies structural knowledge gaps in the graph to direct autonomous learning.
        """
        if not self.graph_store:
            return []

        all_entities = self.graph_store.get_all_entities()
        if not all_entities:
            return []

        gaps = []
        for ent in all_entities:
            out_edges = self.graph_store.get_out_edges(ent.id)
            in_edges = self.graph_store.get_in_edges(ent.id)
            total_edges = len(out_edges) + len(in_edges)

            # Isolated or poorly-connected nodes represent knowledge gaps
            if total_edges <= 1:
                gaps.append({
                    "entity_id": ent.id,
                    "entity_name": ent.name,
                    "edge_count": total_edges,
                    "status": "low_connectivity_gap",
                    "recommendation": f"Explore relational dependencies and causal role for '{ent.name}'",
                })

        # Sort by lowest connectivity first
        gaps.sort(key=lambda g: g["edge_count"])
        return gaps[:max_gaps]

    def check_action_permission(self, action_type: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Tier 3: Deontic Action Authority.
        Gates actions based on reversibility, risk, and authorization tiers.
        """
        action_type = action_type.lower()

        # Class 1: Reflective / Passive actions are always permitted
        if action_type in ("memory_read", "embedding_compute", "inference", "benchmark_eval"):
            return {
                "allowed": True,
                "tier": "Tier 3 Autonomous Reflective",
                "reason": "Passive, non-mutating cognitive operation",
            }

        # Class 2: Memory mutations require confidence verification
        if action_type in ("memory_write", "relation_create"):
            confidence = (context or {}).get("confidence", 0.0)
            if confidence >= self.verification_threshold:
                return {
                    "allowed": True,
                    "tier": "Tier 1 Epistemic Grounding",
                    "reason": f"Sufficient confidence ({confidence:.2f} >= {self.verification_threshold})",
                }
            return {
                "allowed": True,
                "tier": "Tier 1 Hypothesizing",
                "as_hypothesis": True,
                "reason": f"Admitted only as hypothesis ({confidence:.2f} < {self.verification_threshold})",
            }

        # Class 3: External real-world actions require Tier 0 confirmation
        return {
            "allowed": False,
            "tier": "Tier 0 Sovereign Override Required",
            "reason": f"External action '{action_type}' requires explicit user confirmation",
        }

