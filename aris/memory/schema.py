"""
AR.I.S — Neuro-Symbolic Graph RAG Schema (2026 Production Grade)
Defines entity nodes, relation edges, and multi-hop paths.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
import time


@dataclass
class EntityNode:
    """
    Represents an entity node in the neuro-symbolic knowledge graph.
    """
    id: str
    name: str
    entity_type: str = "concept"
    description: str = ""
    embedding: Optional[List[float]] = None
    access_count: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "embedding": self.embedding,
            "access_count": self.access_count,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EntityNode:
        return cls(
            id=data["id"],
            name=data["name"],
            entity_type=data.get("entity_type", "concept"),
            description=data.get("description", ""),
            embedding=data.get("embedding"),
            access_count=data.get("access_count", 0),
            created_at=data.get("created_at", time.time()),
        )


@dataclass
class RelationEdge:
    """
    Represents a directed relation edge between two entities.
    """
    source_id: str
    predicate: str
    target_id: str
    weight: float = 1.0
    snippet: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "predicate": self.predicate,
            "target_id": self.target_id,
            "weight": self.weight,
            "snippet": self.snippet,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RelationEdge:
        return cls(
            source_id=data["source_id"],
            predicate=data["predicate"],
            target_id=data["target_id"],
            weight=data.get("weight", 1.0),
            snippet=data.get("snippet", ""),
            created_at=data.get("created_at", time.time()),
        )


@dataclass
class GraphPath:
    """
    Represents an explicit multi-hop reasoning path across nodes and edges.
    """
    nodes: List[EntityNode]
    edges: List[RelationEdge]
    score: float = 0.0

    def format_factual_narrative(self) -> str:
        """
        Converts the multi-hop path into human-understandable factual sentences.
        """
        if not self.edges:
            if self.nodes:
                return f"{self.nodes[0].name} ({self.nodes[0].entity_type}): {self.nodes[0].description}".strip()
            return ""

        narratives = []
        node_map = {n.id: n for n in self.nodes}
        for edge in self.edges:
            src = node_map.get(edge.source_id)
            tgt = node_map.get(edge.target_id)
            src_name = src.name if src else edge.source_id
            tgt_name = tgt.name if tgt else edge.target_id
            stmt = f"{src_name} {edge.predicate} {tgt_name}"
            if edge.snippet:
                stmt += f" ({edge.snippet})"
            narratives.append(stmt)

        return " -> ".join(narratives)

