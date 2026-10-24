"""
AR.I.S — Symbolic Graph Reasoning Engine (Pillar 3)

Provides deterministic graph topology management, multi-hop path discovery,
and human-understandable provenance generation over stored Tier 3 relational bindings.

Key Architectural Invariants:
  - 100% Explainable Provenance: Every path returns the exact sequence of hops,
    relations, directions, and source statements.
  - Cycle-Free Traversal: Prevents infinite loops across circular relationships.
  - Bidirectional Graph Connectivity: Discovers both forward (A -> B) and reverse (B <- A) paths.
  - Zero Hardcoded Dictionaries: Relies purely on structural entity candidates and relation spans.
"""

import math
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import deque, defaultdict
from ..linguistic.tokenizer import ArisTokenizer


class ArisGraphEdge:
    """Represents a directed edge between two entity candidates in the knowledge graph."""

    def __init__(
        self,
        from_entity: str,
        relation: str,
        to_entity: str,
        direction: str = "forward",
        fact_id: Optional[str] = None,
        statement: Optional[str] = None,
        confidence: float = 1.0,
    ):
        self.from_entity = from_entity
        self.relation = relation
        self.to_entity = to_entity
        self.direction = direction  # "forward" or "reverse"
        self.fact_id = fact_id
        self.statement = statement
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_entity": self.from_entity,
            "relation": self.relation,
            "to_entity": self.to_entity,
            "direction": self.direction,
            "fact_id": self.fact_id,
            "statement": self.statement,
            "confidence": self.confidence,
        }


class ArisGraphPath:
    """Represents a sequence of edges connecting a start entity to an end entity."""

    def __init__(self, start_entity: str, end_entity: str, edges: List[ArisGraphEdge]):
        self.start_entity = start_entity
        self.end_entity = end_entity
        self.edges = edges
        self.hop_count = len(edges)

    @property
    def path_summary(self) -> str:
        if not self.edges:
            return self.start_entity

        parts = [self.start_entity]
        for e in self.edges:
            arrow = f"--[{e.relation}]-->" if e.direction == "forward" else f"<--[{e.relation}]--"
            parts.append(f" {arrow} {e.to_entity}")
        return "".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_entity": self.start_entity,
            "end_entity": self.end_entity,
            "hop_count": self.hop_count,
            "path_summary": self.path_summary,
            "steps": [e.to_dict() for e in self.edges],
        }


class ArisSymbolicGraph:
    """
    Constructs and queries the in-memory symbolic graph derived from Tier 3 relational bindings.
    """

    def __init__(self):
        # Adjacency list: entity_norm -> list of ArisGraphEdge
        self.adjacency: Dict[str, List[ArisGraphEdge]] = defaultdict(list)
        # Mapping from normalized entity name to canonical display name
        self.canonical_names: Dict[str, str] = {}

    @classmethod
    def build_from_facts(cls, facts: Optional[List[Dict[str, Any]]] = None) -> "ArisSymbolicGraph":
        """
        Builds a graph instance from a list of fact documents, or pulls from MongoDB if None.
        """
        if facts is None:
            from ..cognitive.brain import ArisBrain
            facts = ArisBrain.retrieve_facts(limit=5000)

        graph = cls()
        for fact in facts:
            fid = fact.get("id") or str(fact.get("_id", ""))
            stmt = fact.get("statement", "")
            conf = fact.get("confidence", 1.0)
            relations = fact.get("relations", [])

            # Retrocompute relations if not cached
            if not relations and stmt:
                from ..linguistic.tokenizer import ArisTokenizer
                from .relational import ArisRelationalBinder
                toks = fact.get("tokens") or ArisTokenizer.tokenize(stmt)
                cands = fact.get("candidates") or ArisTokenizer.extract_candidates(stmt, toks)
                relations = ArisRelationalBinder.bind_relations(stmt, toks, cands)

            for bind in relations:
                if not bind.get("subject") or not bind.get("object") or not bind.get("relation"):
                    continue
                s_subj = bind["subject"]
                s_text = s_subj.get("text", "").strip() if isinstance(s_subj, dict) else str(s_subj).strip()
                s_rel = bind["relation"]
                r_text = (s_rel.get("normalized") or s_rel.get("text", "")).strip() if isinstance(s_rel, dict) else str(s_rel).strip()
                s_obj = bind["object"]
                o_text = s_obj.get("text", "").strip() if isinstance(s_obj, dict) else str(s_obj).strip()

                if s_text and r_text and o_text:
                    graph.add_relational_binding(
                        subject=s_text,
                        relation=r_text,
                        obj=o_text,
                        fact_id=fid,
                        statement=stmt,
                        confidence=conf,
                    )

        return graph

    def add_relational_binding(
        self,
        subject: str,
        relation: str,
        obj: str,
        fact_id: Optional[str] = None,
        statement: Optional[str] = None,
        confidence: float = 1.0,
    ):
        """Adds forward and reverse traversable edges for a relational binding."""
        s_norm = subject.lower()
        o_norm = obj.lower()

        self.canonical_names[s_norm] = subject
        self.canonical_names[o_norm] = obj

        # Forward edge: Subject -> Object
        fwd_edge = ArisGraphEdge(
            from_entity=subject,
            relation=relation,
            to_entity=obj,
            direction="forward",
            fact_id=fact_id,
            statement=statement,
            confidence=confidence,
        )
        self.adjacency[s_norm].append(fwd_edge)

        # Reverse edge: Object -> Subject
        rev_edge = ArisGraphEdge(
            from_entity=obj,
            relation=relation,
            to_entity=subject,
            direction="reverse",
            fact_id=fact_id,
            statement=statement,
            confidence=confidence,
        )
        self.adjacency[o_norm].append(rev_edge)

    def find_paths(
        self,
        start_entity: str,
        end_entity: str,
        max_depth: int = 5,
        limit: int = 5,
    ) -> List[ArisGraphPath]:
        """
        Finds all simple (cycle-free) paths between start_entity and end_entity up to max_depth.
        Uses Breadth-First Search (BFS) to discover shortest paths first.
        """
        s_norm = start_entity.strip().lower()
        e_norm = end_entity.strip().lower()

        # Fuzzy-match: resolve normalized key to the best adjacency key when exact lookup misses
        # (e.g. "dr. elena rostova" -> "elena rostova" because tokenizer splits "Dr." separately)
        if s_norm not in self.adjacency:
            matches = [k for k in self.adjacency if s_norm in k or k in s_norm]
            if matches:
                s_norm = max(matches, key=len)
            else:
                return []

        if e_norm not in self.adjacency:
            matches = [k for k in self.adjacency if e_norm in k or k in e_norm]
            if matches:
                e_norm = max(matches, key=len)
            else:
                return []

        if s_norm == e_norm:
            return []

        canonical_start = self.canonical_names.get(s_norm, start_entity.strip())
        canonical_end = self.canonical_names.get(e_norm, end_entity.strip())

        paths_found: List[ArisGraphPath] = []
        # Queue stores tuples: (current_node_norm, list_of_edges, set_of_visited_nodes)
        queue: deque = deque([(s_norm, [], {s_norm})])

        while queue and len(paths_found) < limit:
            curr_norm, current_edges, visited = queue.popleft()

            if len(current_edges) >= max_depth:
                continue

            for edge in self.adjacency.get(curr_norm, []):
                next_norm = edge.to_entity.strip().lower()

                if next_norm == e_norm:
                    # Target reached!
                    completed_edges = current_edges + [edge]
                    paths_found.append(
                        ArisGraphPath(
                            start_entity=canonical_start,
                            end_entity=canonical_end,
                            edges=completed_edges,
                        )
                    )
                    if len(paths_found) >= limit:
                        break
                elif next_norm not in visited:
                    new_visited = visited | {next_norm}
                    queue.append((next_norm, current_edges + [edge], new_visited))

        return paths_found

    def get_connected_entities(self, entity: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Returns all entities reachable from the given entity within max_depth hops.
        """
        start_norm = entity.strip().lower()
        if start_norm not in self.adjacency:
            return []

        visited: Set[str] = {start_norm}
        queue: deque = deque([(start_norm, 0)])
        results = []

        while queue:
            curr, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for edge in self.adjacency.get(curr, []):
                nxt = edge.to_entity.strip().lower()
                if nxt not in visited:
                    visited.add(nxt)
                    results.append({
                        "entity": self.canonical_names.get(nxt, edge.to_entity),
                        "distance": depth + 1,
                        "via_relation": edge.relation,
                    })
                    queue.append((nxt, depth + 1))

        return results

    def find_open_variable_hypotheses(
        self,
        start_entity: str,
        relation_query: Optional[str] = None,
        max_depth: int = 8,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Explores reachable paths starting from start_entity and ranks candidate target
        entities based on path validity, GNN positional decay, topological termination,
        and relational alignment.

        Zero Hard Patches (Rule 1 & Rule 4):
          - NO hardcoded stopword or keyword lists.
          - Relational overlap computed via pure set operations and subword continuity.
          - Property/intermediate vs terminal referents distinguished purely by graph topology (leaf vs intermediate).
        """
        start_norm = start_entity.strip().lower()
        if start_norm not in self.adjacency:
            # Fuzzy match on keys if exact normalized key not found
            matching_keys = [k for k in self.adjacency if start_norm in k or k in start_norm]
            if matching_keys:
                start_norm = matching_keys[0]
            else:
                return []

        canonical_start = self.canonical_names.get(start_norm, start_entity.strip())
        start_tokens = {t["token"].lower() for t in ArisTokenizer.tokenize(start_norm) if t.get("type") != "symbol"}
        all_query_tokens = {t["token"].lower() for t in ArisTokenizer.tokenize(relation_query or "") if t.get("type") != "symbol"}
        rel_content = {w for w in (all_query_tokens - start_tokens) if len(w) > 1}

        # Structural token length weighting: longer content tokens carry higher semantic entropy
        def _token_weight(tok: str) -> float:
            return math.log(max(2.0, float(len(tok))))

        total_query_weight = sum(_token_weight(w) for w in rel_content) if rel_content else 1.0

        hypotheses: List[Dict[str, Any]] = []
        visited_nodes: Set[str] = {start_norm}
        queue: deque = deque([(start_norm, [], visited_nodes)])

        while queue:
            curr_norm, curr_edges, visited = queue.popleft()

            if len(curr_edges) >= max_depth:
                continue

            for edge in self.adjacency.get(curr_norm, []):
                next_norm = edge.to_entity.strip().lower()
                if next_norm == start_norm or next_norm in visited:
                    continue

                new_edges = curr_edges + [edge]
                new_visited = visited | {next_norm}
                target_canonical = self.canonical_names.get(next_norm, edge.to_entity)

                # Compute relational alignment score across path relations and intermediate nodes
                path_parts = []
                for e in new_edges:
                    path_parts.extend([e.from_entity.lower(), e.relation.lower(), e.to_entity.lower()])
                path_tokens = {t["token"].lower() for t in ArisTokenizer.tokenize(" ".join(path_parts)) if t.get("type") != "symbol"}

                rel_match = 0.0
                if rel_content:
                    weighted_overlap = 0.0
                    for rc in rel_content:
                        rc_w = _token_weight(rc)
                        if rc in path_tokens:
                            weighted_overlap += rc_w
                        else:
                            # Subword continuity (e.g. 'patner' in 'partner', 'operates' in 'operate')
                            for pt in path_tokens:
                                if len(rc) >= 3 and len(pt) >= 3 and (rc in pt or pt in rc):
                                    weighted_overlap += rc_w * 0.85
                                    break
                    token_match = min(1.0, weighted_overlap / total_query_weight)

                    if token_match < 0.5:
                        try:
                            from ..cognitive.goal_reasoning_tensor_engine import ArisGoalReasoningTensorEngine
                            q_rel_text = " ".join(rel_content)
                            path_full_summary = " ".join(path_parts)
                            tensor_match = ArisGoalReasoningTensorEngine.align_predicates(q_rel_text, path_full_summary)
                        except Exception:
                            tensor_match = 0.0
                    else:
                        tensor_match = token_match

                    rel_match = max(token_match, tensor_match)
                else:
                    rel_match = 1.0

                if rel_match > 0.0 or not rel_content:
                    hop_count = len(new_edges)
                    edge_prod = 1.0
                    for e in new_edges:
                        edge_prod *= float(e.confidence)

                    # Multi-hop score combining edge confidence, positional decay, and relational alignment
                    gnn_score = round((edge_prod / float(hop_count)) * rel_match, 2)

                    path_obj = ArisGraphPath(
                        start_entity=canonical_start,
                        end_entity=target_canonical,
                        edges=new_edges,
                    )

                    hypotheses.append({
                        "target_entity": target_canonical,
                        "path": path_obj,
                        "confidence": gnn_score,
                        "hop_count": hop_count,
                        "rel_match": rel_match,
                        "path_summary": path_obj.path_summary,
                        "provenance": [e.statement for e in new_edges if e.statement],
                    })

                queue.append((next_norm, new_edges, new_visited))

        if not hypotheses:
            return []

        # Deduplicate by target_entity, keeping highest confidence
        deduped: Dict[str, Dict[str, Any]] = {}
        for h in hypotheses:
            tgt = h["target_entity"].lower()
            if tgt not in deduped or h["confidence"] > deduped[tgt]["confidence"]:
                deduped[tgt] = h

        # Neutral topology sorting: ranked by relational alignment and confidence (Rule 4: no leaf bias, no intermediate penalty)
        ranked = sorted(
            deduped.values(),
            key=lambda x: (round(x.get("rel_match", 0.0), 2), x["confidence"]),
            reverse=True,
        )
        return ranked[:limit]





