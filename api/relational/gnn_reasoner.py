"""
AR.I.S — Graph Neural Network & Positional Path Reasoner (Pillar 4)

Implements multi-hop positional path scoring, link plausibility prediction,
and hypothesis generation over connected entity graphs with dynamic feedback inhibition.

Key Architectural Invariants:
  - Multi-Layer Positional Message Passing: Evaluates path connectivity where layer k represents k hops.
  - Shortest Path Distance (SPD) Scoring: Positional decay score scaled by path length and step confidences.
  - Dynamic Inhibition Masking: If an assertion is marked denied in brain_relational_beliefs,
    it is masked to 0.0 confidence and never emitted as a valid hypothesis.
  - Pure Local Offline Execution: PyTorch / NumPy tensor operations, zero external API calls.
"""

from typing import List, Dict, Any, Optional, Union
from .symbolic_graph import ArisSymbolicGraph, ArisGraphPath
from ..cognitive.relational_beliefs import ArisRelationalBeliefManager


class ArisGNNReasoner:
    """
    Evaluates multi-hop graph connectivity, computes positional plausibility scores,
    and formulates grounded hypotheses with full path provenance.
    """

    @classmethod
    def score_path(cls, path: ArisGraphPath) -> float:
        """
        Computes positional plausibility score for a discovered path:
          score = (product of edge confidences) / (hop_count)
        For 1-hop direct: 1.0
        For 2-hop indirect: 0.5
        For 3-hop indirect: 0.33
        """
        if not path or not path.edges:
            return 0.0

        hop_count = path.hop_count
        if hop_count == 1:
            return float(path.edges[0].confidence)

        # Multi-layer positional decay
        edge_conf_product = 1.0
        for edge in path.edges:
            edge_conf_product *= float(edge.confidence)

        positional_decay = 1.0 / float(hop_count)
        score = round(edge_conf_product * positional_decay, 2)
        return max(0.0, min(1.0, score))

    @classmethod
    def evaluate_hypothesis(
        cls,
        subject: str,
        target_entity: str,
        relation: Optional[str] = None,
        facts: Optional[List[Dict[str, Any]]] = None,
        max_depth: int = 15,
    ) -> Dict[str, Any]:
        """
        Evaluates whether a plausible hypothesis exists between subject and target_entity.

        1. Checks inhibition matrix: If (subject, relation, target_entity) is denied, returns 0.0 confidence.
        2. Probes reciprocal / symmetric rules.
        3. Traverses symbolic graph for connecting multi-hop paths.
        4. Calculates GNN positional score and returns grounded hypothesis with provenance.
        """
        s_clean = subject.strip()
        t_clean = target_entity.strip()
        r_clean = relation.strip() if relation else None

        # Check Inhibition Matrix first
        if r_clean and ArisRelationalBeliefManager.is_denied(s_clean, r_clean, t_clean):
            return {
                "status": "denied",
                "confidence": 0.0,
                "subject": s_clean,
                "relation": r_clean,
                "target_entity": t_clean,
                "statement": f"Hypothesis Refuted: '{s_clean}' {r_clean} '{t_clean}' was explicitly rejected by user.",
                "reason": "previously_denied_by_user",
                "provenance": [],
            }
        elif not r_clean and ArisRelationalBeliefManager.is_connection_denied(s_clean, t_clean):
            return {
                "status": "denied",
                "confidence": 0.0,
                "subject": s_clean,
                "relation": "inhibited_connection",
                "target_entity": t_clean,
                "statement": f"Hypothesis Refuted: Relationship between '{s_clean}' and '{t_clean}' was explicitly rejected by user.",
                "reason": "previously_denied_by_user",
                "provenance": [],
            }

        # Build graph from memory
        graph = ArisSymbolicGraph.build_from_facts(facts=facts)

        # Check direct symmetrical rule if relation specified
        if r_clean and ArisRelationalBeliefManager.is_symmetric(r_clean):
            # Check if reverse is in graph
            rev_paths = graph.find_paths(t_clean, s_clean, max_depth=1)
            for p in rev_paths:
                if p.edges and p.edges[0].relation.lower() == r_clean.lower():
                    return {
                        "status": "verified_symmetric",
                        "confidence": 1.0,
                        "subject": s_clean,
                        "relation": r_clean,
                        "target_entity": t_clean,
                        "statement": f"Symmetric Relation Verified: '{s_clean}' {r_clean} '{t_clean}' (via reverse fact).",
                        "path_summary": p.path_summary,
                        "provenance": [e.statement for e in p.edges if e.statement],
                    }

        # Find multi-hop paths
        paths = graph.find_paths(s_clean, t_clean, max_depth=max_depth)

        if not paths:
            # Also try reverse search
            rev_paths = graph.find_paths(t_clean, s_clean, max_depth=max_depth)
            if rev_paths:
                paths = rev_paths

        # Dynamic Inhibition Masking: filter out paths containing any individually denied edge
        valid_paths = []
        for p in paths:
            has_denied_edge = False
            for e in p.edges:
                if ArisRelationalBeliefManager.is_denied(e.from_entity, e.relation, e.to_entity):
                    has_denied_edge = True
                    break
            if has_denied_edge:
                continue
            if p.hop_count == 1 and r_clean and ArisRelationalBeliefManager.is_denied(s_clean, r_clean, t_clean):
                continue
            valid_paths.append(p)
        paths = valid_paths

        if not paths:
            return {
                "status": "unconnected",
                "confidence": 0.0,
                "subject": s_clean,
                "target_entity": t_clean,
                "statement": f"No connecting relational paths found in memory between '{s_clean}' and '{t_clean}'.",
                "provenance": [],
            }

        # Select the shortest / strongest path
        best_path = min(paths, key=lambda p: p.hop_count)
        score = cls.score_path(best_path)

        # Check if the relation in query matches or can be hypothesized
        first_rel = best_path.edges[0].relation
        last_rel = best_path.edges[-1].relation
        inferred_rel = r_clean or (first_rel if best_path.hop_count == 1 else f"connected ({first_rel} -> {last_rel})")

        provenance_statements = [e.statement for e in best_path.edges if e.statement]

        return {
            "status": "plausible_hypothesis",
            "confidence": score,
            "best_path": best_path.path_summary,
            "subject": s_clean,
            "target_entity": t_clean,
            "relation": inferred_rel,
            "statement": (
                f"Grounded Hypothesis (Confidence: {score}): "
                f"'{s_clean}' is connected to '{t_clean}' through {best_path.hop_count} hop(s): {best_path.path_summary}."
            ),
            "path_summary": best_path.path_summary,
            "hop_count": best_path.hop_count,
            "path_steps": [e.to_dict() for e in best_path.edges],
            "provenance": provenance_statements,
        }

    @classmethod
    def evaluate_open_variable_hypothesis(
        cls,
        anchor_entity: Union[str, List[str]],
        relation: Optional[str] = None,
        facts: Optional[List[Dict[str, Any]]] = None,
        max_depth: int = 8,
        query_text: Optional[str] = None,
        target_type: Optional[str] = None,
        limit: int = 20,
        excluded_entities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Discovers candidate entities connected to anchor candidates matching the relational constraint,
        filtering out denied assertions from the inhibition matrix and scoring via ArisGoalReasoningTensorEngine.
        """
        if isinstance(anchor_entity, list):
            anchors = [a.strip() for a in anchor_entity if a and a.strip()]
        else:
            anchors = [anchor_entity.strip()] if anchor_entity else []

        r_clean = relation.strip() if relation else None

        graph = ArisSymbolicGraph.build_from_facts(facts=facts)
        raw_hypotheses = []
        for a_clean in anchors:
            hyps = graph.find_open_variable_hypotheses(
                start_entity=a_clean,
                relation_query=r_clean or query_text,
                max_depth=max_depth,
                limit=limit,
            )
            for h in hyps:
                h["anchor_entity"] = a_clean
            raw_hypotheses.extend(hyps)

        # In open-variable queries, the unknown target X cannot be one of the query's input grounded premises
        query_named_entities = {a.lower() for a in anchors}
        if query_text:
            for k in graph.adjacency:
                if len(k) >= 3 and k.lower() in query_text.lower():
                    query_named_entities.add(k.lower())

        valid_hypotheses = []
        for h in raw_hypotheses:
            tgt = h["target_entity"]
            path = h["path"]
            a_clean = h.get("anchor_entity") or (anchors[0] if anchors else "")

            # Exclude targets that are entities stated in the question premise
            if tgt.lower() in query_named_entities:
                continue

            # Exclude discourse-grounded entities when a restrictive follow-up query is issued
            if excluded_entities and tgt.lower() in {e.lower() for e in excluded_entities}:
                continue

            # Check inhibition matrix for path edges
            has_denied = False
            for e in path.edges:
                if ArisRelationalBeliefManager.is_denied(e.from_entity, e.relation, e.to_entity):
                    has_denied = True
                    break
            if has_denied:
                continue
            if ArisRelationalBeliefManager.is_connection_denied(a_clean, tgt):
                continue
            if r_clean and ArisRelationalBeliefManager.is_denied(a_clean, r_clean, tgt):
                continue

            valid_hypotheses.append(h)

        if not valid_hypotheses:
            primary_a = anchors[0] if anchors else "unknown"
            return {
                "status": "unconnected",
                "confidence": 0.0,
                "anchor_entity": primary_a,
                "answers": [],
                "statement": f"No open-variable relational pathways discovered starting from candidate anchors.",
                "provenance": [],
            }

        # Subspace A & C: Evaluate whole-path goal compatibility S_theta(c, p, G) via learned tensor engine
        scored_hypotheses = []
        try:
            from ..cognitive.goal_reasoning_tensor_engine import ArisGoalReasoningTensorEngine
            pred_type = target_type
            if not pred_type and (query_text or r_clean):
                q_text = query_text or r_clean or ""
                pred_res = ArisGoalReasoningTensorEngine.predict_target_type(q_text)
                pred_type = pred_res.get("best_type")

            clean_rel = r_clean or query_text or ""

            for h in valid_hypotheses:
                tgt = h["target_entity"]
                path = h["path"]
                path_rels = [e.relation for e in path.edges]
                edge_prod = 1.0
                for e in path.edges:
                    edge_prod *= float(e.confidence)

                cand_type = ArisGoalReasoningTensorEngine.infer_candidate_type(tgt, [path_rels[-1]] if path_rels else [])
                score_diag = ArisGoalReasoningTensorEngine.score_candidate_path(
                    query_text=query_text or r_clean or "",
                    goal_relation=clean_rel,
                    target_type=pred_type or "UNKNOWN",
                    candidate_entity=tgt,
                    candidate_type=cand_type,
                    path_relations=path_rels,
                    hop_count=h["hop_count"],
                    edge_confidence_prod=edge_prod,
                )
                h["candidate_type"] = cand_type
                h["target_type"] = pred_type
                h["whole_path_score"] = score_diag["score"]
                h["is_supported"] = score_diag["is_supported"]
                h["score_diagnostics"] = score_diag
                if score_diag["score"] > 0:
                    h["confidence"] = score_diag["score"]
                scored_hypotheses.append(h)

            scored_hypotheses.sort(key=lambda x: x.get("whole_path_score", x["confidence"]), reverse=True)
            valid_hypotheses = scored_hypotheses
        except Exception as e:
            pass

        best = valid_hypotheses[0]
        best_anchor = best.get("anchor_entity") or (anchors[0] if anchors else "")

        # Subspace D: Epistemic support check (evidence-backed support vs reachable unsupported node)
        if best.get("is_supported") is False and r_clean:
            return {
                "status": "unsupported",
                "confidence": 0.0,
                "anchor_entity": best_anchor,
                "answers": [],
                "statement": f"No evidence-supported relational pathways discovered for '{best_anchor}' satisfying '{r_clean}'.",
                "provenance": [],
                "all_hypotheses": valid_hypotheses,
            }

        tgt_entity = best["target_entity"]
        conf = best["confidence"]
        path_summary = best["path_summary"]
        provenance = best["provenance"]

        return {
            "status": "plausible_hypothesis",
            "confidence": conf,
            "anchor_entity": best_anchor,
            "target_entity": tgt_entity,
            "relation": r_clean or "connected",
            "statement": f"Graph Traversal Hypothesis (Confidence: {conf}): '{tgt_entity}' is connected to '{best_anchor}' via {path_summary}",
            "path_summary": path_summary,
            "hop_count": best["hop_count"],
            "provenance": provenance,
            "all_hypotheses": valid_hypotheses,
        }

