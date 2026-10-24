# pyright: ignore
"""
AR.I.S — Query Binding Evaluator

Benchmark evaluation separating the 10 stages of Question -> Semantic Query understanding:
  1. Entity extraction
  2. Question-type recognition
  3. Relation phrase extraction
  4. Canonical relation mapping
  5. Subject binding
  6. Object binding
  7. Answer-slot detection
  8. Semantic query accuracy
  9. Graph path accuracy
  10. Final answer accuracy

Tracks failure distribution:
  - ENTITY_FAILURE
  - RELATION_FAILURE
  - DIRECTION_FAILURE
  - QUESTION_STRUCTURE_FAILURE
  - ANSWER_SLOT_FAILURE

Strict Non-Negotiable Directives:
  - Zero external APIs, zero LLMs, zero hard patches.
  - Graph search NEVER silently repairs an incorrect semantic query.
  - Complete per-question diagnostic provenance recorded into aris_query_binding_eval.json.
"""

import json
import os
from typing import Dict, Any, List, Optional, Tuple

from api.query_representation import ArisQueryBinder
from api.symbolic_graph import ArisSymbolicGraph


class ArisQueryEvaluator:
    """
    Evaluates Question -> Semantic Query binding fidelity across isolated cognitive stages.
    """

    @classmethod
    def load_facts(cls, facts_path: str) -> List[Dict[str, Any]]:
        """Loads canonical knowledge facts."""
        if not os.path.exists(facts_path):
            return []
        with open(facts_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def load_dataset(cls, dataset_path: str) -> List[Dict[str, Any]]:
        """Loads query binding dataset."""
        if not os.path.exists(dataset_path):
            return []
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def evaluate_stages(
        cls,
        dataset_path: Optional[str] = "aris_query_binding_dataset.json",
        facts: Optional[List[Dict[str, Any]]] = None,
        facts_path: Optional[str] = None,
        output_path: Optional[str] = "aris_query_binding_eval.json",
    ) -> Dict[str, Any]:
        """
        Executes complete benchmark evaluation separating all 10 stages.
        """
        dataset = cls.load_dataset(dataset_path) if dataset_path else []
        if facts is None:
            if facts_path and os.path.exists(facts_path):
                facts = cls.load_facts(facts_path)
            else:
                from ..cognitive.brain import ArisBrain
                facts = ArisBrain.retrieve_facts(limit=1000)

        # Build symbolic graph from active facts for offline graph reachability
        graph = ArisSymbolicGraph.build_from_facts(facts=facts)

        # Gather registered relations and known entities from facts
        known_entities = list({f["subject"] for f in facts} | {f["object"] for f in facts})
        registered_relations = list({f["relation"] for f in facts})

        total = len(dataset)
        stage_counts = {
            "entity_extraction": 0,
            "question_type": 0,
            "relation_extraction": 0,
            "canonical_relation": 0,
            "subject_binding": 0,
            "object_binding": 0,
            "answer_slot": 0,
            "semantic_query": 0,
            "graph_path": 0,
            "final_answer": 0,
        }

        failure_counts = {
            "ENTITY_FAILURE": 0,
            "RELATION_FAILURE": 0,
            "DIRECTION_FAILURE": 0,
            "QUESTION_STRUCTURE_FAILURE": 0,
            "ANSWER_SLOT_FAILURE": 0,
        }

        eval_records = []

        for item in dataset:
            question_text = item["question"]
            expected_sf = item.get("surface_form", {})
            expected_sq = item.get("semantic_query", {})

            # 1. Parse question through pure cognitive structural pipeline
            aris_result = ArisQueryBinder.parse_question(
                text=question_text,
                known_entities=known_entities,
                registered_relations=registered_relations,
            )

            qu = aris_result["query_understanding"]
            sq = aris_result["semantic_query"]

            # Expected fields
            exp_ent = str(expected_sf.get("entity", "")).strip().lower()
            exp_qtype = str(expected_sf.get("question_type", "")).strip().lower()
            exp_rel_phrase = str(expected_sf.get("relation_phrase", "")).strip().lower()

            exp_subj = expected_sq.get("subject")
            exp_rel = expected_sq.get("relation")
            exp_obj = expected_sq.get("object")
            exp_slot = expected_sq.get("answer_slot")

            # Actual fields
            act_entities = [str(e).strip().lower() for e in qu.get("entities", [])]
            act_qtype = str(qu.get("wh_type") or "").strip().lower()
            act_rel_phrase = str(qu.get("relation_surface", "")).strip().lower()

            act_subj = qu.get("subject")
            act_rel = qu.get("relation_canonical")
            act_obj = qu.get("object")
            act_slot = qu.get("answer_slot")

            # Stage Evaluations
            # 1. Entity Extraction
            m_entity = (exp_ent in act_entities) if exp_ent else (len(act_entities) == 0)

            # 2. Question-type Recognition
            if not exp_qtype:
                m_qtype = not act_qtype
            else:
                m_qtype = (exp_qtype in act_qtype) or (act_qtype in exp_qtype) or (exp_qtype == "does" and act_slot == "boolean") or (exp_qtype == "is" and act_slot == "boolean")

            # 3. Relation Phrase Extraction
            if not exp_rel_phrase:
                m_rel_phrase = not act_rel_phrase
            else:
                m_rel_phrase = (exp_rel_phrase in act_rel_phrase) or any(
                    w in act_rel_phrase for w in exp_rel_phrase.split()
                )

            # 4. Canonical Relation Mapping
            m_canonical_rel = (act_rel == exp_rel)

            # 5. Subject Binding
            if exp_subj is None:
                m_subj = (act_subj is None or act_subj == "?")
            else:
                m_subj = (act_subj is not None and str(act_subj).strip().lower() == str(exp_subj).strip().lower())

            # 6. Object Binding
            if exp_obj is None:
                m_obj = (act_obj is None or act_obj == "?")
            else:
                m_obj = (act_obj is not None and str(act_obj).strip().lower() == str(exp_obj).strip().lower())

            # 7. Answer Slot Detection
            m_slot = (act_slot == exp_slot)

            # 8. Semantic Query Accuracy (ALL four components match)
            m_semantic_query = m_subj and m_canonical_rel and m_obj and m_slot

            # 9 & 10: Graph Traversal & Final Answer strictly over the constructed semantic_query
            path = []
            final_answer = None

            subj_for_graph = act_subj
            rel_for_graph = act_rel
            obj_for_graph = act_obj

            if act_slot == "object" and subj_for_graph and subj_for_graph != "?":
                # Forward lookup in symbolic graph
                edges = graph.adjacency.get(subj_for_graph, [])
                for edge in edges:
                    if edge.relation == rel_for_graph and edge.direction == "forward":
                        path = [[subj_for_graph, edge.relation, edge.to_entity]]
                        final_answer = edge.to_entity
                        break

            elif act_slot == "subject" and obj_for_graph and obj_for_graph != "?":
                # Reverse lookup in symbolic graph
                for src, edges in graph.adjacency.items():
                    for edge in edges:
                        if edge.to_entity == obj_for_graph and edge.relation == rel_for_graph and edge.direction == "forward":
                            path = [[edge.from_entity, edge.relation, obj_for_graph]]
                            final_answer = edge.from_entity
                            break
                    if path:
                        break

            elif act_slot == "boolean" and subj_for_graph and obj_for_graph:
                # Direct edge check
                edges = graph.adjacency.get(subj_for_graph, [])
                edge_found = any(
                    edge.to_entity == obj_for_graph and edge.relation == rel_for_graph and edge.direction == "forward"
                    for edge in edges
                )
                if edge_found:
                    path = [[subj_for_graph, rel_for_graph, obj_for_graph]]
                    final_answer = "Yes"
                else:
                    path = []
                    final_answer = "No"

            # Check expected ground truth answer from facts
            exp_path = []
            exp_answer = None
            if exp_slot == "object" and exp_subj:
                edges = graph.adjacency.get(exp_subj, [])
                for edge in edges:
                    if edge.relation == exp_rel and edge.direction == "forward":
                        exp_path = [[exp_subj, edge.relation, edge.to_entity]]
                        exp_answer = edge.to_entity
                        break
            elif exp_slot == "subject" and exp_obj:
                for src, edges in graph.adjacency.items():
                    for edge in edges:
                        if edge.to_entity == exp_obj and edge.relation == exp_rel and edge.direction == "forward":
                            exp_path = [[edge.from_entity, edge.relation, exp_obj]]
                            exp_answer = edge.from_entity
                            break
                    if exp_path:
                        break
            elif exp_slot == "boolean" and exp_subj and exp_obj:
                edges = graph.adjacency.get(exp_subj, [])
                edge_found = any(
                    edge.to_entity == exp_obj and edge.relation == exp_rel and edge.direction == "forward"
                    for edge in edges
                )
                if edge_found:
                    exp_path = [[exp_subj, exp_rel, exp_obj]]
                    exp_answer = "Yes"
                else:
                    exp_path = []
                    exp_answer = "No"

            m_path = (len(path) > 0 and len(exp_path) > 0 and path == exp_path)
            m_answer = (final_answer is not None and exp_answer is not None and str(final_answer).lower() == str(exp_answer).lower())

            # Update stage counters
            if m_entity:
                stage_counts["entity_extraction"] += 1
            if m_qtype:
                stage_counts["question_type"] += 1
            if m_rel_phrase:
                stage_counts["relation_extraction"] += 1
            if m_canonical_rel:
                stage_counts["canonical_relation"] += 1
            if m_subj:
                stage_counts["subject_binding"] += 1
            if m_obj:
                stage_counts["object_binding"] += 1
            if m_slot:
                stage_counts["answer_slot"] += 1
            if m_semantic_query:
                stage_counts["semantic_query"] += 1
            if m_path:
                stage_counts["graph_path"] += 1
            if m_answer:
                stage_counts["final_answer"] += 1

            # Failure category assignment
            failure_category = None
            if not m_semantic_query:
                if not m_entity:
                    failure_category = "ENTITY_FAILURE"
                    failure_counts["ENTITY_FAILURE"] += 1
                elif not m_qtype:
                    failure_category = "QUESTION_STRUCTURE_FAILURE"
                    failure_counts["QUESTION_STRUCTURE_FAILURE"] += 1
                elif not m_slot:
                    failure_category = "ANSWER_SLOT_FAILURE"
                    failure_counts["ANSWER_SLOT_FAILURE"] += 1
                elif not m_canonical_rel:
                    failure_category = "RELATION_FAILURE"
                    failure_counts["RELATION_FAILURE"] += 1
                elif not m_subj or not m_obj:
                    failure_category = "DIRECTION_FAILURE"
                    failure_counts["DIRECTION_FAILURE"] += 1

            record = {
                "input": question_text,
                "query_understanding": qu,
                "semantic_query": sq,
                "reasoning": {
                    "path": path,
                    "answer": final_answer,
                },
                "expected": {
                    "surface_form": expected_sf,
                    "semantic_query": expected_sq,
                    "reasoning": {
                        "path": exp_path,
                        "answer": exp_answer,
                    }
                },
                "evaluation": {
                    "entity_match": m_entity,
                    "qtype_match": m_qtype,
                    "relation_phrase_match": m_rel_phrase,
                    "canonical_relation_match": m_canonical_rel,
                    "subject_match": m_subj,
                    "object_match": m_obj,
                    "answer_slot_match": m_slot,
                    "semantic_query_match": m_semantic_query,
                    "graph_path_match": m_path,
                    "final_answer_match": m_answer,
                    "failure_category": failure_category,
                }
            }
            eval_records.append(record)

        # Compute metric percentages
        metrics = {
            "Total Questions": total,
            "Entity extraction": round((stage_counts["entity_extraction"] / max(total, 1)) * 100, 1),
            "Question type": round((stage_counts["question_type"] / max(total, 1)) * 100, 1),
            "Relation extraction": round((stage_counts["relation_extraction"] / max(total, 1)) * 100, 1),
            "Canonical relation": round((stage_counts["canonical_relation"] / max(total, 1)) * 100, 1),
            "Subject binding": round((stage_counts["subject_binding"] / max(total, 1)) * 100, 1),
            "Object binding": round((stage_counts["object_binding"] / max(total, 1)) * 100, 1),
            "Answer slot": round((stage_counts["answer_slot"] / max(total, 1)) * 100, 1),
            "Semantic Query Accuracy": round((stage_counts["semantic_query"] / max(total, 1)) * 100, 1),
            "Graph Path Accuracy": round((stage_counts["graph_path"] / max(total, 1)) * 100, 1),
            "Final Answer Accuracy": round((stage_counts["final_answer"] / max(total, 1)) * 100, 1),
        }

        output_data = {
            "metrics": metrics,
            "failure_distribution": failure_counts,
            "results": eval_records,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        return output_data
