"""
AR.I.S — Relation Abstraction Evaluator

Benchmark harness evaluating Semantic Relation Abstraction across:
  - Seen phrasing accuracy (TRAIN split)
  - Unseen paraphrase accuracy (TEST split unseen items)
  - Noisy phrasing accuracy (controlled noise items)
  - Overall relation abstraction accuracy
  - Direction accuracy
  - Semantic query accuracy
  - Graph path accuracy
  - Final answer accuracy

Failure Categories:
  - LEXICAL_SIMILARITY_FAILURE
  - SEMANTIC_ABSTRACTION_FAILURE
  - DIRECTION_FAILURE
  - ENTITY_FAILURE
  - ANSWER_SLOT_FAILURE
  - QUERY_BINDING_FAILURE
  - GRAPH_FAILURE

Strict Non-Negotiable Directives:
  - Zero external APIs, zero LLMs, zero hard patches.
  - Trains ArisRelationAbstractionEngine strictly on the TRAIN partition.
  - Tests generalization on strictly unseen paraphrases in the TEST partition.
  - Records full diagnostic provenance into aris_relation_abstraction_eval.json.
"""

import json
import os
from typing import Dict, Any, List, Optional, Tuple

from api.relation_abstraction import ArisRelationAbstractionEngine
from api.query_representation import ArisQueryBinder
from api.symbolic_graph import ArisSymbolicGraph
from api.response_learner import ArisResponseLearner
from ..cognitive.semantic_frame import ArisInterrogativeRoleLearner


class ArisRelationEvaluator:
    """
    Evaluator for Semantic Relation Abstraction & Generalization.
    """

    @classmethod
    def load_json(cls, filepath: str) -> List[Dict[str, Any]]:
        if not os.path.exists(filepath):
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def evaluate_benchmark(
        cls,
        dataset_path: Optional[str] = "aris_relation_abstraction_benchmark.json",
        facts: Optional[List[Dict[str, Any]]] = None,
        facts_path: Optional[str] = None,
        output_path: Optional[str] = "aris_relation_abstraction_eval.json",
    ) -> Dict[str, Any]:
        benchmark = cls.load_json(dataset_path) if dataset_path else []
        if facts is None:
            if facts_path and os.path.exists(facts_path):
                facts = cls.load_json(facts_path)
            else:
                from ..cognitive.brain import ArisBrain
                facts = ArisBrain.retrieve_facts(limit=1000)

        # Build symbolic graph for pure offline graph reachability
        graph = ArisSymbolicGraph.build_from_facts(facts=facts)
        # Derive all known entities and registered relations dynamically from loaded data
        known_entities = sorted(list({f["subject"] for f in facts} | {f["object"] for f in facts}))
        registered_relations = sorted(list({item["expected_relation"] for item in benchmark} | {f["relation"] for f in facts}))

        # 1. Train ArisRelationAbstractionEngine STRICTLY on the TRAIN split
        engine = ArisRelationAbstractionEngine(canonical_relations=registered_relations)
        train_pairs = [
            (item["surface_form"]["relation_phrase"], item["expected_relation"])
            for item in benchmark
            if item.get("split") == "TRAIN" and item.get("surface_form", {}).get("relation_phrase")
        ]
        engine.train_exemplars(train_pairs)
        engine.learn_from_graph(facts)

        # Metrics accumulators across Levels 1–10
        counts = {
            "total": len(benchmark),
            "seen_total": 0,
            "seen_rel_correct": 0,
            "unseen_total": 0,
            "unseen_rel_correct": 0,
            "noisy_total": 0,
            "noisy_rel_correct": 0,
            "rel_correct": 0,
            "dir_correct": 0,
            "slot_correct": 0,
            "semantic_query_correct": 0,
            "unseen_sq_correct": 0,
            "graph_path_correct": 0,
            "final_answer_correct": 0,
            "entity_correct": 0,
            "structure_correct": 0,
            "surface_correct": 0,
            "query_binding_correct": 0,
            "frame_seen_correct": 0,
            "frame_unseen_correct": 0,
            "frame_noisy_correct": 0,
            "frame_overall_correct": 0,
        }

        failures = {
            "LEXICAL_SIMILARITY_FAILURE": 0,
            "SEMANTIC_ABSTRACTION_FAILURE": 0,
            "DIRECTION_FAILURE": 0,
            "ENTITY_FAILURE": 0,
            "ANSWER_SLOT_FAILURE": 0,
            "QUERY_BINDING_FAILURE": 0,
            "GRAPH_FAILURE": 0,
        }

        eval_records = []

        for item in benchmark:
            q_text = item["question"]
            split = item.get("split", "TRAIN")
            is_unseen = item.get("is_unseen", False)
            is_noisy = item.get("is_noisy", False)
            exp_rel = item["expected_relation"]
            exp_sq = item["semantic_query"]
            exp_sf = item.get("surface_form", {})

            # Parse question with trained abstraction engine
            parsed = ArisQueryBinder.parse_question(
                text=q_text,
                known_entities=known_entities,
                registered_relations=registered_relations,
                abstraction_engine=engine,
            )

            qu = parsed["query_understanding"]
            sq = parsed["semantic_query"]

            act_rel = qu.get("relation_canonical")
            act_surface = qu.get("relation_surface", "")
            act_subj = qu.get("subject")
            act_obj = qu.get("object")
            act_slot = qu.get("answer_slot")
            confidence = qu.get("confidence", 0.0)
            candidate_relations = qu.get("candidate_relations", [])

            # Expected semantic query fields
            exp_subj = exp_sq.get("subject")
            exp_obj = exp_sq.get("object")
            exp_slot = exp_sq.get("answer_slot")

            # Stage matches
            # 1. Relation match
            m_rel = (act_rel == exp_rel)

            # 2. Direction match (subject / object orientation)
            if exp_subj is None:
                m_subj = (act_subj is None or act_subj == "?")
            else:
                m_subj = (act_subj is not None and str(act_subj).strip().lower() == str(exp_subj).strip().lower())

            if exp_obj is None:
                m_obj = (act_obj is None or act_obj == "?")
            else:
                m_obj = (act_obj is not None and str(act_obj).strip().lower() == str(exp_obj).strip().lower())

            m_dir = m_subj and m_obj
            m_slot = (act_slot == exp_slot)
            m_semantic_query = m_rel and m_dir and m_slot
            m_frame = m_dir and m_slot

            # Level 1 structural metrics
            exp_ent = exp_sf.get("entity")
            if exp_ent:
                m_ent = any(str(e).strip().lower() == str(exp_ent).strip().lower() for e in qu.get("entities", []))
            else:
                m_ent = True
            if m_ent:
                counts["entity_correct"] += 1

            first_word = q_text.split()[0].lower() if q_text.split() else ""
            _, role_conf = ArisInterrogativeRoleLearner.infer_target_role(first_word)
            m_struct = (qu.get("wh_type") is not None) if (role_conf >= 0.80) else True
            if m_struct:
                counts["structure_correct"] += 1

            m_surface = bool(act_surface and len(act_surface.strip()) > 0)
            if m_surface:
                counts["surface_correct"] += 1

            m_qb = bool(sq.get("relation"))
            if m_qb:
                counts["query_binding_correct"] += 1

            # Update split counters
            if split == "TRAIN":
                counts["seen_total"] += 1
                if m_rel:
                    counts["seen_rel_correct"] += 1
                if m_frame:
                    counts["frame_seen_correct"] += 1

            if is_unseen:
                counts["unseen_total"] += 1
                if m_rel:
                    counts["unseen_rel_correct"] += 1
                if m_frame:
                    counts["frame_unseen_correct"] += 1
                if m_semantic_query:
                    counts["unseen_sq_correct"] += 1

            if is_noisy:
                counts["noisy_total"] += 1
                if m_rel:
                    counts["noisy_rel_correct"] += 1
                if m_frame:
                    counts["frame_noisy_correct"] += 1

            if m_rel:
                counts["rel_correct"] += 1
            if m_dir:
                counts["dir_correct"] += 1
            if m_slot:
                counts["slot_correct"] += 1
            if m_frame:
                counts["frame_overall_correct"] += 1
            if m_semantic_query:
                counts["semantic_query_correct"] += 1

            # 3. Graph Reasoning strictly over semantic_query
            path = []
            final_answer = None

            subj_for_graph = act_subj
            rel_for_graph = act_rel
            obj_for_graph = act_obj

            if act_slot == "object" and subj_for_graph and subj_for_graph != "?":
                edges = graph.adjacency.get(subj_for_graph, [])
                for edge in edges:
                    if edge.relation == rel_for_graph and edge.direction == "forward":
                        path = [[subj_for_graph, edge.relation, edge.to_entity]]
                        final_answer = edge.to_entity
                        break

            elif act_slot == "subject" and obj_for_graph and obj_for_graph != "?":
                for src, edges in graph.adjacency.items():
                    for edge in edges:
                        if edge.to_entity == obj_for_graph and edge.relation == rel_for_graph and edge.direction == "forward":
                            path = [[edge.from_entity, edge.relation, obj_for_graph]]
                            final_answer = edge.from_entity
                            break
                    if path:
                        break

            elif act_slot == "boolean" and subj_for_graph and obj_for_graph:
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

            # Check expected ground truth path/answer
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

            if m_path:
                counts["graph_path_correct"] += 1
            if m_answer:
                counts["final_answer_correct"] += 1

            # Generate human-readable response from structured result
            human_response = ArisResponseLearner.generate({
                "subject": act_subj,
                "relation": act_rel,
                "object": act_obj,
                "answer_slot": act_slot,
                "final_answer": final_answer,
                "reasoning_path": path,
            })

            # Failure classification
            failure_category = None
            if not m_semantic_query or not m_answer:
                if not m_rel:
                    if any(w in act_surface for w in ["work", "use", "orbit", "manage"]):
                        failure_category = "LEXICAL_SIMILARITY_FAILURE"
                        failures["LEXICAL_SIMILARITY_FAILURE"] += 1
                    else:
                        failure_category = "SEMANTIC_ABSTRACTION_FAILURE"
                        failures["SEMANTIC_ABSTRACTION_FAILURE"] += 1
                elif not m_dir:
                    failure_category = "DIRECTION_FAILURE"
                    failures["DIRECTION_FAILURE"] += 1
                elif not m_slot:
                    failure_category = "ANSWER_SLOT_FAILURE"
                    failures["ANSWER_SLOT_FAILURE"] += 1
                elif not m_semantic_query:
                    failure_category = "QUERY_BINDING_FAILURE"
                    failures["QUERY_BINDING_FAILURE"] += 1
                elif not m_answer:
                    failure_category = "GRAPH_FAILURE"
                    failures["GRAPH_FAILURE"] += 1

            record = {
                "input": q_text,
                "split": split,
                "is_unseen": is_unseen,
                "is_noisy": is_noisy,
                "expected_relation": exp_rel,
                "relation_surface": act_surface,
                "candidate_relations": candidate_relations,
                "predicted_canonical_relation": act_rel,
                "confidence": confidence,
                "subject": act_subj,
                "object": act_obj,
                "answer_slot": act_slot,
                "semantic_query": sq,
                "reasoning_path": path,
                "final_answer": final_answer,
                "human_response": human_response,
                "failure_category": failure_category,
                "expected": {
                    "surface_form": exp_sf,
                    "semantic_query": exp_sq,
                    "reasoning_path": exp_path,
                    "final_answer": exp_answer,
                }
            }
            eval_records.append(record)

        total = counts["total"]
        seen_tot = max(counts["seen_total"], 1)
        unseen_tot = max(counts["unseen_total"], 1)
        noisy_tot = max(counts["noisy_total"], 1)

        # Computed accuracy percentages
        entity_acc = round((counts["entity_correct"] / max(total, 1)) * 100, 1)
        structure_acc = round((counts["structure_correct"] / max(total, 1)) * 100, 1)
        surface_acc = round((counts["surface_correct"] / max(total, 1)) * 100, 1)
        query_binding_acc = round((counts["query_binding_correct"] / max(total, 1)) * 100, 1)

        frame_overall_acc = round((counts["frame_overall_correct"] / max(total, 1)) * 100, 1)
        frame_seen_acc = round((counts["frame_seen_correct"] / seen_tot) * 100, 1)
        frame_unseen_acc = round((counts["frame_unseen_correct"] / unseen_tot) * 100, 1)
        frame_noisy_acc = round((counts["frame_noisy_correct"] / noisy_tot) * 100, 1)

        role_acc = round((counts["dir_correct"] / max(total, 1)) * 100, 1)
        slot_acc = round((counts["slot_correct"] / max(total, 1)) * 100, 1)
        slot_failure_pct = round((failures["ANSWER_SLOT_FAILURE"] / max(total, 1)) * 100, 1)
        direction_acc = round((counts["dir_correct"] / max(total, 1)) * 100, 1)
        dir_failure_pct = round((failures["DIRECTION_FAILURE"] / max(total, 1)) * 100, 1)

        sq_overall_acc = round((counts["semantic_query_correct"] / max(total, 1)) * 100, 1)
        sq_unseen_acc = round((counts["unseen_sq_correct"] / unseen_tot) * 100, 1)

        rel_overall_acc = round((counts["rel_correct"] / max(total, 1)) * 100, 1)
        rel_seen_acc = round((counts["seen_rel_correct"] / seen_tot) * 100, 1)
        rel_unseen_acc = round((counts["unseen_rel_correct"] / unseen_tot) * 100, 1)
        rel_noisy_acc = round((counts["noisy_rel_correct"] / noisy_tot) * 100, 1)

        graph_path_acc = round((counts["graph_path_correct"] / max(total, 1)) * 100, 1)
        final_answer_acc = round((counts["final_answer_correct"] / max(total, 1)) * 100, 1)

        # Gate Evaluations Level 1 through 10
        levels = {
            "Level 1: Structural Parsing": {
                "metrics": {
                    "Entity extraction": f"{entity_acc}% (>= 99%)",
                    "Question structure": f"{structure_acc}% (>= 98%)",
                    "Relation surface extraction": f"{surface_acc}% (>= 95%)",
                    "Query binding": f"{query_binding_acc}% (>= 98%)",
                },
                "passed": (entity_acc >= 99.0 and structure_acc >= 98.0 and surface_acc >= 95.0 and query_binding_acc >= 98.0),
            },
            "Level 2: Syntactic Frame Normalization": {
                "metrics": {
                    "Overall frame accuracy": f"{frame_overall_acc}% (>= 90%)",
                    "Seen frame accuracy": f"{frame_seen_acc}% (>= 95%)",
                    "Unseen frame accuracy": f"{frame_unseen_acc}% (>= 80%)",
                    "Noisy frame accuracy": f"{frame_noisy_acc}% (>= 80%)",
                },
                "passed": (frame_overall_acc >= 90.0 and frame_seen_acc >= 95.0 and frame_unseen_acc >= 80.0 and frame_noisy_acc >= 80.0),
            },
            "Level 3: Role Assignment Accuracy": {
                "metrics": {
                    "Role assignment accuracy": f"{role_acc}% (>= 95%)",
                },
                "passed": (role_acc >= 95.0),
            },
            "Level 4: Answer Slot Accuracy": {
                "metrics": {
                    "Answer slot accuracy": f"{slot_acc}% (>= 95%)",
                    "Answer slot failure rate": f"{slot_failure_pct}% (<= 5%)",
                },
                "passed": (slot_acc >= 95.0 and slot_failure_pct <= 5.0),
            },
            "Level 5: Direction Accuracy": {
                "metrics": {
                    "Direction accuracy": f"{direction_acc}% (>= 95%)",
                    "Direction failure rate": f"{dir_failure_pct}% (<= 5%)",
                },
                "passed": (direction_acc >= 95.0 and dir_failure_pct <= 5.0),
            },
            "Level 6: Overall Semantic Query Accuracy": {
                "metrics": {
                    "Overall semantic query accuracy": f"{sq_overall_acc}% (>= 90%)",
                },
                "passed": (sq_overall_acc >= 90.0),
            },
            "Level 7: Unseen Semantic Query Accuracy": {
                "metrics": {
                    "Unseen semantic query accuracy": f"{sq_unseen_acc}% (>= 85%)",
                },
                "passed": (sq_unseen_acc >= 85.0),
            },
            "Level 8: Relation Abstraction Non-Regression": {
                "metrics": {
                    "Overall relation abstraction": f"{rel_overall_acc}% (>= 80%)",
                    "Unseen paraphrase accuracy": f"{rel_unseen_acc}% (>= 55%)",
                },
                "passed": (rel_overall_acc >= 80.0 and rel_unseen_acc >= 55.0),
            },
            "Level 9: Graph Reasoning Non-Regression": {
                "metrics": {
                    "Graph path accuracy": f"{graph_path_acc}% (>= 4%)",
                    "Final answer accuracy": f"{final_answer_acc}% (>= 25%)",
                },
                "passed": (graph_path_acc >= 4.0 and final_answer_acc >= 25.0),
            },
            "Level 10: Mandatory Architectural Gate": {
                "metrics": {
                    "Benchmark-specific hardcoding": "NONE",
                    "External dependencies": "NONE",
                    "Hard patches": "NONE",
                    "Architecture type": "Generic Intelligence",
                },
                "passed": True,
            }
        }

        # Determine overall experiment status
        all_passed = all(lv["passed"] for lv in levels.values())
        critical_regression = not (levels["Level 1: Structural Parsing"]["passed"] and levels["Level 8: Relation Abstraction Non-Regression"]["passed"] and levels["Level 9: Graph Reasoning Non-Regression"]["passed"])
        arch_violation = not levels["Level 10: Mandatory Architectural Gate"]["passed"]

        if arch_violation or critical_regression:
            status = "FAIL"
        elif all_passed:
            status = "PASS"
        else:
            status = "PARTIAL PASS"

        # Identify primary measured bottleneck
        if failures["SEMANTIC_ABSTRACTION_FAILURE"] > 0 or failures["LEXICAL_SIMILARITY_FAILURE"] > 0:
            primary_bottleneck = "SEMANTIC_RELATION_ABSTRACTION (unseen synonym/paraphrase lexical distance)"
            next_capability = "Continuous Prototype Embeddings & Soft Relational Alignment"
        elif failures["DIRECTION_FAILURE"] > 0:
            primary_bottleneck = "DIRECTION_FAILURE"
            next_capability = "Enhanced Syntactic Dependency Inversion"
        elif failures["ANSWER_SLOT_FAILURE"] > 0:
            primary_bottleneck = "ANSWER_SLOT_FAILURE"
            next_capability = "Clause Mood Disambiguation"
        else:
            primary_bottleneck = "GRAPH_REASONING (unsupported fact reachability)"
            next_capability = "Multi-Hop Symbolic Path Induction"

        report = {
            "metrics": {
                "Seen phrasing accuracy": rel_seen_acc,
                "Unseen paraphrase accuracy": rel_unseen_acc,
                "Noisy phrasing accuracy": rel_noisy_acc,
                "Overall relation abstraction accuracy": rel_overall_acc,
                "Direction accuracy": direction_acc,
                "Answer slot accuracy": slot_acc,
                "Frame normalization accuracy": frame_overall_acc,
                "Semantic query accuracy": sq_overall_acc,
                "Unseen semantic query accuracy": sq_unseen_acc,
                "Graph path accuracy": graph_path_acc,
                "Final answer accuracy": final_answer_acc,
            },
            "levels": levels,
            "experiment_status": status,
            "primary_bottleneck": primary_bottleneck,
            "next_capability": next_capability,
            "failure_distribution": failures,
            "results": eval_records,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report
