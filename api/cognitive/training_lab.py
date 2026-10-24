# pyright: ignore
"""
AR.I.S Cognitive Training Lab & Curriculum Generalization Engine.

Evaluates AR.I.S's zero-shot & few-shot cognitive generalization across 14 distinct curriculum tiers:
  Level 1:  Direct Fact Retrieval (1-hop seen)
  Level 2:  Unseen Entity Zero-Shot Transfer
  Level 3:  Unseen Relation Induction
  Level 4:  Reverse Directional Reasoning
  Level 5:  Multi-Hop Relational Deduction (2-hop transitive)
  Level 6:  3-Hop Deep Relational Deduction
  Level 7:  Structural Variable Parameter Binding
  Level 8:  Mathematical Arithmetic Substrate (deterministic eval)
  Level 9:  Algebraic Equation Root Solving (affine finite difference)
  Level 10: Positional / Coordinate Geometric Distance
  Level 11: Composite Relational + Mathematical Inference
  Level 12: Noise Tolerance & Distractor Robustness
  Level 13: Epistemic Boundary Respect (Unsupported query rejection)
  Level 14: Conflicting Fact Temporal Overwrite / Conflict Resolution

Strictly adheres to:
  - Rule 1 (Zero Hard Patches)
  - Rule 4 (Generic Cognitive Intelligence)
"""

import time
import uuid
from typing import Dict, Any, List, Optional
from api.linguistic.tokenizer import ArisTokenizer
from api.relational.relational import ArisRelationalBinder
from api.cognitive.cognitive_learner import ArisCognitiveReasoner, ArisCognitiveLearner
from api.semantic.executable_tools import ArisMathTool


class ArisTrainingLab:
    """
    Curriculum Runner and Generalization Benchmarker for AR.I.S.
    Tests model against both 'trained_on' seed facts and 'unseen' test prompts.
    """

    CURRICULUM_LEVELS = [
        {
            "level": 1,
            "name": "Direct Fact Retrieval (1-Hop Seen)",
            "description": "Evaluate ability to recall exact relational knowledge learned during sensory ingestion.",
            "seed_facts": ["Axinix founder is Loig"],
            "test_queries": [
                {"query": "Who is the founder of Axinix?", "expected_entity": "Loig", "type": "seen"}
            ]
        },
        {
            "level": 2,
            "name": "Unseen Entity Zero-Shot Transfer",
            "description": "Evaluate structural reasoning on novel entities never encountered in base ontology.",
            "seed_facts": ["Vanguard director is Elena"],
            "test_queries": [
                {"query": "Who is the director of Vanguard?", "expected_entity": "Elena", "type": "unseen_entity"}
            ]
        },
        {
            "level": 3,
            "name": "Unseen Relation Induction",
            "description": "Evaluate binding and extraction over arbitrary new predicate relations.",
            "seed_facts": ["Helios synthesizes Solarium"],
            "test_queries": [
                {"query": "What does Helios synthesize?", "expected_entity": "Solarium", "type": "unseen_relation"}
            ]
        },
        {
            "level": 4,
            "name": "Reverse Directional Reasoning",
            "description": "Infer inverse relation without explicit duplicate storage.",
            "seed_facts": ["Aria mentor is Marcus"],
            "test_queries": [
                {"query": "Who does Marcus mentor?", "expected_entity": "Aria", "type": "reverse_direction"}
            ]
        },
        {
            "level": 5,
            "name": "2-Hop Transitive Deduction",
            "description": "Multi-hop graph traversal across intermediate cognitive nodes.",
            "seed_facts": ["Kiran works_at NovaTech", "NovaTech located_in Berlin"],
            "test_queries": [
                {"query": "Where is Kiran's company located?", "expected_entity": "Berlin", "type": "multi_hop_2"}
            ]
        },
        {
            "level": 6,
            "name": "3-Hop Deep Relational Chain",
            "description": "Deep sequential deductive reasoning across 3 relational hops.",
            "seed_facts": ["Zara leads ProjectAlpha", "ProjectAlpha funded_by QuantumFund", "QuantumFund headquarters Tokyo"],
            "test_queries": [
                {"query": "Where is the headquarters of the fund backing Zara's project?", "expected_entity": "Tokyo", "type": "multi_hop_3"}
            ]
        },
        {
            "level": 7,
            "name": "Structural Variable Parameter Binding",
            "description": "Binding arbitrary named variables into functional executable arguments.",
            "seed_facts": ["A is 20", "B is 30"],
            "test_queries": [
                {"query": "Add A and B", "expected_contains": "50", "type": "param_binding"}
            ]
        },
        {
            "level": 8,
            "name": "Deterministic Arithmetic Substrate",
            "description": "AST mathematical computation across basic arithmetic operators.",
            "seed_facts": [],
            "test_queries": [
                {"query": "What is 45 * 12 + 60?", "expected_contains": "600", "type": "math_arithmetic"},
                {"query": "Compute 1000 - 250 * 3", "expected_contains": "250", "type": "math_arithmetic"}
            ]
        },
        {
            "level": 9,
            "name": "Algebraic Linear Equation Solving",
            "description": "Universal affine finite-difference root finding over arbitrary unknown variable.",
            "seed_facts": [],
            "test_queries": [
                {"query": "1 + x = 3", "expected_contains": "x = 2", "type": "math_algebra"},
                {"query": "100 - n = 37", "expected_contains": "n = 63", "type": "math_algebra"},
                {"query": "2 * y + 10 = 30", "expected_contains": "y = 10", "type": "math_algebra"}
            ]
        },
        {
            "level": 10,
            "name": "Positional / Geometric Distance",
            "description": "Pure structural coordinate computation between 2D/3D spatial embeddings.",
            "seed_facts": [],
            "test_queries": [
                {"query": "distance between (0, 0) and (3, 4)", "expected_contains": "5.0", "type": "spatial_distance"}
            ]
        },
        {
            "level": 11,
            "name": "Composite Relational + Math Inference",
            "description": "Extract numerical entity values from memory graph and feed to mathematical engine.",
            "seed_facts": ["ItemA price is 150", "ItemB price is 350"],
            "test_queries": [
                {"query": "What is the total price of ItemA and ItemB?", "expected_contains": "500", "type": "composite_rel_math"}
            ]
        },
        {
            "level": 12,
            "name": "Noise & Distractor Robustness",
            "description": "Retain correct entity extraction and routing amidst conversational filler words.",
            "seed_facts": ["Orion captain is Sarah"],
            "test_queries": [
                {"query": "Um, can you please tell me actually who the captain of Orion is right now?", "expected_entity": "Sarah", "type": "noise_robustness"}
            ]
        },
        {
            "level": 13,
            "name": "Epistemic Boundary Rejection",
            "description": "Properly identify ungrounded / unsupported queries without hallucinating.",
            "seed_facts": ["Earth orbits Sun"],
            "test_queries": [
                {"query": "Who is the secret emperor of Galaxy 99?", "expected_unsupported": True, "type": "epistemic_boundary"}
            ]
        },
        {
            "level": 14,
            "name": "Conflict Resolution & Temporal Update",
            "description": "Update existing belief state when superseded by newer evidence without stale residue.",
            "seed_facts": ["David role is Junior Developer", "David role is Principal Engineer"],
            "test_queries": [
                {"query": "What is David's role?", "expected_entity": "Principal Engineer", "type": "conflict_update"}
            ]
        }
    ]

    @classmethod
    def _build_fact(cls, statement: str) -> Dict[str, Any]:
        tokens = ArisTokenizer.tokenize(statement)
        candidates = ArisTokenizer.extract_candidates(statement, tokens)
        relations = ArisRelationalBinder.bind_relations(statement, tokens, candidates)
        return {
            "statement": statement,
            "tokens": tokens,
            "token_count": len(tokens),
            "candidates": candidates,
            "candidate_count": len(candidates),
            "relations": relations,
            "relation_count": len(relations),
            "confidence": 1.0,
            "topic": "training_curriculum",
        }

    def __init__(self):
        self.reasoner = ArisCognitiveReasoner()
        self.learner = ArisCognitiveLearner()

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Executes full 14-level curriculum benchmark suite.
        Returns aggregate scores, generalization curves, and granular level diagnostics.
        """
        start_time = time.time()
        level_results = []
        total_tests = 0
        total_passed = 0

        seen_total = 0
        seen_passed = 0
        unseen_total = 0
        unseen_passed = 0

        for level_cfg in self.CURRICULUM_LEVELS:
            lvl_num = level_cfg["level"]
            lvl_name = level_cfg["name"]
            lvl_desc = level_cfg["description"]
            seeds = level_cfg["seed_facts"]
            tests = level_cfg["test_queries"]

            # Fresh isolated memory environment for each level test
            level_facts = [self._build_fact(seed) for seed in seeds]

            lvl_passed = 0
            lvl_tests_count = len(tests)
            test_details = []

            for t in tests:
                total_tests += 1
                q = t["query"]
                q_type = t.get("type", "unseen")
                is_seen = (q_type == "seen")

                if is_seen:
                    seen_total += 1
                else:
                    unseen_total += 1

                # Execute cognitive reasoner
                t_start = time.time()
                result = self.reasoner.reason(q, facts=level_facts)
                t_latency = round((time.time() - t_start) * 1000, 2)

                answers = result.get("answers", [])
                ans_text = ""
                if answers and len(answers) > 0:
                    ans_text = str(answers[0].get("statement") or answers[0].get("answer") or "")
                elif "result" in result and result.get("result"):
                    ans_text = str(result["result"])

                trace = result.get("reasoning_trace", {})
                epistemic_cognition = trace.get("epistemic_cognition", {})
                epistemic_state = epistemic_cognition.get("state", "")

                passed = False
                reason_fail = ""

                if t.get("expected_unsupported"):
                    if epistemic_state in ["UNSUPPORTED", "ACQUIRABLE"] or "unsupported" in ans_text.lower() or "not have" in ans_text.lower() or "unknown" in ans_text.lower():
                        passed = True
                    else:
                        reason_fail = f"Expected UNSUPPORTED/ACQUIRABLE boundary, got {epistemic_state}: {ans_text}"
                elif t.get("expected_entity"):
                    exp = t["expected_entity"].lower()
                    if exp in ans_text.lower():
                        passed = True
                    else:
                        reason_fail = f"Expected entity '{t['expected_entity']}' in response '{ans_text}'"
                elif t.get("expected_contains"):
                    exp = t["expected_contains"].lower()
                    if exp in ans_text.lower():
                        passed = True
                    else:
                        reason_fail = f"Expected '{t['expected_contains']}' in response '{ans_text}'"

                if passed:
                    lvl_passed += 1
                    total_passed += 1
                    if is_seen:
                        seen_passed += 1
                    else:
                        unseen_passed += 1

                test_details.append({
                    "query": q,
                    "type": q_type,
                    "passed": passed,
                    "latency_ms": t_latency,
                    "epistemic_state": epistemic_state,
                    "answer": ans_text,
                    "failure_reason": reason_fail if not passed else None
                })

            lvl_accuracy = round((lvl_passed / lvl_tests_count) * 100, 1) if lvl_tests_count > 0 else 0.0

            level_results.append({
                "level": lvl_num,
                "name": lvl_name,
                "description": lvl_desc,
                "total_tests": lvl_tests_count,
                "passed_tests": lvl_passed,
                "accuracy": lvl_accuracy,
                "details": test_details
            })

        overall_accuracy = round((total_passed / total_tests) * 100, 1) if total_tests > 0 else 0.0
        seen_acc = round((seen_passed / seen_total) * 100, 1) if seen_total > 0 else 100.0
        unseen_acc = round((unseen_passed / unseen_total) * 100, 1) if unseen_total > 0 else 0.0
        total_duration = round(time.time() - start_time, 3)

        # Generalization curve points: [(level, accuracy)]
        curve = [{"level": r["level"], "name": r["name"], "accuracy": r["accuracy"]} for r in level_results]

        return {
            "summary": {
                "overall_accuracy": overall_accuracy,
                "total_tests": total_tests,
                "total_passed": total_passed,
                "seen_accuracy": seen_acc,
                "seen_total": seen_total,
                "seen_passed": seen_passed,
                "unseen_accuracy": unseen_acc,
                "unseen_total": unseen_total,
                "unseen_passed": unseen_passed,
                "total_duration_sec": total_duration,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "generalization_curve": curve,
            "levels": level_results
        }
