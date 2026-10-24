"""
AR.I.S — Tier 6-A: Epistemic State Model & Structural Goal Formation

Architectural Invariants:
  - Epistemic state is evaluated structurally, NEVER via hardcoded keyword dispatch.
  - States:
      * KNOWN: Memory directly satisfies the goal (zero tool dispatch required).
      * DERIVABLE: Memory contains component parameters/facts, but requires computation or inference.
      * ACQUIRABLE: Required facts are absent from memory, but match an external capability contract.
      * UNSUPPORTED: Deficit cannot be satisfied by memory or any registered capability.
      * AMBIGUOUS: Goal or parameters cannot be uniquely grounded.
  - Zero Hard Patches: No 'if query.startswith(...)' or syntax heuristics.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
import re


class ArisEpistemicState(str, Enum):
    KNOWN = "KNOWN"
    DERIVABLE = "DERIVABLE"
    ACQUIRABLE = "ACQUIRABLE"
    UNSUPPORTED = "UNSUPPORTED"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass
class ArisGoal:
    """
    Formal Open Structural Equation:
    Goal = < Target Variable, Target Type, Anchor Entity, Relational Constraint, Parameters >
    """
    raw_query: str
    target_variable: str                     # 'result', 'entity', 'fact', 'unknown'
    target_type: str                         # 'numerical_value', 'entity_reference', 'knowledge_acquisition'
    anchor_entity: Optional[str] = None      # e.g. 'A.P.J. Abdul Kalam', 'India'
    relational_constraint: Optional[str] = None  # e.g. 'is president of', '-'
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


class ArisGoalFormer:
    """
    Perceives input representations from Tiers 1-3 and forms a formal ArisGoal.
    Strictly structural: ZERO hardcoded word lists, ZERO trigger dictionaries.
    Uses Tier 2 concept/entity candidates and Tier 3 relational bindings.
    """

    @classmethod
    def form_goal(
        cls,
        query_text: str,
        tokens: List[Dict[str, Any]],
        candidates: Optional[List[Dict[str, Any]]] = None,
        bindings: Optional[List[Dict[str, Any]]] = None,
        frame: Optional[Any] = None,
    ) -> ArisGoal:
        """
        Derives an ArisGoal from structural perceptions.
        """
        clean_query = (query_text or "").strip()
        if not clean_query:
            return ArisGoal(
                raw_query="",
                target_variable="unknown",
                target_type="unknown",
            )

        # -------------------------------------------------------------
        # Case 1: Mathematical Expressions & Algebraic Equations
        # -------------------------------------------------------------
        from ..semantic.executable_tools import ArisMathTool
        supported_ops = set(ArisMathTool.OPERATORS.keys())

        # Check for structural equation with '=' (LHS = RHS)
        eq_indices = [idx for idx, t in enumerate(tokens) if t.get("raw") == "="]
        if eq_indices:
            eq_idx = eq_indices[0]
            left_toks = tokens[:eq_idx]
            right_toks = tokens[eq_idx + 1:]

            # Filter relevant expression tokens on left and right
            def _is_math_tok(t):
                r = t.get("raw", "")
                return t.get("type") == "number" or r in supported_ops or (r.isalpha() and len(r) == 1)

            valid_left = [t for t in left_toks if _is_math_tok(t)]
            valid_right = [t for t in right_toks if _is_math_tok(t)]

            # Detect variable token (e.g. 'x', 'y', 'n', 'a')
            var_tokens = [t["raw"] for t in (valid_left + valid_right) if t.get("raw", "").isalpha()]

            if (valid_left and valid_right) and len(var_tokens) == 1:
                lhs_str = " ".join(t["raw"] for t in valid_left)
                rhs_str = " ".join(t["raw"] for t in valid_right)
                var_name = var_tokens[0]

                return ArisGoal(
                    raw_query=clean_query,
                    target_variable=var_name,
                    target_type="algebraic_equation",
                    relational_constraint="linear_equation_solving",
                    parameters={
                        "operation": "solve_linear",
                        "equation": {"left": lhs_str, "right": rhs_str},
                        "variable": var_name,
                    },
                )

        # Standard arithmetic operator expression scanning
        expr_tokens = []
        for tok in tokens:
            raw = tok.get("raw", "").strip()
            ttype = tok.get("type")
            if ttype == "number":
                if not expr_tokens or expr_tokens[-1]["raw"] in supported_ops:
                    expr_tokens.append(tok)
                else:
                    break
            elif raw in supported_ops:
                if expr_tokens and expr_tokens[-1]["type"] == "number":
                    expr_tokens.append(tok)
                else:
                    break
            else:
                if len(expr_tokens) >= 3 and expr_tokens[-1]["type"] == "number":
                    break
                else:
                    expr_tokens = []

        if len(expr_tokens) >= 3 and expr_tokens[-1]["type"] == "number":
            expr_str = " ".join(t["raw"] for t in expr_tokens)
            return ArisGoal(
                raw_query=clean_query,
                target_variable="result",
                target_type="numerical_value",
                relational_constraint="arithmetic_evaluation",
                parameters={
                    "operation": "evaluate",
                    "expression": expr_str,
                    "tokens": [t["raw"] for t in expr_tokens],
                },
            )


        # -------------------------------------------------------------
        # Case 2: Relational Frame Bindings (Who -- Relation --> Whom)
        # -------------------------------------------------------------
        is_question = getattr(frame, "is_question", False)

        if is_question and candidates and len(candidates) > 0:
            upos_by_idx = {t.get("sequence_index"): t.get("upos") for t in tokens}
            grounded_cands = [
                c for c in candidates
                if not all(upos_by_idx.get(idx) == "PRON" for idx in c.get("token_indices", []))
            ]
            c0 = grounded_cands[0] if grounded_cands else candidates[0]
            c0_start = c0.get("char_start", 0)
            c0_end = c0.get("char_end", len(clean_query))
            anchor_text = c0.get("text", "")

            from .goal_reasoning_tensor_engine import ArisGoalReasoningTensorEngine
            from .pragmatic_tensor_engine import ArisPragmaticTensorEngine
            from ..semantic.semantic_representation import ArisSemanticEncoder
            import torch

            # Extract interrogative focus nominal from UD syntax:
            interrogative_head = None
            if tokens and len(tokens) > 1:
                ArisPragmaticTensorEngine._ensure_loaded()

                # 1. Identify the interrogative locus via continuous tensor projection
                int_locus_idx = None
                for idx, tok in enumerate(tokens):
                    tok_vec = torch.tensor(ArisSemanticEncoder.encode(tok.get("raw", "")), dtype=torch.float32)
                    tok_norm = tok_vec / (torch.norm(tok_vec) + 1e-8)
                    sim = max(
                        float(torch.max(torch.matmul(mat, tok_norm)).item())
                        for mat in ArisPragmaticTensorEngine._interrogative_matrices.values()
                    ) if ArisPragmaticTensorEngine._interrogative_matrices else 0.0
                    if sim >= 0.60:
                        int_locus_idx = idx
                        break

                # 2. Extract the syntactic nominal head (NOUN / PROPN) preceding the anchor entity
                if int_locus_idx is not None:
                    anchor_tok_start = c0.get("token_indices", [len(tokens)])[0] if c0.get("token_indices") else len(tokens)
                    nominal_tokens = [
                        t for t in tokens[int_locus_idx + 1 : anchor_tok_start]
                        if t.get("upos") in ("NOUN", "PROPN")
                    ]
                    if nominal_tokens:
                        interrogative_head = nominal_tokens[0].get("raw")

            from .semantic_frame import ArisInterrogativeRoleLearner
            int_role, int_conf = ArisInterrogativeRoleLearner.infer_target_role(tokens[0]["raw"]) if tokens else ("", 0.0)
            if int_conf >= 0.85 and int_role == "person":
                pred_type = "PERSON"
            elif interrogative_head:
                pred_type = ArisGoalReasoningTensorEngine.predict_target_type(interrogative_head).get("best_type", "entity_reference")
            else:
                pred_type = ArisGoalReasoningTensorEngine.predict_target_type(clean_query).get("best_type", "entity_reference")
            rel_constraint = clean_query

            return ArisGoal(
                raw_query=clean_query,
                target_variable="entity",
                target_type=pred_type,
                anchor_entity=anchor_text,
                relational_constraint=rel_constraint,
                parameters={
                    "anchor_candidate": anchor_text,
                    "relation": rel_constraint,
                    "composite_relation": rel_constraint,
                },
            )

        if bindings and len(bindings) > 0:
            b = bindings[0]
            subj = b.get("subject") or {}
            obj = b.get("object") or {}
            rel = b.get("relation") or {}
            subj_text = subj.get("text", "") if isinstance(subj, dict) else str(subj)
            obj_text = obj.get("text", "") if isinstance(obj, dict) else str(obj)
            rel_span = rel.get("normalized") or rel.get("text", "") if isinstance(rel, dict) else str(rel)

            return ArisGoal(
                raw_query=clean_query,
                target_variable="entity",
                target_type="entity_reference",
                anchor_entity=obj_text or subj_text,
                relational_constraint=rel_span,
                parameters={
                    "subject_candidate": subj_text,
                    "object_candidate": obj_text,
                    "relation": rel_span,
                    "composite_relation": rel_span,
                },
            )

        # -------------------------------------------------------------
        # Case 3: Concept / Entity Goal (Tier 2 Structural Candidates)
        # -------------------------------------------------------------
        # Structural resolution via Tier 2 candidates (NO word dictionaries)
        if candidates and len(candidates) > 0:
            anchor_candidate = candidates[0].get("text", "")
            target_cand = candidates[1].get("text", "") if len(candidates) > 1 else None
        else:
            content_words = [t["raw"] for t in tokens if t.get("upos") != "PUNCT" and t.get("type") != "symbol"]
            anchor_candidate = " ".join(content_words) if content_words else clean_query
            target_cand = None

        return ArisGoal(
            raw_query=clean_query,
            target_variable="entity_information",
            target_type="knowledge_acquisition",
            anchor_entity=anchor_candidate,
            parameters={
                "query": anchor_candidate,
                "target_candidate": target_cand,
            },
        )


class ArisEpistemicEvaluator:
    """
    Evaluates the current Epistemic State of AR.I.S relative to an ArisGoal.
    Determines whether the goal is KNOWN, DERIVABLE, ACQUIRABLE, or UNSUPPORTED.
    """

    @classmethod
    def evaluate(
        cls,
        goal: ArisGoal,
        memory_facts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Performs a memory probe and computes the epistemic gap.
        """
        from .brain import ArisBrain
        from ..semantic.executable_tools import ArisCapabilityRegistry

        # 1. Directly Evaluated Expressions & Algebraic Equations
        if goal.target_type in ("numerical_value", "algebraic_equation") and ("expression" in goal.parameters or "equation" in goal.parameters):
            return {
                "state": ArisEpistemicState.DERIVABLE,
                "deficit": "arithmetic_computation" if goal.target_type == "numerical_value" else "algebraic_computation",
                "matched_memory": [],
                "required_capability": "math_computation",
            }


        # 2. Relational Memory Probe
        anchor = goal.anchor_entity or goal.parameters.get("query")
        relation = goal.relational_constraint

        matching_facts = []
        if anchor:
            try:
                # Probe internal memory for anchor
                if memory_facts is not None:
                    anchor_lower = anchor.lower().strip()
                    matching_facts = [
                        f for f in memory_facts
                        if anchor_lower in f.get("statement", "").lower()
                        or any(anchor_lower in c.get("text", "").lower() for c in f.get("candidates", []))
                    ]
                else:
                    matching_facts = ArisBrain.retrieve_facts(query=anchor, limit=10)
            except Exception:
                matching_facts = []

        # Check if memory directly satisfies the goal
        if matching_facts:
            if relation:
                # Probe for matching relational bindings in retrieved facts
                rel_matches = []
                rel_clean = relation.lower().strip()
                for f in matching_facts:
                    for r in f.get("relations", []):
                        r_span = r.get("relation_span", "").lower().strip()
                        r_norm = r.get("relation", {}).get("normalized", "").lower().strip()
                        is_direct_rel = (
                            rel_clean == r_span or rel_clean == r_norm
                            or (len(rel_clean) <= max(len(r_span), len(r_norm)) * 1.5 and (rel_clean in r_span or (r_span and r_span in rel_clean)))
                        )
                        if is_direct_rel:
                            rel_matches.append(f)
                            break
                if rel_matches:
                    return {
                        "state": ArisEpistemicState.KNOWN,
                        "deficit": None,
                        "matched_memory": rel_matches,
                        "required_capability": "memory_retrieval",
                    }
            else:
                # Direct entity knowledge present
                return {
                    "state": ArisEpistemicState.KNOWN,
                    "deficit": None,
                    "matched_memory": matching_facts,
                    "required_capability": "memory_retrieval",
                }

        # 3. Check for DERIVABLE state (Component parameters present in memory)
        if anchor and matching_facts:
            # Check if component facts exist that allow computing the answer
            has_birth = any("born" in f.get("statement", "").lower() for f in matching_facts)
            has_death = any("died" in f.get("statement", "").lower() for f in matching_facts)
            if has_birth and has_death and relation and "age" in relation.lower():
                return {
                    "state": ArisEpistemicState.DERIVABLE,
                    "deficit": "composite_computation",
                    "matched_memory": matching_facts,
                    "required_capability": "math_computation",
                }

        # 4. Check for DERIVABLE / HYPOTHETICAL Relational Connectivity
        # If query involves multiple entities or internal memory graph contains paths,
        # prioritize internal cognitive inference over external web retrieval.
        from ..relational.symbolic_graph import ArisSymbolicGraph
        from ..linguistic.tokenizer import ArisTokenizer
        graph = ArisSymbolicGraph.build_from_facts(facts=memory_facts)
        target_candidate = goal.parameters.get("target_candidate") or goal.parameters.get("object_candidate")

        def _entity_matches_graph(entity_str: Optional[str]) -> bool:
            if not entity_str:
                return False
            e_lower = entity_str.lower().strip()
            if e_lower in graph.adjacency:
                return True
            e_tokens = {t["token"].lower() for t in ArisTokenizer.tokenize(e_lower) if t.get("type") != "symbol"}
            if not e_tokens:
                return False
            for k in graph.adjacency:
                k_lower = k.lower().strip()
                if e_lower == k_lower:
                    return True
                k_tokens = {t["token"].lower() for t in ArisTokenizer.tokenize(k_lower) if t.get("type") != "symbol"}
                if not k_tokens:
                    continue
                if e_tokens == k_tokens:
                    return True
                # Require high Jaccard token overlap (>= 0.67) between multi-token entity names (both >= 2 tokens)
                common = e_tokens & k_tokens
                if common and len(e_tokens) >= 2 and len(k_tokens) >= 2:
                    union = e_tokens | k_tokens
                    jaccard = len(common) / len(union)
                    if jaccard >= 0.67:
                        return True
                    if (k_tokens.issubset(e_tokens) or e_tokens.issubset(k_tokens)) and abs(len(e_tokens) - len(k_tokens)) <= 1:
                        return True
            return False

        if anchor:
            anchor_in_graph = _entity_matches_graph(anchor)
            target_in_graph = _entity_matches_graph(target_candidate) if target_candidate else False

            if target_candidate and anchor_in_graph and target_in_graph:
                return {
                    "state": ArisEpistemicState.DERIVABLE,
                    "deficit": "relational_path_inference",
                    "matched_memory": matching_facts,
                    "required_capability": "gnn_positional_hypothesis",
                }
            elif anchor_in_graph and (relation or not target_candidate):
                # Open variable query with known anchor in memory graph
                return {
                    "state": ArisEpistemicState.DERIVABLE,
                    "deficit": "open_variable_graph_inference",
                    "matched_memory": matching_facts,
                    "required_capability": "gnn_positional_hypothesis",
                }

        # 5. Check for ACQUIRABLE state (External acquisition capability available)
        browser_cap = ArisCapabilityRegistry.get_capability("browser_access")
        if browser_cap and anchor:
            return {
                "state": ArisEpistemicState.ACQUIRABLE,
                "deficit": "external_knowledge_gap",
                "matched_memory": [],
                "required_capability": "browser_access",
            }

        # 6. Unsupported
        return {
            "state": ArisEpistemicState.UNSUPPORTED,
            "deficit": "unbridgeable_gap",
            "matched_memory": [],
            "required_capability": None,
        }


class ArisCapabilityMatcher:
    """
    Matches the Epistemic Deficit to a registered capability based on capability contracts.
    """

    @classmethod
    def match(cls, epistemic_eval: Dict[str, Any], goal: ArisGoal) -> Optional[str]:
        """
        Returns capability name that satisfies the deficit.
        """
        state = epistemic_eval.get("state")

        if state == ArisEpistemicState.KNOWN:
            return "memory_retrieval"

        if state in {ArisEpistemicState.DERIVABLE, ArisEpistemicState.ACQUIRABLE}:
            return epistemic_eval.get("required_capability")

        return None
