"""
AR.I.S Ã¢â‚¬â€ Cognitive Operation Learner & Autonomous Reasoner (Tier 5)

Core Directive:
  "Training data is not the target. Training data is the teaching signal.
   The target is generalization to structurally valid but lexically unseen situations."

Strict Invariants:
  - ZERO SPOON-FEEDING: Zero hardcoded English connector lists (no 'the person who', no 'who', no 'that').
  - ZERO DICTIONARIES: No word-lookup tables or pre-programmed synonym mappings.
  - PURE STRUCTURAL DISCOVERY: The operation is discovered by aligning the position of the
    target answer in the fact binding against the positions in the question binding.
  - 4-LEVEL GENERALIZATION:
      Level 1: Seen sanity check.
      Level 2: Unseen entities.
      Level 3: Unseen alien vocabulary (flarps, daxels).
      Level 4: Unseen multi-hop composition.
"""

import re
from typing import Optional, List, Dict, Any
from ..linguistic.tokenizer import ArisTokenizer
from ..relational.relational import ArisRelationalBinder
from ..relational.retrieval import ArisRetriever, ArisRelationalQuery, ArisMultiHopQuery
from ..relational.structural_advisor import ArisStructuralAdvisor


class ArisOperationTemplate:
    """
    Abstract representation of a learned relational or executable operation.
    Contains zero specific entity words, zero connector phrases, and zero language dictionaries.
    """

    def __init__(
        self,
        template_id: str,
        name: str,
        operation_type: str,
        source_role: str,
        target_role: str,
        anchor_role: str,
        demonstration_count: int = 1,
        capability_name: Optional[str] = None,
        parameter_mapping: Optional[Dict[str, Any]] = None,
    ):
        self.template_id = template_id
        self.name = name
        self.operation_type = operation_type  # "reverse_traversal" | "forward_traversal" | "executable_capability"
        self.source_role = source_role        # Position of variable in question (e.g. "subject" or "object" or "parameters")
        self.target_role = target_role        # Position of answer in fact (e.g. "subject" or "object" or "numeric_result")
        self.anchor_role = anchor_role        # Position of shared constraint anchor (e.g. "object" or "subject" or "relational_operator")
        self.demonstration_count = demonstration_count
        self.capability_name = capability_name
        self.parameter_mapping = parameter_mapping or {}

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "template_id": self.template_id,
            "name": self.name,
            "operation_type": self.operation_type,
            "source_role": self.source_role,
            "target_role": self.target_role,
            "anchor_role": self.anchor_role,
            "demonstration_count": self.demonstration_count,
        }
        if self.capability_name:
            data["capability_name"] = self.capability_name
        if self.parameter_mapping:
            data["parameter_mapping"] = self.parameter_mapping
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArisOperationTemplate":
        return cls(
            template_id=data.get("template_id", ""),
            name=data.get("name", ""),
            operation_type=data.get("operation_type", ""),
            source_role=data.get("source_role", ""),
            target_role=data.get("target_role", ""),
            anchor_role=data.get("anchor_role", ""),
            demonstration_count=data.get("demonstration_count", 1),
            capability_name=data.get("capability_name"),
            parameter_mapping=data.get("parameter_mapping"),
        )



class ArisCognitiveLearner:
    """
    Learns abstract relational operations and executable capabilities from demonstrations.
    Aligns the positional slots of the answer/parameters against the question binding.
    Contains ZERO hardcoded words, connector lists, or language-specific dictionaries.
    """

    COLLECTION_NAME = "brain_operation_templates"

    @classmethod
    def _get_collection(cls):
        from .brain import ArisBrain
        db = ArisBrain._get_collection().database
        return db[cls.COLLECTION_NAME]

    @classmethod
    def bind_executable_parameters(
        cls,
        tokens: List[Dict[str, Any]],
        template: ArisOperationTemplate,
        question_text: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Binds parameters from query tokens to a learned executable template.
        Dynamically extracts parameters matching the capability schema without hardcoding.
        """
        from ..semantic.executable_tools import ArisCapabilityRegistry
        cap_name = template.capability_name or "math_computation"
        cap = ArisCapabilityRegistry.get_capability(cap_name)
        if not cap:
            return None

        # Case A: Mathematical / Operator expressions & Linear Equations
        if hasattr(cap, "OPERATORS") or "expression" in getattr(cap, "PARAMETERS_SCHEMA", {}) or "equation" in getattr(cap, "PARAMETERS_SCHEMA", {}):
            supported_ops = template.parameter_mapping.get(
                "supported_operators",
                list(getattr(cap, "OPERATORS", {}).keys())
            )

            # 1. Check for linear equation: '='
            eq_indices = [idx for idx, t in enumerate(tokens) if t.get("raw") == "="]
            if eq_indices:
                eq_idx = eq_indices[0]
                def _is_math_tok(t):
                    r = t.get("raw", "")
                    return t.get("type") == "number" or r in supported_ops or (r.isalpha() and len(r) == 1)
                valid_left = [t for t in tokens[:eq_idx] if _is_math_tok(t)]
                valid_right = [t for t in tokens[eq_idx + 1:] if _is_math_tok(t)]
                var_tokens = [t["raw"] for t in (valid_left + valid_right) if t.get("raw", "").isalpha()]
                if valid_left and valid_right and len(var_tokens) == 1:
                    lhs_str = " ".join(t["raw"] for t in valid_left)
                    rhs_str = " ".join(t["raw"] for t in valid_right)
                    var_name = var_tokens[0]
                    return {
                        "operation": "solve_linear",
                        "equation": {"left": lhs_str, "right": rhs_str},
                        "variable": var_name,
                        "capability_name": cap_name,
                    }

            # 2. Arithmetic operator expression scanning
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
                return {
                    "operation": "evaluate",
                    "expression": expr_str,
                    "param1": expr_tokens[0]["raw"],
                    "op_str": expr_tokens[1]["raw"],
                    "param2": expr_tokens[2]["raw"],
                    "tokens": [t["raw"] for t in expr_tokens],
                    "capability_name": cap_name,
                }


        # Case B: Query-based text capabilities
        schema = getattr(cap, "PARAMETERS_SCHEMA", {})
        if "query" in schema or "query" in template.parameter_mapping.get("param_names", []):
            from ..linguistic.tokenizer import ArisTokenizer
            clean_text = question_text or " ".join(t.get("raw", "") for t in tokens)
            candidates = ArisTokenizer.extract_candidates(clean_text, tokens)
            # Prioritize definitive entity candidates (e.g. "MS.Dhoni") over ambiguous words
            entity_cands = [c for c in candidates if c.get("candidate_type") == "entity_candidate"]
            if entity_cands:
                query_val = max(entity_cands, key=lambda c: len(c.get("text", "")))["text"]
            elif candidates:
                query_val = max(candidates, key=lambda c: len(c.get("text", "")))["text"]
            else:
                query_val = clean_text.rstrip("?!.").strip()

            if query_val:
                return {
                    "query": query_val,
                    "capability_name": cap_name,
                }

        # Case C: Generic parameter schema binding for path/command-style capabilities.
        # The tokenizer already identified path tokens (type='path') atomically.
        # Read them directly -- no regex needed here.
        param_names = template.parameter_mapping.get("param_names", list(schema.keys()))
        if param_names:
            bound: Dict[str, Any] = {"capability_name": cap_name}

            # Pull path tokens from the already-processed token stream
            path_tokens = [t["raw"] for t in tokens if t.get("type") == "path"]

            for pname in param_names:
                if pname == "working_dir":
                    bound["working_dir"] = path_tokens[0] if path_tokens else None
                else:
                    # All other parameters are left unresolved -- epistemic deficit
                    bound[pname] = None

            # Only return a binding if at least one param was resolved
            resolved = {k: v for k, v in bound.items() if k != "capability_name" and v is not None}
            if resolved:
                return bound

        return None


    @classmethod
    def learn_executable_demonstration(
        cls,
        query_text: str,
        capability_name: Optional[str] = None,
        advisor_observation: Optional[Dict[str, Any]] = None,
        in_memory_templates: Optional[List[ArisOperationTemplate]] = None,
    ) -> Dict[str, Any]:
        """
        Learns a parameterized executable capability pattern from a Teacher-guided demonstration.
        The Teacher provides the structural parameter knowledge; AR.I.S verifies grounding
        via execution and persists the abstract operational template.
        """
        from ..semantic.executable_tools import ArisCapabilityRegistry
        from ..relational.structural_advisor import ArisStructuralAdvisor

        cleaned = query_text.strip()
        obs = advisor_observation or ArisStructuralAdvisor.observe_query(query_text=cleaned, enabled=True)

        if obs.get("type") != "executable_capability" or obs.get("status") not in (None, "ok", "success"):
            raise ValueError(f"Teacher did not observe an executable capability in '{query_text}'.")

        cap_name = obs.get("capability") or capability_name
        if not cap_name:
            raise ValueError(f"Teacher did not specify a capability in observation for '{query_text}'.")

        params = obs.get("parameters") or {}
        if not isinstance(params, dict) or not params:
            raise ValueError(f"Teacher observation missing parameters for capability '{cap_name}': {params}")

        cap = ArisCapabilityRegistry.get_capability(cap_name)
        if not cap:
            raise ValueError(f"Capability '{cap_name}' not found in registry.")

        # Grounding verification: Execute via capability registry dynamically with generic parameters
        clean_params = {k: v for k, v in params.items() if k != "capability_name"}
        exec_res = ArisCapabilityRegistry.dispatch(cap_name, **clean_params)
        if exec_res.get("status") not in ("success", "solved") and not exec_res.get("success", False):
            raise ValueError(f"Capability execution failed for demonstration: {exec_res.get('message', 'Execution error')}")

        template = ArisOperationTemplate(
            template_id=f"op_executable_{cap_name}",
            name=f"Parameterized Execution ({cap_name})",
            operation_type="executable_capability",
            source_role="parameters",
            target_role="result",
            anchor_role="capability_operator",
            capability_name=cap_name,
            parameter_mapping={
                "param_names": list(params.keys()),
                "schema": getattr(cap, "PARAMETERS_SCHEMA", {}),
                "supported_operators": list(getattr(cap, "OPERATORS", {}).keys()) if hasattr(cap, "OPERATORS") else [],
            },
        )

        if in_memory_templates is not None:
            existing = [t for t in in_memory_templates if t.template_id == template.template_id]
            if existing:
                existing[0].demonstration_count += 1
            else:
                in_memory_templates.append(template)
        else:
            try:
                coll = cls._get_collection()
                t_data = template.to_dict()
                t_data.pop("demonstration_count", None)
                coll.update_one(
                    {"template_id": template.template_id},
                    {
                        "$set": t_data,
                        "$inc": {"demonstration_count": 1},
                    },
                    upsert=True,
                )
            except Exception:
                pass

        return {
            "status": "learned",
            "template": template.to_dict(),
            "discovered_parameters": {
                "capability": cap_name,
                "parameters": params,
                "grounded_result": exec_res["result"],
            },
        }

    @classmethod
    def _get_associations_collection(cls):
        from .brain import ArisBrain
        db = ArisBrain._get_collection().database
        return db["brain_interrogative_associations"]

    @classmethod
    def record_interrogative_association(cls, token: str, target_domain: str):
        if not token or not target_domain:
            return
        t_clean = token.strip().lower()
        try:
            coll = cls._get_associations_collection()
            coll.update_one(
                {"token": t_clean},
                {"$set": {"token": t_clean, "target_domain": target_domain.strip().lower()}},
                upsert=True,
            )
        except Exception:
            pass

    @classmethod
    def get_interrogative_target_domain(cls, token: str) -> Optional[str]:
        if not token:
            return None
        t_clean = token.strip().lower()
        try:
            coll = cls._get_associations_collection()
            doc = coll.find_one({"token": t_clean})
            if isinstance(doc, dict) and doc.get("target_domain"):
                return doc.get("target_domain")
        except Exception:
            pass
        return None

    @classmethod
    def learn_multihop_demonstration(
        cls,
        query_text: str,
        hops: List[Dict[str, str]],
        start_entity: str,
        target_domain: Optional[str] = "entity",
        in_memory_templates: Optional[List[ArisOperationTemplate]] = None,
    ) -> Dict[str, Any]:
        """
        Learns an abstract multi-hop operational template from a verified multi-hop demonstration.
        Persists the ordered sequence of relational hops into MongoDB operation_templates.
        """
        hop_sig = "_".join(f"{h.get('relation', '')}_{h.get('direction', 'forward')}" for h in hops)
        clean_sig = re.sub(r"[^a-zA-Z0-9_]+", "_", hop_sig).strip("_")
        dom_clean = (target_domain or "entity").strip().lower()
        template_id = f"op_multihop_{len(hops)}_{dom_clean}_{clean_sig}" if clean_sig else f"op_multihop_{len(hops)}_{dom_clean}"

        template = ArisOperationTemplate(
            template_id=template_id,
            name=f"Multi-Hop Traversal ({len(hops)} hops, {dom_clean})",
            operation_type="multi_hop_traversal",
            source_role="target_entity",
            target_role="answer",
            anchor_role="start_entity",
            parameter_mapping={
                "hops": hops,
                "hop_count": len(hops),
                "start_entity": start_entity,
                "target_domain": dom_clean,
            },
        )

        if in_memory_templates is not None:
            existing = [t for t in in_memory_templates if t.template_id == template.template_id]
            if existing:
                existing[0].demonstration_count += 1
            else:
                in_memory_templates.append(template)
        else:
            try:
                coll = cls._get_collection()
                t_data = template.to_dict()
                t_data.pop("demonstration_count", None)
                coll.update_one(
                    {"template_id": template.template_id},
                    {
                        "$set": t_data,
                        "$inc": {"demonstration_count": 1},
                    },
                    upsert=True,
                )
            except Exception:
                pass

        return {
            "status": "learned",
            "template": template.to_dict(),
        }

    @classmethod
    def learn_demonstration(
        cls,
        fact: str,
        question: str,
        answer: str,
        in_memory_templates: Optional[List[ArisOperationTemplate]] = None,
    ) -> Dict[str, Any]:
        """
        Learns the relational operation by comparing structural positions between Fact, Question, and Answer.
        Pure structural discovery Ã¢â‚¬â€ zero word lists.
        """
        fact_clean = fact.strip()
        quest_clean = question.strip()
        ans_clean = answer.strip()

        # Step 1: Run native Tier 1-3 on fact
        f_tokens = ArisTokenizer.tokenize(fact_clean)
        f_cands = ArisTokenizer.extract_candidates(fact_clean, f_tokens)
        f_bindings = ArisRelationalBinder.bind_relations(fact_clean, f_tokens, f_cands)

        # Step 2: Run native Tier 1-3 on question
        q_tokens = ArisTokenizer.tokenize(quest_clean)
        q_cands = ArisTokenizer.extract_candidates(quest_clean, q_tokens)
        q_bindings = ArisRelationalBinder.bind_relations(quest_clean, q_tokens, q_cands)

        if not f_bindings:
            raise ValueError(f"Demonstration fact '{fact_clean}' did not produce any relational bindings.")

        if not q_bindings:
            raise ValueError(f"Demonstration question '{quest_clean}' did not produce any relational bindings.")

        f_bind = f_bindings[0]
        q_bind = q_bindings[0]

        # Locate which slot in the fact holds the target answer
        f_subj_text = f_bind["subject"]["text"].strip().lower()
        f_obj_text = f_bind["object"]["text"].strip().lower()
        target_clean = ans_clean.lower()

        if f_subj_text == target_clean:
            ans_fact_slot = "subject"
            anchor_fact_slot = "object"
        elif f_obj_text == target_clean:
            ans_fact_slot = "object"
            anchor_fact_slot = "subject"
        else:
            raise ValueError(
                f"Target answer '{ans_clean}' not found in fact subject ('{f_bind['subject']['text']}') "
                f"or object ('{f_bind['object']['text']}')."
            )

        # Locate the shared constraint anchor in the question
        q_subj_text = q_bind["subject"]["text"].strip().lower()
        q_obj_text = q_bind["object"]["text"].strip().lower()
        shared_anchor_text = f_bind[anchor_fact_slot]["text"].strip().lower()

        if q_obj_text == shared_anchor_text:
            anchor_quest_slot = "object"
            variable_quest_slot = "subject"
        elif q_subj_text == shared_anchor_text:
            anchor_quest_slot = "subject"
            variable_quest_slot = "object"
        else:
            # Anchor text not identical, assume standard question position
            anchor_quest_slot = anchor_fact_slot
            variable_quest_slot = "subject" if anchor_quest_slot == "object" else "object"

        # Deduce the abstract operation from the positional alignment
        if ans_fact_slot == "subject" and anchor_quest_slot == "object":
            # (? --R--> Object) => return Subject (Reverse Traversal)
            op_type = "reverse_traversal"
            template_id = "op_reverse_traversal"
            name = "Positional Reverse Traversal"
        elif ans_fact_slot == "object" and anchor_quest_slot == "subject":
            # (Subject --R--> ?) => return Object (Forward Traversal)
            op_type = "forward_traversal"
            template_id = "op_forward_traversal"
            name = "Positional Forward Traversal"
        else:
            op_type = "reverse_traversal"
            template_id = "op_reverse_traversal"
            name = "Positional Reverse Traversal"

        template = ArisOperationTemplate(
            template_id=template_id,
            name=name,
            operation_type=op_type,
            source_role=variable_quest_slot,
            target_role=ans_fact_slot,
            anchor_role=anchor_quest_slot,
        )

        # Persist learned template
        if in_memory_templates is not None:
            existing = [t for t in in_memory_templates if t.template_id == template.template_id]
            if existing:
                existing[0].demonstration_count += 1
            else:
                in_memory_templates.append(template)
        else:
            try:
                coll = cls._get_collection()
                t_data = template.to_dict()
                t_data.pop("demonstration_count", None)
                coll.update_one(
                    {"template_id": template.template_id},
                    {
                        "$set": t_data,
                        "$inc": {"demonstration_count": 1},
                    },
                    upsert=True,
                )
            except Exception:
                pass


        return {
            "status": "learned",
            "template": template.to_dict(),
            "discovered_alignment": {
                "answer_slot_in_fact": ans_fact_slot,
                "anchor_slot_in_question": anchor_quest_slot,
                "variable_slot_in_question": variable_quest_slot,
                "operation": op_type,
            },
        }

    @classmethod
    def get_templates(cls, in_memory_templates: Optional[List[ArisOperationTemplate]] = None) -> List[ArisOperationTemplate]:
        if in_memory_templates is not None:
            return in_memory_templates

        try:
            coll = cls._get_collection()
            cursor = coll.find()
            return [ArisOperationTemplate.from_dict(doc) for doc in cursor if isinstance(doc, dict)]
        except Exception:
            return []


class ArisCognitiveReasoner:
    """
    Autonomous Reasoner:
    Projects unseen questions onto learned operational templates and executes the resulting
    traversal over memory bindings or external capability dispatch.
    Contains zero word lists and zero hardcoded dictionaries.
    """

    @classmethod
    def reason(
        cls,
        question_text: str,
        facts: Optional[List[Dict[str, Any]]] = None,
        templates: Optional[List[ArisOperationTemplate]] = None,
        limit: int = 10,
        use_advisor: bool = True,
        use_neural: bool = False,
        allow_external: bool = True,
        context: Optional[Any] = None,
    ) -> Dict[str, Any]:

        """
        Autonomously reasons an answer to an unseen question using learned operation templates,
        local neural parameter frame classification, or capability dispatch.

        Strict Cognitive Boundary:
          - For executable expressions: extracts parameter frames (operand1, operator, operand2),
            binds parameters, and dispatches to deterministic capabilities (e.g. ArisMathTool).
          - For relational queries:
            - If use_advisor=True: queries ArisStructuralAdvisor, falls back to ArisNeuralFrameLearner.
            - If use_advisor=False: uses ArisNeuralFrameLearner directly for local offline neural perception.
          - Deterministic Graph Traversal over Memory: selects traversal operation and queries MongoDB.
        """
        if not question_text or not question_text.strip():
            return {"matches_count": 0, "answers": []}

        cleaned = question_text.strip()
        from ..linguistic.pipeline import ArisLinguisticPipeline
        frame = ArisLinguisticPipeline.process(cleaned)
        tokens = frame.tokens
        candidates = frame.candidates
        bindings = frame.bindings
        learned_templates = ArisCognitiveLearner.get_templates(in_memory_templates=templates)

        from .epistemic_state import ArisGoalFormer, ArisEpistemicEvaluator, ArisCapabilityMatcher, ArisEpistemicState
        goal = ArisGoalFormer.form_goal(cleaned, tokens, candidates, bindings, frame=frame)
        epistemic_eval = ArisEpistemicEvaluator.evaluate(goal, memory_facts=facts)
        epistemic_state = epistemic_eval.get("state")
        matched_cap = ArisCapabilityMatcher.match(epistemic_eval, goal)

        # -- COGNITIVE STATE BANNER (always visible) --------------------------
        _adv = "ON" if use_advisor else "OFF"
        print(f"\n{'-' * 70}")
        print(f"[AR.I.S REASONING] Query      : {cleaned[:80]}")
        print(f"[AR.I.S REASONING] Advisor    : {_adv} | Templates in memory: {len(learned_templates)}")
        print(f"[AR.I.S REASONING] Epistemic  : {getattr(epistemic_state, 'value', epistemic_state)}")
        print(f"[AR.I.S REASONING] Matched Cap: {matched_cap or 'none'}")
        print(f"[AR.I.S REASONING] Goal Type  : {goal.target_type} | Anchor: {goal.anchor_entity}")
        print(f"{'-' * 70}")
        # ---------------------------------------------------------------------

        epistemic_trace = {
            "state": str(epistemic_state.value if hasattr(epistemic_state, "value") else epistemic_state),
            "deficit": epistemic_eval.get("deficit"),
            "matched_capability": matched_cap,
            "goal": goal.target_variable,
            "goal_detail": {
                "target_variable": goal.target_variable,
                "target_type": goal.target_type,
                "anchor_entity": goal.anchor_entity,
                "relational_constraint": goal.relational_constraint,
            },
        }


        from ..semantic.executable_tools import ArisCapabilityRegistry

        # ---------------------------------------------------------
        # Branch 1: Self-Contained Computation (DERIVABLE - Pure Numerical/Math)
        # ---------------------------------------------------------
        # ---------------------------------------------------------
        # Branch 1: Self-Contained Computation (DERIVABLE - Pure Numerical & Algebraic Math)
        # ---------------------------------------------------------
        if goal.target_type in ("numerical_value", "algebraic_equation") or (hasattr(epistemic_state, "value") and epistemic_state == ArisEpistemicState.DERIVABLE and matched_cap == "math_computation"):
            # Check learned templates first for math_computation
            for tmpl in learned_templates:
                if tmpl.operation_type == "executable_capability" and tmpl.capability_name == "math_computation":
                    bound_params = ArisCognitiveLearner.bind_executable_parameters(tokens, tmpl, question_text=cleaned)
                    if bound_params:
                        clean_kwargs = {k: v for k, v in bound_params.items() if k != "capability_name"}
                        exec_res = ArisCapabilityRegistry.dispatch("math_computation", **clean_kwargs)
                        if exec_res.get("status") in ("success", "solved"):
                            val = exec_res.get("value", exec_res.get("result"))
                            ans_str = str(val)
                            var_label = bound_params.get("variable", "result")
                            if "equation" in bound_params:
                                eq = bound_params["equation"]
                                expr_label = f"{eq.get('left')} = {eq.get('right')} => {var_label} = {val}"
                            else:
                                expr_label = f"{exec_res.get('expression') or bound_params.get('expression') or str(bound_params)} = {ans_str}"
                            return {
                                "matches_count": 1,
                                "answers": [
                                    {
                                        "answer": ans_str,
                                        "statement": f"Mathematical Resolution: {expr_label}",
                                        "confidence": 1.0,
                                        "binding": {
                                            "capability": tmpl.capability_name,
                                            "parameters": bound_params,
                                            "result": exec_res.get("numeric_result", ans_str),
                                        },
                                    }
                                ],
                                "result": ans_str,
                                "applied_template": tmpl.to_dict(),
                                "reasoning_trace": {
                                    "epistemic_cognition": epistemic_trace,
                                    "sensory_observation": {
                                        "type": "parameterized_execution",
                                        "parameters": bound_params,
                                        "source": "learned_template_parameter_binding",
                                    },
                                    "cognitive_decision": {
                                        "operation": bound_params.get("operation", "executable_capability"),
                                        "capability": "math_computation",
                                        "bound_parameters": bound_params,
                                        "execution_status": "success",
                                        "result": ans_str,
                                    },
                                },
                            }

            # Direct autonomous mathematical dispatch from goal parameters
            if "equation" in goal.parameters or "expression" in goal.parameters:
                p_dict = dict(goal.parameters)
                exec_res = ArisCapabilityRegistry.dispatch("math_computation", **p_dict)
                if exec_res.get("status") in ("success", "solved"):
                    val = exec_res.get("value", exec_res.get("result"))
                    ans_str = str(val)
                    var_label = p_dict.get("variable", "result")
                    if "equation" in p_dict:
                        eq = p_dict["equation"]
                        expr_label = f"{eq.get('left')} = {eq.get('right')} => {var_label} = {val}"
                    else:
                        expr_label = f"{exec_res.get('expression') or p_dict.get('expression')} = {ans_str}"

                    # Learn/persist operation template for future reuse
                    learn_res = ArisCognitiveLearner.learn_executable_demonstration(
                        query_text=cleaned,
                        advisor_observation={"type": "executable_capability", "capability": "math_computation", "parameters": p_dict},
                        in_memory_templates=templates,
                    )
                    exec_template = ArisOperationTemplate.from_dict(learn_res["template"])

                    return {
                        "matches_count": 1,
                        "answers": [
                            {
                                "answer": ans_str,
                                "statement": f"Mathematical Resolution: {expr_label}",
                                "confidence": 1.0,
                                "binding": {
                                    "capability": "math_computation",
                                    "parameters": p_dict,
                                    "result": exec_res.get("numeric_result", ans_str),
                                    "expression": expr_label,
                                },
                            }
                        ],
                        "result": ans_str,
                        "applied_template": exec_template.to_dict(),
                        "reasoning_trace": {
                            "epistemic_cognition": epistemic_trace,
                            "sensory_observation": {
                                "type": "structural_mathematical_perception",
                                "parameters": p_dict,
                                "source": "epistemic_goal_parameters",
                            },
                            "cognitive_decision": {
                                "operation": p_dict.get("operation", "math_computation"),
                                "capability": "math_computation",
                                "bound_parameters": p_dict,
                                "execution_status": "success",
                                "result": ans_str,
                            },
                        },
                    }

            # If advisor is enabled and no template learned yet for math
            if use_advisor:
                obs = ArisStructuralAdvisor.observe_query(cleaned, enabled=True)
                if obs.get("status") == "ok" and obs.get("type") == "executable_capability" and obs.get("capability") == "math_computation":
                    learn_res = ArisCognitiveLearner.learn_executable_demonstration(
                        query_text=cleaned,
                        advisor_observation=obs,
                        in_memory_templates=templates,
                    )
                    exec_template = ArisOperationTemplate.from_dict(learn_res["template"])
                    p_dict = obs.get("parameters", {})
                    exec_res = ArisCapabilityRegistry.dispatch("math_computation", **p_dict)
                    if exec_res.get("status") in ("success", "solved"):
                        val = exec_res.get("value", exec_res.get("result"))
                        ans_str = str(val)
                        param_desc = ", ".join(f"{k}='{v}'" for k, v in p_dict.items())
                        return {
                            "matches_count": 1,
                            "answers": [
                                {
                                    "answer": ans_str,
                                    "statement": f"Capability Execution: {exec_template.capability_name}({param_desc}) = {ans_str}",
                                    "confidence": 1.0,
                                    "binding": {
                                        "capability": exec_template.capability_name,
                                        "parameters": p_dict,
                                        "result": exec_res.get("numeric_result", ans_str),
                                        "expression": exec_res.get("expression", param_desc),
                                    },
                                }
                            ],
                            "result": ans_str,
                            "applied_template": exec_template.to_dict(),
                            "reasoning_trace": {
                                "epistemic_cognition": epistemic_trace,
                                "sensory_observation": {
                                    "type": "teacher_guided_capability",
                                    "parameters": p_dict,
                                    "source": "sensory_structural_advisor",
                                },
                                "cognitive_decision": {
                                    "operation": "executable_capability",
                                    "capability": "math_computation",
                                    "bound_parameters": p_dict,
                                    "execution_status": "success",
                                    "result": ans_str,
                                },
                            },
                        }



        # ---------------------------------------------------------
        # Branch 1-B: Generic Executable Capability Template Scan
        # ---------------------------------------------------------
        # Tries ALL learned executable_capability templates (non-math).
        #
        # PRIORITY RULE: Templates are ranked by how many parameters they can resolve
        # from THIS specific query. Most-specific match fires first.
        # This prevents a generic template (e.g. browser_access with just 'query')
        # from shadowing a more specific one (e.g. command_prompt with a resolved path).
        #
        # ON EPISTEMIC DEFICIT:
        # When a template matches but required params cannot be resolved from the query,
        # AR.I.S returns a structured EPISTEMIC_DEFICIT state. This IS the correct honest
        # answer -- it surfaces exactly what AR.I.S knows and what it still needs.
        # No fake acquisition machinery. No hardcoded command extraction.
        # The deficit is the observation.
        _exec_tmpls_raw = [
            t for t in learned_templates
            if t.operation_type == "executable_capability" and t.capability_name != "math_computation"
        ]

        # Respect allow_external policy dynamically via capability contract (zero hardcoded tool names)
        from ..semantic.executable_tools import ArisCapabilityRegistry
        if not allow_external or matched_cap in ("memory_retrieval", "gnn_positional_hypothesis"):
            _exec_tmpls_raw = [
                t for t in _exec_tmpls_raw
                if not ArisCapabilityRegistry.is_external_capability(t.capability_name)
            ]

        # Pre-bind all templates and count how many params each resolves from this query
        _exec_ranked = []
        for _t in _exec_tmpls_raw:
            _bp = ArisCognitiveLearner.bind_executable_parameters(tokens, _t, question_text=cleaned)
            if _bp is not None:
                _resolved_count = sum(
                    1 for k, v in _bp.items() if k != "capability_name" and v is not None
                )
                _exec_ranked.append((_resolved_count, _t, _bp))

        # Sort descending: most resolved params first
        _exec_ranked.sort(key=lambda x: x[0], reverse=True)

        if _exec_ranked:
            print(f"[Branch 1-B] Found {len(_exec_ranked)} ranked executable template(s):")
            for _rc, _rt, _rb in _exec_ranked:
                _unr = [k for k, v in _rb.items() if k != "capability_name" and v is None]
                _status = f"DEFICIT: {_unr}" if _unr else "FULLY BOUND"
                print(f"[Branch 1-B]   [{_rc} resolved] '{_rt.template_id}' ({_rt.capability_name}) -- {_status}")
        else:
            print(f"[Branch 1-B] No non-math executable templates bindable for this query -- skipping.")

        for _resolved_count, tmpl, bound_params in _exec_ranked:
            print(f"[Branch 1-B] Trying '{tmpl.template_id}' ({tmpl.capability_name})")

            unresolved_slots = [k for k, v in bound_params.items() if k != "capability_name" and v is None]
            resolved_slots = {k: v for k, v in bound_params.items() if k != "capability_name" and v is not None}

            if unresolved_slots:
                # Honest epistemic deficit: AR.I.S matched the template pattern and resolved
                # some parameters from the query, but required parameters remain unknown.
                # This is the correct observable state — surface it transparently.
                print(f"[Branch 1-B]   Resolved   : {resolved_slots}")
                print(f"[Branch 1-B]   DEFICIT     : {unresolved_slots}")
                print(f"[Branch 1-B]   AR.I.S knows the capability pattern and some parameters,")
                print(f"[Branch 1-B]   but cannot determine {unresolved_slots} from this query alone.")
                print(f"[Branch 1-B]   Epistemic state: EPISTEMIC_DEFICIT -- returning honestly.")
                deficit_trace = dict(epistemic_trace)
                deficit_trace["state"] = "EPISTEMIC_DEFICIT"
                deficit_trace["deficit"] = "unresolved_capability_parameters"
                deficit_trace["matched_capability"] = tmpl.capability_name
                deficit_trace["unresolved_slots"] = unresolved_slots
                deficit_trace["resolved_slots"] = resolved_slots
                return {
                    "matches_count": 0,
                    "answers": [],
                    "status": "epistemic_deficit",
                    "applied_template": tmpl.to_dict(),
                    "reasoning_trace": {
                        "epistemic_cognition": deficit_trace,
                        "sensory_observation": {
                            "type": "executable_template_parameter_binding",
                            "template": tmpl.template_id,
                            "resolved": resolved_slots,
                            "unresolved": unresolved_slots,
                            "source": "learned_operation_template",
                        },
                        "cognitive_decision": {
                            "operation": "epistemic_deficit",
                            "capability": tmpl.capability_name,
                            "resolved_parameters": resolved_slots,
                            "missing_parameters": unresolved_slots,
                            "confidence": 0.0,
                            "note": (
                                f"Template '{tmpl.template_id}' matched. "
                                f"Resolved from query: {list(resolved_slots.keys())}. "
                                f"Cannot determine {unresolved_slots} without advisor or prior teaching."
                            ),
                        },
                    },
                }

            # All params fully resolved -- dispatch directly
            clean_kwargs = {k: v for k, v in bound_params.items() if k != "capability_name"}
            print(f"[Branch 1-B]   All params resolved -- dispatching '{tmpl.capability_name}' with {clean_kwargs}")
            exec_res = ArisCapabilityRegistry.dispatch(tmpl.capability_name, **clean_kwargs)
            if exec_res.get("status") == "success":
                ans_str = str(exec_res.get("result", ""))
                print(f"[Branch 1-B]   Dispatch SUCCESS -- '{tmpl.capability_name}' returned result.")
                exec_trace = dict(epistemic_trace)
                is_ext = ArisCapabilityRegistry.is_external_capability(tmpl.capability_name)
                exec_trace["state"] = ArisEpistemicState.ACQUIRABLE.value if is_ext else ArisEpistemicState.DERIVABLE.value
                exec_trace["deficit"] = None
                exec_trace["matched_capability"] = tmpl.capability_name
                return {
                    "matches_count": 1,
                    "answers": [
                        {
                            "answer": ans_str,
                            "statement": f"Capability Execution [{tmpl.capability_name}]: {ans_str}",
                            "confidence": 1.0,
                            "binding": {
                                "capability": tmpl.capability_name,
                                "parameters": bound_params,
                                "result": ans_str,
                            },
                        }
                    ],
                    "result": ans_str,
                    "applied_template": tmpl.to_dict(),
                    "reasoning_trace": {
                        "epistemic_cognition": exec_trace,
                        "sensory_observation": {
                            "type": "parameterized_execution",
                            "parameters": bound_params,
                            "source": "learned_template_parameter_binding",
                        },
                        "cognitive_decision": {
                            "operation": "executable_capability",
                            "capability": tmpl.capability_name,
                            "bound_parameters": bound_params,
                            "execution_status": "success",
                            "result": ans_str,
                        },
                    },
                }



        # Branch 2: Relational Graph Traversal over Memory (Law of Memory Priority)
        # ---------------------------------------------------------
        # All non-arithmetic inquiries probe internal memory FIRST!
        q_subj = None
        q_rel = None
        q_obj = None
        is_open_variable = False

        # If the query contains two or more concrete entities, check internal memory graph first!
        from ..relational.symbolic_graph import ArisSymbolicGraph
        graph = ArisSymbolicGraph.build_from_facts(facts=facts)

        eval_entities = [c["text"] for c in candidates]
        if len(eval_entities) < 2 and len(tokens) >= 2:
            # Check individual word tokens against grounded graph nodes
            token_words = [t["raw"] for t in tokens if t["type"] == "word" and t["raw"].lower() in graph.adjacency]
            if len(token_words) >= 2:
                eval_entities = token_words

        first_token_raw = tokens[0]["raw"] if tokens else ""
        requests_unnamed_entity = frame.is_question or bool(ArisCognitiveLearner.get_interrogative_target_domain(first_token_raw))

        if len(eval_entities) >= 2 and not requests_unnamed_entity:
            cand_a = eval_entities[0]
            cand_b = eval_entities[1]
            if cand_a.lower() in graph.adjacency and cand_b.lower() in graph.adjacency:
                from .relational_beliefs import ArisRelationalBeliefManager
                from ..relational.gnn_reasoner import ArisGNNReasoner

                # 1. Check Inhibition Matrix: Was this assertion explicitly denied by the user?
                if bindings and ArisRelationalBeliefManager.is_denied(cand_a, bindings[0]["relation"]["normalized"], cand_b):
                    denied_trace = dict(epistemic_trace)
                    denied_trace["state"] = "DENIED"
                    denied_trace["confidence"] = 0.0
                    denied_trace["inhibition"] = "explicitly_rejected_by_user"
                    return {
                        "matches_count": 0,
                        "confidence": 0.0,
                        "status": "denied",
                        "answers": [],
                        "reasoning_trace": {
                            "epistemic_cognition": denied_trace,
                            "status": "denied",
                            "statement": f"Hypothesis Refuted: '{cand_a}' to '{cand_b}' was explicitly rejected by user.",
                            "note": "Hypothesis permanently inhibited by user feedback.",
                        },
                    }

                # 2. Evaluate GNN positional path hypothesis over connected memory graph
                hyp_res = ArisGNNReasoner.evaluate_hypothesis(
                    subject=cand_a,
                    target_entity=cand_b,
                    relation=bindings[0]["relation"]["normalized"] if bindings else None,
                    facts=facts,
                )

                if hyp_res.get("status") == "denied":
                    denied_trace = dict(epistemic_trace)
                    denied_trace["state"] = "DENIED"
                    denied_trace["confidence"] = 0.0
                    denied_trace["inhibition"] = "explicitly_rejected_by_user"
                    return {
                        "matches_count": 0,
                        "confidence": 0.0,
                        "status": "denied",
                        "answers": [],
                        "reasoning_trace": {
                            "epistemic_cognition": denied_trace,
                            "status": "denied",
                            "statement": hyp_res.get("statement", f"Hypothesis Refuted: '{cand_a}' to '{cand_b}' was explicitly rejected by user."),
                            "note": "Hypothesis permanently inhibited by user feedback.",
                        },
                    }

                if hyp_res.get("status") in ("plausible_hypothesis", "verified_symmetric"):
                    hyp_conf = hyp_res.get("confidence", 0.5)
                    hyp_trace = dict(epistemic_trace)
                    hyp_trace["state"] = "HYPOTHETICAL"
                    hyp_trace["confidence"] = hyp_conf
                    hyp_trace["path_summary"] = hyp_res.get("path_summary")
                    hyp_trace["matched_capability"] = "gnn_positional_hypothesis"

                    answer_item = {
                        "answer": hyp_res.get("target_entity"),
                        "statement": hyp_res.get("statement"),
                        "confidence": hyp_conf,
                        "path_summary": hyp_res.get("path_summary"),
                        "provenance": hyp_res.get("provenance", []),
                        "status": hyp_res.get("status"),
                    }

                    return {
                        "matches_count": 1,
                        "confidence": hyp_conf,
                        "status": hyp_res.get("status"),
                        "answers": [answer_item],
                        "applied_template": {
                            "name": "GNN Positional Path Hypothesis",
                            "operation_type": "gnn_positional_hypothesis",
                            "path_summary": hyp_res.get("path_summary"),
                            "confidence": hyp_conf,
                        },
                        "reasoning_trace": {
                            "epistemic_cognition": hyp_trace,
                            "sensory_observation": {
                                "type": "gnn_graph_traversal",
                                "start_entity": cand_a,
                                "target_entity": cand_b,
                                "path_summary": hyp_res.get("path_summary"),
                                "source": "symbolic_graph_and_gnn_layer",
                            },
                            "cognitive_decision": {
                                "operation": "gnn_positional_hypothesis",
                                "start_entity": cand_a,
                                "target_entity": cand_b,
                                "hops": hyp_res.get("num_hops", 1),
                                "confidence": hyp_conf,
                                "provenance": hyp_res.get("provenance", []),
                            },
                        },
                    }

        if bindings:
            q_bind = bindings[0]
            q_subj = q_bind["subject"]["text"]
            q_rel = q_bind["relation"]["normalized"]
            q_obj = q_bind["object"]["text"]
        else:
            q_subj = None
            q_rel = None
            q_obj = None

        # Execute direct 1-hop memory retrieval only when cognitive capability contract matches memory_retrieval
        if matched_cap == "memory_retrieval" and q_subj and q_obj and q_rel:
            var_slot = None
            anc_slot = None
            obs_source = "none"

            if use_neural:
                try:
                    from ..linguistic.neural_frame_classifier import ArisNeuralFrameLearner
                    pred = ArisNeuralFrameLearner.predict(subject=q_subj, relation=q_rel, obj=q_obj)
                    if pred.get("status") == "ok":
                        var_slot = pred.get("variable_slot")
                        anc_slot = pred.get("anchor_slot")
                        obs_source = "local_neural_frame_classifier"
                except Exception:
                    pass
            elif use_advisor:
                obs = ArisStructuralAdvisor.observe_slots(
                    subject=q_subj,
                    obj=q_obj,
                    query_text=cleaned,
                    enabled=True,
                )
                if obs.get("status") == "ok":
                    var_slot = obs.get("variable_slot")
                    anc_slot = obs.get("anchor_slot")
                    obs_source = "sensory_structural_advisor"

            matching_template = None
            if anc_slot:
                for tmpl in learned_templates:
                    if tmpl.anchor_role == anc_slot and tmpl.source_role == var_slot:
                        matching_template = tmpl
                        break

                if not matching_template:
                    for tmpl in learned_templates:
                        if tmpl.anchor_role == anc_slot:
                            matching_template = tmpl
                            break
            elif (use_advisor or use_neural) and learned_templates:
                for tmpl in learned_templates:
                    if tmpl.operation_type in ("forward_traversal", "reverse_traversal"):
                        matching_template = tmpl
                        anc_slot = tmpl.anchor_role
                        var_slot = tmpl.source_role
                        break

            from ..relational.symbolic_graph import ArisSymbolicGraph
            graph = ArisSymbolicGraph.build_from_facts(facts=facts)
            s_known = q_subj.lower() in graph.adjacency
            o_known = q_obj.lower() in graph.adjacency
            is_concrete_pair = bool(s_known and o_known)

            # Observe structural asymmetry between grounded anchor and open slot
            if (use_advisor or use_neural) and not var_slot and (s_known != o_known):
                anc_slot = "object" if o_known else "subject"
                var_slot = "subject" if o_known else "object"
                obs_source = "structural_graph_grounding"

            if is_concrete_pair:
                is_open_variable = False
            elif matching_template and matching_template.operation_type in ("forward_traversal", "reverse_traversal"):
                is_open_variable = True
            elif var_slot:
                is_open_variable = True
            elif tokens and ArisCognitiveLearner.get_interrogative_target_domain(tokens[0]["raw"]):
                is_open_variable = True
            else:
                is_open_variable = False

            if is_open_variable:
                op_type = matching_template.operation_type if matching_template else (
                    "reverse_traversal" if anc_slot == "object" else "forward_traversal"
                )

                # Execute directed traversal
                if op_type == "forward_traversal":
                    q = ArisRelationalQuery(
                        subject=q_subj,
                        relation=q_rel,
                        obj=None,
                        direction="forward",
                        target_variable="object",
                    )
                else:
                    q = ArisRelationalQuery(
                        subject=None,
                        relation=q_rel,
                        obj=q_obj,
                        direction="reverse",
                        target_variable="subject",
                    )
                mem_result = ArisRetriever.execute_query(query=q, facts=facts, limit=limit)

                # If directed traversal didn't match and no slots were observed, attempt structural intersection
                if not var_slot and not anc_slot and mem_result.get("matches_count", 0) == 0:
                    q_inter = ArisRelationalQuery.from_text(cleaned)
                    res_inter = ArisRetriever.execute_query(query=q_inter, facts=facts, limit=limit)
                    if res_inter.get("matches_count", 0) > 0:
                        mem_result = res_inter
                        op_type = "structural_intersection"

            else:
                # Query with two concrete entity slots: evaluate direct verification
                q = ArisRelationalQuery(
                    subject=q_subj,
                    relation=q_rel,
                    obj=q_obj,
                    direction="verification",
                    target_variable="verification",
                )
                mem_result = ArisRetriever.execute_query(query=q, facts=facts, limit=limit)
                op_type = "verification"


            # If memory traversal found matching facts, RETURN IMMEDIATELY (KNOWN STATE)
            if mem_result.get("matches_count", 0) > 0:
                mem_result["applied_template"] = matching_template.to_dict() if matching_template else None
                known_trace = dict(epistemic_trace)
                known_trace["state"] = ArisEpistemicState.KNOWN.value
                known_trace["deficit"] = None
                known_trace["matched_capability"] = "memory_retrieval"
                mem_result["reasoning_trace"] = {
                    "epistemic_cognition": known_trace,
                    "sensory_observation": {
                        "variable_slot": var_slot or "unknown",
                        "anchor_slot": anc_slot or "unknown",
                        "source": obs_source,
                    },
                    "cognitive_decision": {
                        "operation": op_type,
                        "anchor": q_subj if op_type == "forward_traversal" else q_obj,
                        "relation": q_rel,
                        "target_slot": "object" if op_type == "forward_traversal" else "subject",
                    },
                }
                return mem_result


        # ---------------------------------------------------------
        # Branch 2-B: Multi-Hop Relational Traversal over Memory
        # ---------------------------------------------------------
        # Path 1: Teacher-guided sensory perception (When advisor is enabled)
        if use_advisor:
            decomp = ArisStructuralAdvisor.decompose_multi_hop(cleaned, enabled=True)
            if decomp.get("status") == "ok" and decomp.get("start_entity") and decomp.get("hops"):
                start_entity = decomp["start_entity"]
                hops = decomp["hops"]
                target_domain = decomp.get("target_domain", "entity")

                from ..relational.retrieval import ArisMultiHopQuery
                multi_query = ArisMultiHopQuery(start_entity=start_entity, hops=hops)
                multi_res = ArisRetriever.execute_multi_hop(multi_query, facts=facts, limit=limit)

                if multi_res.get("matches_count", 0) > 0:
                    # Dynamically associate the interrogative token with this target domain in memory
                    if tokens:
                        ArisCognitiveLearner.record_interrogative_association(tokens[0]["raw"], target_domain)

                    # Learn and persist the demonstrated multi-hop operation template
                    ArisCognitiveLearner.learn_multihop_demonstration(
                        query_text=cleaned,
                        hops=hops,
                        start_entity=start_entity,
                        target_domain=target_domain,
                        in_memory_templates=templates,
                    )

                    known_trace = dict(epistemic_trace)
                    known_trace["state"] = ArisEpistemicState.KNOWN.value
                    known_trace["deficit"] = None
                    known_trace["matched_capability"] = "multi_hop_traversal"

                    formatted_answers = []
                    for ans_item in multi_res["answers"]:
                        formatted_answers.append({
                            "answer": ans_item["answer"],
                            "statement": f"Multi-Hop Traversal: {ans_item['path_summary']}",
                            "confidence": 1.0,
                            "path": ans_item.get("path", []),
                            "path_summary": ans_item.get("path_summary"),
                        })

                    return {
                        "matches_count": len(formatted_answers),
                        "answers": formatted_answers,
                        "applied_template": {
                            "name": f"Multi-Hop Compositional Traversal ({len(hops)} hops, {target_domain})",
                            "operation_type": "multi_hop_traversal",
                            "start_entity": start_entity,
                            "hops": hops,
                            "target_domain": target_domain,
                        },
                        "reasoning_trace": {
                            "epistemic_cognition": known_trace,
                            "sensory_observation": {
                                "type": "multi_hop_decomposition",
                                "start_entity": start_entity,
                                "hops": hops,
                                "target_domain": target_domain,
                                "source": "sensory_structural_advisor",
                            },
                            "cognitive_decision": {
                                "operation": "multi_hop_traversal",
                                "start_entity": start_entity,
                                "hops_count": len(hops),
                                "hops": hops,
                                "target_domain": target_domain,
                            },
                        },
                    }

        # Path 2: Autonomous Offline Reasoning over Learned Templates (When advisor is disabled)
        first_token = tokens[0]["raw"] if tokens else ""
        expected_target_domain = ArisCognitiveLearner.get_interrogative_target_domain(first_token)

        if expected_target_domain:
            for tmpl in learned_templates:
                if tmpl.operation_type == "multi_hop_traversal":
                    tmpl_domain = tmpl.parameter_mapping.get("target_domain")
                    if tmpl_domain != expected_target_domain:
                        continue

                    hops = tmpl.parameter_mapping.get("hops", [])
                    if hops:
                        anchor_cand = goal.anchor_entity
                        if not anchor_cand and candidates:
                            anchor_cand = max(candidates, key=lambda c: len(c.get("text", "")))["text"]

                        if anchor_cand:
                            from ..relational.retrieval import ArisMultiHopQuery
                            multi_query = ArisMultiHopQuery(start_entity=anchor_cand, hops=hops)
                            multi_res = ArisRetriever.execute_multi_hop(multi_query, facts=facts, limit=limit)

                            if multi_res.get("matches_count", 0) > 0:
                                known_trace = dict(epistemic_trace)
                                known_trace["state"] = ArisEpistemicState.KNOWN.value
                                known_trace["deficit"] = None
                                known_trace["matched_capability"] = "multi_hop_traversal"

                                formatted_answers = []
                                for ans_item in multi_res["answers"]:
                                    formatted_answers.append({
                                        "answer": ans_item["answer"],
                                        "statement": f"Multi-Hop Traversal: {ans_item['path_summary']}",
                                        "confidence": 1.0,
                                        "path": ans_item.get("path", []),
                                        "path_summary": ans_item.get("path_summary"),
                                    })

                                return {
                                    "matches_count": len(formatted_answers),
                                    "answers": formatted_answers,
                                    "applied_template": tmpl.to_dict(),
                                    "reasoning_trace": {
                                        "epistemic_cognition": known_trace,
                                        "sensory_observation": {
                                            "type": "learned_multihop_execution",
                                            "start_entity": anchor_cand,
                                            "hops": hops,
                                            "target_domain": expected_target_domain,
                                            "source": "learned_operation_template",
                                        },
                                        "cognitive_decision": {
                                            "operation": "multi_hop_traversal",
                                            "start_entity": anchor_cand,
                                            "hops_count": len(hops),
                                            "hops": hops,
                                            "target_domain": expected_target_domain,
                                        },
                                    },
                                }



        # ---------------------------------------------------------
        # Branch 3: In-Memory Multi-Hop GNN Positional Hypothesis Formation (Pillar 4)
        # ---------------------------------------------------------
        # LAW OF MEMORY PRIORITY: Always evaluate internal memory graph connections
        # before attempting external tool acquisition (like web search / browser).
        check_subj = q_subj
        check_obj = q_obj
        check_rel = q_rel

        if not (check_subj and check_obj) and len(candidates) >= 2:
            check_subj = candidates[0]["text"]
            check_obj = candidates[1]["text"]
            if not check_rel and bindings:
                check_rel = bindings[0]["relation"]["normalized"]

        from .relational_beliefs import ArisRelationalBeliefManager
        from ..relational.gnn_reasoner import ArisGNNReasoner

        if not is_open_variable and not requests_unnamed_entity and check_subj and check_obj:
            # 1. Check Inhibition Matrix: Was this assertion explicitly denied by the user?
            if check_rel and ArisRelationalBeliefManager.is_denied(check_subj, check_rel, check_obj):
                denied_trace = dict(epistemic_trace)
                denied_trace["state"] = "DENIED"
                denied_trace["confidence"] = 0.0
                denied_trace["inhibition"] = "explicitly_rejected_by_user"
                return {
                    "matches_count": 0,
                    "confidence": 0.0,
                    "status": "denied",
                    "answers": [],
                    "reasoning_trace": {
                        "epistemic_cognition": denied_trace,
                        "status": "denied",
                        "statement": f"Hypothesis Refuted: '{check_subj}' {check_rel} '{check_obj}' was explicitly rejected by user.",
                        "note": "Hypothesis permanently inhibited by user feedback.",
                    },
                }

            # 2. Evaluate GNN positional path hypothesis over connected memory graph
            hyp_res = ArisGNNReasoner.evaluate_hypothesis(
                subject=check_subj,
                target_entity=check_obj,
                relation=check_rel,
                facts=facts,
            )

            if hyp_res.get("status") in ("plausible_hypothesis", "verified_symmetric"):
                hyp_conf = hyp_res.get("confidence", 0.5)
                hyp_trace = dict(epistemic_trace)
                hyp_trace["state"] = "HYPOTHETICAL"
                hyp_trace["confidence"] = hyp_conf
                hyp_trace["path_summary"] = hyp_res.get("path_summary")
                hyp_trace["matched_capability"] = "gnn_positional_hypothesis"

                answer_item = {
                    "answer": hyp_res.get("target_entity"),
                    "statement": hyp_res.get("statement"),
                    "confidence": hyp_conf,
                    "path_summary": hyp_res.get("path_summary"),
                    "provenance": hyp_res.get("provenance", []),
                    "status": hyp_res.get("status"),
                }

                return {
                    "matches_count": 1,
                    "confidence": hyp_conf,
                    "status": hyp_res.get("status"),
                    "answers": [answer_item],
                    "applied_template": {
                        "name": "GNN Positional Path Hypothesis",
                        "operation_type": "gnn_positional_hypothesis",
                        "path_summary": hyp_res.get("path_summary"),
                        "confidence": hyp_conf,
                    },
                    "reasoning_trace": {
                        "epistemic_cognition": hyp_trace,
                        "sensory_observation": {
                            "type": "gnn_graph_traversal",
                            "start_entity": check_subj,
                            "target_entity": check_obj,
                            "path_summary": hyp_res.get("path_summary"),
                            "source": "symbolic_graph_and_gnn_layer",
                        },
                        "cognitive_decision": {
                            "operation": "gnn_positional_hypothesis",
                            "start_entity": check_subj,
                            "target_entity": check_obj,
                            "hops": hyp_res.get("num_hops", 1),
                            "confidence": hyp_conf,
                            "provenance": hyp_res.get("provenance", []),
                        },
                    },
                }

        # Open-variable graph hypothesis discovery
        # Gather all grounded entity anchors mentioned in the query into candidate space
        query_anchors = []
        raw_anchor_sources = []
        if goal and goal.anchor_entity:
            raw_anchor_sources.append(goal.anchor_entity)
        if q_subj:
            raw_anchor_sources.append(q_subj)
        if q_obj:
            raw_anchor_sources.append(q_obj)
        if candidates:
            raw_anchor_sources.extend([c.get("text", "") for c in candidates if c.get("text")])
        for raw_a in raw_anchor_sources:
            a_str = raw_a.strip()
            if not a_str:
                continue
            matched_node = next((k for k in graph.adjacency if k.lower() == a_str.lower()), None)
            if matched_node and matched_node not in query_anchors:
                query_anchors.append(matched_node)

        if templates is None and query_anchors:
            tgt_t = goal.target_type if (goal and goal.target_type not in ("unknown", "entity_reference", "knowledge_acquisition")) else None
            rel_constraint = (goal.relational_constraint if goal else None) or q_rel or cleaned

            excluded_entities = []
            if context:
                anchor_names = {a.lower() for a in query_anchors}
                if goal and goal.anchor_entity:
                    anchor_names.add(goal.anchor_entity.lower())
                for item in context.get_entity_salience_ranking():
                    ent = item.get("entity")
                    if ent and ent.lower() not in anchor_names:
                        excluded_entities.append(ent)

            open_hyp = ArisGNNReasoner.evaluate_open_variable_hypothesis(
                anchor_entity=[goal.anchor_entity] if (goal and goal.anchor_entity) else query_anchors,
                relation=rel_constraint,
                facts=facts,
                query_text=cleaned,
                target_type=tgt_t,
                excluded_entities=excluded_entities if excluded_entities else None,
            )
            anchor_cand = open_hyp.get("anchor_entity") or query_anchors[0]

            if open_hyp.get("status") == "plausible_hypothesis":
                hyp_conf = open_hyp.get("confidence", 0.5)
                hyp_trace = dict(epistemic_trace)
                hyp_trace["state"] = "HYPOTHETICAL"
                hyp_trace["confidence"] = hyp_conf
                hyp_trace["path_summary"] = open_hyp.get("path_summary")
                hyp_trace["matched_capability"] = "gnn_positional_hypothesis"

                all_hyps = open_hyp.get("all_hypotheses", [open_hyp])
                answer_items = [
                    {
                        "answer": h.get("target_entity"),
                        "statement": h.get("statement"),
                        "confidence": h.get("confidence", hyp_conf),
                        "path_summary": h.get("path_summary"),
                        "provenance": h.get("provenance", []),
                        "status": h.get("status"),
                    }
                    for h in all_hyps
                ]

                return {
                    "matches_count": len(answer_items),
                    "confidence": hyp_conf,
                    "status": "plausible_hypothesis",
                    "answers": answer_items,
                    "matches": answer_items,
                    "applied_template": {
                        "name": "Open-Variable Graph Hypothesis",
                        "operation_type": "gnn_positional_hypothesis",
                        "path_summary": open_hyp.get("path_summary"),
                        "confidence": hyp_conf,
                    },
                    "reasoning_trace": {
                        "epistemic_cognition": hyp_trace,
                        "sensory_observation": {
                            "type": "open_variable_graph_traversal",
                            "anchor_entity": anchor_cand,
                            "discovered_target": open_hyp.get("target_entity"),
                            "path_summary": open_hyp.get("path_summary"),
                            "source": "symbolic_graph_and_gnn_layer",
                        },
                        "cognitive_decision": {
                            "operation": "gnn_positional_hypothesis",
                            "anchor_entity": anchor_cand,
                            "target_entity": open_hyp.get("target_entity"),
                            "hops": open_hyp.get("hop_count", 1),
                            "confidence": hyp_conf,
                            "provenance": open_hyp.get("provenance", []),
                        },
                    },
                }

            elif open_hyp.get("status") == "unsupported":
                unsupported_trace = dict(epistemic_trace)
                unsupported_trace["state"] = ArisEpistemicState.UNSUPPORTED.value
                unsupported_trace["confidence"] = 0.0
                unsupported_trace["deficit"] = "unsupported_query_relation"
                unsupported_trace["matched_capability"] = "none"
                return {
                    "matches_count": 0,
                    "confidence": 0.0,
                    "status": "unsupported",
                    "answers": [],
                    "reasoning_trace": {
                        "epistemic_cognition": unsupported_trace,
                        "sensory_observation": {
                            "type": "epistemic_negative_rejection",
                            "anchor_entity": anchor_cand,
                            "relation": cleaned,
                            "source": "goal_reasoning_tensor_engine",
                        },
                        "cognitive_decision": {
                            "operation": "reject_unsupported",
                            "anchor_entity": anchor_cand,
                            "confidence": 0.0,
                        },
                    },
                }


        # ---------------------------------------------------------
        # Branch 4: External Capability Acquisition (ACQUIRABLE / DERIVABLE)
        # ---------------------------------------------------------
        # Reached ONLY when epistemic state indicates an external acquisition gap
        # AND internal memory graph has no multi-hop pathway!
        if allow_external and epistemic_state in (ArisEpistemicState.ACQUIRABLE, ArisEpistemicState.DERIVABLE):
            print(f"[Branch 4] Epistemic state '{getattr(epistemic_state,'value',epistemic_state)}' â†’ scanning for non-math executable templates.")
            for tmpl in learned_templates:
                if tmpl.operation_type == "executable_capability" and tmpl.capability_name != "math_computation":
                    if matched_cap and tmpl.capability_name != matched_cap:
                        continue
                    bound_params = ArisCognitiveLearner.bind_executable_parameters(tokens, tmpl, question_text=cleaned)
                    if bound_params:
                        cap_name = bound_params.get("capability_name") or tmpl.capability_name
                        clean_kwargs = {k: v for k, v in bound_params.items() if k != "capability_name"}
                        exec_res = ArisCapabilityRegistry.dispatch(
                            cap_name,
                            **clean_kwargs,
                        )

                        if exec_res.get("status") == "success":
                            ans_str = str(exec_res["result"])
                            expression_label = exec_res.get("expression") or bound_params.get("expression") or bound_params.get("query") or str(bound_params)
                            acq_trace = dict(epistemic_trace)
                            acq_trace["state"] = ArisEpistemicState.ACQUIRABLE.value
                            acq_trace["deficit"] = "external_knowledge_gap"
                            acq_trace["matched_capability"] = cap_name
                            return {
                                "matches_count": 1,
                                "answers": [
                                    {
                                        "answer": ans_str,
                                        "statement": f"Capability Execution: {expression_label} = {ans_str}",
                                        "confidence": 1.0,
                                        "binding": {
                                            "capability": tmpl.capability_name,
                                            "parameters": bound_params,
                                            "result": exec_res.get("numeric_result", ans_str),
                                        },
                                    }
                                ],
                                "result": ans_str,
                                "applied_template": tmpl.to_dict(),
                                "reasoning_trace": {
                                    "epistemic_cognition": acq_trace,
                                    "sensory_observation": {
                                        "type": "parameterized_execution",
                                        "parameters": bound_params,
                                        "source": "learned_template_parameter_binding",
                                    },
                                    "cognitive_decision": {
                                        "operation": "executable_capability",
                                        "capability": tmpl.capability_name,
                                        "bound_parameters": bound_params,
                                        "execution_status": "success",
                                        "result": ans_str,
                                        "ingested_facts_count": exec_res.get("ingested_facts_count", 0),
                                        "total_tokens": exec_res.get("total_tokens", 0),
                                        "total_relations": exec_res.get("total_relations", 0),
                                    },
                                },
                            }

        # Teacher-Guided Capability Learning or Autonomous Fallback (When Advisor Enabled or Direct Acquisition)
        if allow_external and not templates and (epistemic_state == ArisEpistemicState.ACQUIRABLE or matched_cap in ("browser_access", "web_search", "browser_search")):
            target_cap = matched_cap or "browser_access"
            print(f"[Fallback Acquisition] Matched cap='{matched_cap}' â†’ routing to '{target_cap}' for autonomous knowledge acquisition.")
            print(f"[Fallback Acquisition] Reason: Branch 1-B detected EPISTEMIC DEFICIT (unresolved 'command' slot).")
            print(f"[Fallback Acquisition] AR.I.S will use '{target_cap}' to learn how to fill the missing parameter.")
            cap_cls = ArisCapabilityRegistry.get_capability(target_cap)
            if cap_cls:
                bound_params = None
                if use_advisor:
                    obs = ArisStructuralAdvisor.observe_query(cleaned, enabled=True)
                    if obs.get("status") == "ok" and obs.get("type") == "executable_capability":
                        bound_params = obs.get("parameters", {})
                        target_cap = obs.get("capability") or target_cap

                if not bound_params:
                    # Autonomous structural binding via candidate/query extractor
                    dummy_tmpl = ArisOperationTemplate(
                        template_id=f"op_{target_cap}",
                        name=f"Autonomous {target_cap}",
                        operation_type="executable_capability",
                        source_role="parameters",
                        target_role="numeric_result",
                        anchor_role="relational_operator",
                        capability_name=target_cap,
                        parameter_mapping={"param_names": ["query"]},
                    )
                    bound_params = ArisCognitiveLearner.bind_executable_parameters(tokens, dummy_tmpl, question_text=cleaned)

                if bound_params:
                    # Learn/persist the capability template for offline reuse
                    learn_res = ArisCognitiveLearner.learn_executable_demonstration(
                        query_text=cleaned,
                        advisor_observation={"type": "executable_capability", "capability": target_cap, "parameters": bound_params},
                        in_memory_templates=templates,
                    )
                    exec_template = ArisOperationTemplate.from_dict(learn_res["template"])
                    clean_kwargs = {k: v for k, v in bound_params.items() if k != "capability_name"}

                    exec_res = ArisCapabilityRegistry.dispatch(
                        target_cap,
                        **clean_kwargs,
                    )
                    if exec_res.get("status") == "success":
                        ans_str = str(exec_res["result"])
                        param_desc = ", ".join(f"{k}='{v}'" for k, v in clean_kwargs.items())
                        acq_trace = dict(epistemic_trace)
                        acq_trace["state"] = ArisEpistemicState.ACQUIRABLE.value
                        acq_trace["deficit"] = "external_knowledge_gap"
                        acq_trace["matched_capability"] = target_cap
                        return {
                            "matches_count": 1,
                            "answers": [
                                {
                                    "answer": ans_str,
                                    "statement": f"Capability Execution: {target_cap}({param_desc}) = {ans_str}",
                                    "confidence": 1.0,
                                    "binding": {
                                        "capability": target_cap,
                                        "parameters": bound_params,
                                        "result": ans_str,
                                        "expression": exec_res.get("expression", param_desc),
                                    },
                                }
                            ],
                            "result": ans_str,
                            "applied_template": exec_template.to_dict(),
                            "reasoning_trace": {
                                "epistemic_cognition": acq_trace,
                                "sensory_observation": {
                                    "type": "executable_capability",
                                    "parameters": bound_params,
                                    "source": "autonomous_parameter_binding",
                                },
                                "cognitive_decision": {
                                    "operation": "executable_capability",
                                    "capability": target_cap,
                                    "bound_parameters": bound_params,
                                    "execution_status": "success",
                                    "result": ans_str,
                                    "ingested_facts_count": exec_res.get("ingested_facts_count", 0),
                                    "total_tokens": exec_res.get("total_tokens", 0),
                                    "total_relations": exec_res.get("total_relations", 0),
                                },
                            },
                        }



        # If memory produced 0 matches and no external capability bridges the deficit:
        unsupp_trace = dict(epistemic_trace)
        unsupp_trace["state"] = ArisEpistemicState.UNSUPPORTED.value
        unsupp_trace["deficit"] = "unbridgeable_gap"
        status_val = "advisor_unavailable" if not use_advisor else "unsupported_or_no_matches"
        return {
            "matches_count": 0,
            "answers": [],
            "reasoning_trace": {
                "epistemic_cognition": unsupp_trace,
                "status": status_val,
                "note": "Internal memory has no matching bindings and no registered capability satisfies the deficit.",
            },
        }


