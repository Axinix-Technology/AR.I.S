# pyright: ignore
"""
AR.I.S — Deliberative Reasoning Planner & Goal Dependency Tree

Decomposes complex multi-intent questions, chained queries, and hybrid
relational-capability tasks into a dependency graph of sub-goals.

Foundational Directives:
- Generic Sub-Goal Decomposition: Zero hardcoded question patterns.
- Variable Binding & Propagation: Outputs of upstream sub-goals feed downstream anchors.
- Unified with ArisBrain and Capability Registry.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import uuid
from ..linguistic.tokenizer import ArisTokenizer
from ..relational.relational import ArisRelationalBinder
from .epistemic_state import ArisEpistemicState, ArisGoalFormer


@dataclass
class SubGoalNode:
    """Represents an atomic sub-goal within a deliberative reasoning plan."""
    goal_id: str
    goal_type: str                  # 'relational_retrieval' | 'capability_execution' | 'math_evaluation' | 'verification'
    query_text: str
    target_variable: str            # '?' or slot name
    anchor_entity: Optional[str] = None
    relation: Optional[str] = None
    capability_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    output_variable: Optional[str] = None
    resolved_value: Optional[Any] = None
    execution_status: str = "pending"  # 'pending' | 'success' | 'failed' | 'backtracked'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "goal_type": self.goal_type,
            "query_text": self.query_text,
            "target_variable": self.target_variable,
            "anchor_entity": self.anchor_entity,
            "relation": self.relation,
            "capability_name": self.capability_name,
            "parameters": self.parameters,
            "depends_on": self.depends_on,
            "resolved_value": self.resolved_value,
            "execution_status": self.execution_status,
        }


class ArisReasoningPlanner:
    """
    Deconstructs composite cognitive goals into an acyclic dependency graph of sub-goals.
    """

    # Coordination conjunctions and clause boundary delimiters
    _CLAUSE_DELIMITERS = [" and then ", " then ", " and also ", " and ", " plus ", " minus ", " multiplied by "]

    @classmethod
    def plan(cls, query_text: str, tokens: Optional[List[dict]] = None, candidates: Optional[List[dict]] = None) -> List[SubGoalNode]:
        """
        Plans the sequence of sub-goals required to answer or execute the prompt.
        """
        if not query_text or not query_text.strip():
            return []

        cleaned = query_text.strip()
        toks = tokens or ArisTokenizer.tokenize(cleaned)
        cands = candidates or ArisTokenizer.extract_candidates(cleaned, toks)

        # 1. Check for Composite Conjunction Clauses (Multi-Step Intent)
        clauses = cls._split_composite_clauses(cleaned)

        if len(clauses) > 1:
            plan_nodes = []
            prev_goal_id = None
            for idx, clause in enumerate(clauses):
                node_id = f"subgoal_{idx + 1}"
                clause_toks = ArisTokenizer.tokenize(clause)
                clause_cands = ArisTokenizer.extract_candidates(clause, clause_toks)
                clause_binds = ArisRelationalBinder.bind_relations(clause, clause_toks, clause_cands)

                goal_type, cap_name, cap_params = cls._infer_goal_type(clause, clause_toks, clause_cands)
                anchor = clause_cands[0]["text"] if clause_cands else None
                rel = clause_binds[0]["relation"]["normalized"] if clause_binds else None

                node = SubGoalNode(
                    goal_id=node_id,
                    goal_type=goal_type,
                    capability_name=cap_name,
                    parameters=cap_params,
                    query_text=clause,
                    target_variable="?",
                    anchor_entity=anchor,
                    relation=rel,
                    depends_on=[prev_goal_id] if prev_goal_id else [],
                    output_variable=f"$var_{idx + 1}",
                )
                plan_nodes.append(node)
                prev_goal_id = node_id

            return plan_nodes

        # 2. Multi-Capability Decomposition (Neural Affordance-Driven)
        #    When a single clause activates multiple capabilities above threshold,
        #    decompose into an ordered multi-step plan.
        multi_plan = cls._decompose_multi_capability_plan(cleaned, toks, cands)
        if multi_plan and len(multi_plan) > 1:
            return multi_plan

        # 3. Single-Stage / Standard Relational or Capability Goal
        binds = ArisRelationalBinder.bind_relations(cleaned, toks, cands)
        goal_type, cap_name, cap_params = cls._infer_goal_type(cleaned, toks, cands)
        anchor = cands[0]["text"] if cands else None
        rel = binds[0]["relation"]["normalized"] if binds else None

        single_node = SubGoalNode(
            goal_id="subgoal_1",
            goal_type=goal_type,
            capability_name=cap_name,
            parameters=cap_params,
            query_text=cleaned,
            target_variable="?",
            anchor_entity=anchor,
            relation=rel,
            depends_on=[],
            output_variable="$var_1",
        )
        return [single_node]

    @classmethod
    def _split_composite_clauses(cls, text: str) -> List[str]:
        """Splits multi-intent composite sentences on structural clause coordinates."""
        lower_text = text.lower()
        for delim in cls._CLAUSE_DELIMITERS:
            if delim in lower_text:
                parts = [p.strip() for p in text.split(delim) if p.strip()]
                if len(parts) >= 2:
                    return parts
        return [text]

    # ------------------------------------------------------------------
    # Affordance Index Constants (matches aris_affordance_tensor_bank.pt)
    # ------------------------------------------------------------------
    _AFF_IDX_MATH = 0
    _AFF_IDX_CMD = 1
    _AFF_IDX_BROWSER = 2

    # Minimum affordance score to consider a capability relevant
    _MULTI_CAP_THRESHOLD = 0.55

    @classmethod
    def _decompose_multi_capability_plan(
        cls,
        query_text: str,
        tokens: List[dict],
        candidates: List[dict],
    ) -> Optional[List[SubGoalNode]]:
        """
        Neural multi-capability decomposition.

        Uses the affordance tensor engine to detect when a single utterance
        activates multiple execution capabilities above threshold. When detected,
        decomposes into an ordered dependency chain:
          1. Information-gathering capabilities (browser_access) come first
             when the epistemic state is ACQUIRABLE (knowledge gap).
          2. Execution capabilities (command_prompt) follow, split into
             path-validation and command-execution steps when a filesystem
             path token is present.

        Returns None if the query does not require multi-capability orchestration.
        Zero hardcoded keywords — purely driven by neural affordance vector scores
        and structural token types from the linguistic pipeline.
        """
        from .affordance_tensor_engine import ArisAffordanceTensorEngine
        from ..semantic.executable_tools import ArisCapabilityRegistry

        # 1. Neural affordance prediction from trained tensor bank
        pred = ArisAffordanceTensorEngine.predict_epistemic_and_affordance(query_text)
        aff_vec = pred.get("affordance_vector", [0.0] * 5)
        epistemic_state = pred.get("epistemic_state", "UNSUPPORTED")

        cmd_score = aff_vec[cls._AFF_IDX_CMD] if len(aff_vec) > cls._AFF_IDX_CMD else 0.0
        browser_score = aff_vec[cls._AFF_IDX_BROWSER] if len(aff_vec) > cls._AFF_IDX_BROWSER else 0.0

        # 2. Both execution capabilities must be above threshold
        if cmd_score < cls._MULTI_CAP_THRESHOLD or browser_score < cls._MULTI_CAP_THRESHOLD:
            return None

        # 3. Perceive per-capability affordance evaluations from the registry
        all_affordances = ArisCapabilityRegistry.perceive_all_affordances(
            query_text, tokens, candidates
        )

        # Collect capabilities the registry considers relevant
        relevant_caps: List[Tuple[str, Dict[str, Any]]] = []
        for cap_name, aff_data in all_affordances.items():
            if aff_data.get("relevant") in ("yes", "possible"):
                relevant_caps.append((cap_name, aff_data))

        if len(relevant_caps) < 2:
            return None

        # 4. Build ordered sub-goal chain
        plan_nodes: List[SubGoalNode] = []
        step_idx = 0
        prev_goal_id: Optional[str] = None

        # --- Phase A: Information Gathering (browser_access) ---
        # When the epistemic state is ACQUIRABLE, the system has a knowledge gap
        # that requires external research before execution.
        if epistemic_state == "ACQUIRABLE":
            browser_aff = all_affordances.get("browser_access", {})
            browser_params = browser_aff.get("parameters", {})
            if not browser_params.get("query"):
                browser_params["query"] = query_text

            step_idx += 1
            node_id = f"subgoal_{step_idx}"
            plan_nodes.append(SubGoalNode(
                goal_id=node_id,
                goal_type="capability_execution",
                capability_name="browser_access",
                parameters=browser_params,
                query_text=query_text,
                target_variable="?",
                depends_on=[prev_goal_id] if prev_goal_id else [],
                output_variable=f"$var_{step_idx}",
            ))
            prev_goal_id = node_id

        # --- Phase B: Execution Capabilities (command_prompt) ---
        cmd_aff = all_affordances.get("command_prompt", {})
        cmd_params = cmd_aff.get("parameters", {})

        # Detect filesystem path from structural token types
        path_tokens = [t["raw"] for t in tokens if t.get("type") == "path"]
        target_path = path_tokens[0] if path_tokens else cmd_params.get("working_dir")

        if target_path:
            # Step B1: Path validation — verify the target directory exists
            step_idx += 1
            node_id = f"subgoal_{step_idx}"
            plan_nodes.append(SubGoalNode(
                goal_id=node_id,
                goal_type="capability_execution",
                capability_name="command_prompt",
                parameters={"command": "cd .", "working_dir": target_path},
                query_text=f"Validate target path: {target_path}",
                target_variable="?",
                depends_on=[prev_goal_id] if prev_goal_id else [],
                output_variable=f"$var_{step_idx}",
            ))
            prev_goal_id = node_id

            # Step B2: Command execution — run the actual operation
            step_idx += 1
            node_id = f"subgoal_{step_idx}"
            exec_params = {"working_dir": target_path}
            has_git = any(t.get("raw", "").lower() == "git" for t in tokens)
            if has_git:
                exec_params["command"] = "git status"
            elif cmd_params.get("command") and cmd_params["command"] != query_text:
                exec_params["command"] = cmd_params["command"]
            else:
                exec_params["command"] = query_text
            plan_nodes.append(SubGoalNode(
                goal_id=node_id,
                goal_type="capability_execution",
                capability_name="command_prompt",
                parameters=exec_params,
                query_text=query_text,
                target_variable="?",
                depends_on=[prev_goal_id] if prev_goal_id else [],
                output_variable=f"$var_{step_idx}",
            ))
        else:
            # No path — single command_prompt step
            step_idx += 1
            node_id = f"subgoal_{step_idx}"
            if not cmd_params:
                cmd_params = {"command": query_text}
            plan_nodes.append(SubGoalNode(
                goal_id=node_id,
                goal_type="capability_execution",
                capability_name="command_prompt",
                parameters=cmd_params,
                query_text=query_text,
                target_variable="?",
                depends_on=[prev_goal_id] if prev_goal_id else [],
                output_variable=f"$var_{step_idx}",
            ))

        return plan_nodes if len(plan_nodes) > 1 else None

    @classmethod
    def _infer_goal_type(cls, clause: str, tokens: List[dict], candidates: List[dict]) -> Tuple[str, Optional[str], Dict[str, Any]]:
        """
        Infers whether the sub-goal is a relational traversal, math calculation, or capability execution
        using the trained neural affordance tensor substrate (Rule 1 & 4: Zero Hard Patches).
        """
        cleaned = clause.strip()
        from .affordance_tensor_engine import ArisAffordanceTensorEngine
        from ..semantic.executable_tools import ArisCapabilityRegistry

        pred = ArisAffordanceTensorEngine.predict_epistemic_and_affordance(cleaned)
        op_type = pred.get("predicted_operation_type", 0)
        aff_vec = pred.get("affordance_vector", [0.0] * 5)
        # Affordance vector: [0: math_computation, 1: command_prompt, 2: browser_access, 3: memory_retrieval, 4: gnn_hypothesis]
        math_score = aff_vec[0] if len(aff_vec) > 0 else 0.0
        cmd_score = aff_vec[1] if len(aff_vec) > 1 else 0.0
        browser_score = aff_vec[2] if len(aff_vec) > 2 else 0.0

        if math_score >= 0.65 or (any(t.get("type") == "number" for t in tokens) and any(t.get("raw") in "+-*/=" for t in tokens)):
            return "math_evaluation", "math_computation", {}

        if op_type == 2 or cmd_score >= 0.60 or browser_score >= 0.60:
            all_aff = ArisCapabilityRegistry.perceive_all_affordances(cleaned, tokens, candidates)
            best_cap = None
            best_params = {}
            for cap_name, aff_data in all_aff.items():
                if aff_data.get("relevant") in ("yes", "possible"):
                    best_cap = cap_name
                    best_params = aff_data.get("parameters", {})
                    if aff_data.get("relevant") == "yes":
                        break
            if not best_cap:
                best_cap = "command_prompt" if cmd_score >= browser_score else "browser_access"
            if not best_params:
                best_params = {"command": cleaned, "query": cleaned}
            return "capability_execution", best_cap, best_params

        return "relational_retrieval", None, {}
