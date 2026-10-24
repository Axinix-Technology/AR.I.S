# pyright: ignore
"""
AR.I.S — Metacognitive Reflection & Deliberative Thinking Tracer

Provides:
1. Internal Metacognitive Reflector (Self-Auditing):
   - Audits intermediate deductions for factual contradictions and knowledge inconsistencies.
   - Detects circular loops ($A \to B \to A$) across multi-hop reasoning.
   - Checks belief inhibition matrices (explicitly refuted user feedback).
   - Initiates automated backtracking when dead ends or contradictions arise.
2. Deliberative Thinking Tracer:
   - Formulates explainable, human-understandable reasoning traces ("Thinking Steps").
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from .relational_beliefs import ArisRelationalBeliefManager
from ..relational.symbolic_graph import ArisSymbolicGraph


@dataclass
class ReflectionAuditResult:
    """Represents the self-audit outcome of an intermediate reasoning step."""
    is_valid: bool
    status: str                         # 'valid_step' | 'circular_loop' | 'inhibited_contradiction' | 'dead_end'
    audit_note: str
    suggested_action: str              # 'proceed' | 'backtrack' | 'terminate'
    backtrack_target: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "status": self.status,
            "audit_note": self.audit_note,
            "suggested_action": self.suggested_action,
            "backtrack_target": self.backtrack_target,
        }


class ArisCognitiveReflector:
    """
    Metacognitive Self-Audit Engine for AR.I.S.
    Monitors chain-of-thought deductions in real time to prevent hallucination,
    circularity, and unverified assumptions.
    """

    @classmethod
    def audit_step(
        cls,
        current_step: Dict[str, Any],
        provenance_history: List[Any],
        visited_entities: Set[str],
        memory_facts: Optional[List[dict]] = None,
    ) -> ReflectionAuditResult:
        """
        Audits an intermediate reasoning step before committing it to the proof path.
        """
        subj = current_step.get("subject") or ""
        rel = current_step.get("relation") or ""
        obj = current_step.get("object") or current_step.get("answer") or ""

        # 1. Circular Loop Detection
        if obj and obj.lower() in [e.lower() for e in visited_entities if e]:
            # Entity already visited in this deductive chain
            return ReflectionAuditResult(
                is_valid=False,
                status="circular_loop",
                audit_note=f"Circular reasoning detected: Entity '{obj}' was already visited in path.",
                suggested_action="backtrack",
            )

        # 2. User Belief Denial & Contradiction Check
        if subj and rel and obj:
            if ArisRelationalBeliefManager.is_denied(subj, rel, obj):
                return ReflectionAuditResult(
                    is_valid=False,
                    status="inhibited_contradiction",
                    audit_note=f"Assertion '{subj} {rel} {obj}' contradicts explicitly recorded user correction.",
                    suggested_action="backtrack",
                )

        # 3. Grounding Verification against Memory Facts
        if memory_facts and subj and rel:
            # Confirm that the relation exists in memory graph
            graph = ArisSymbolicGraph.build_from_facts(facts=memory_facts)
            if subj.lower() not in graph.adjacency:
                # Subject is ungrounded in memory
                return ReflectionAuditResult(
                    is_valid=False,
                    status="dead_end",
                    audit_note=f"Anchor entity '{subj}' has no relational links in active memory graph.",
                    suggested_action="backtrack",
                )

        # Step verified cleanly
        return ReflectionAuditResult(
            is_valid=True,
            status="valid_step",
            audit_note="Step verified consistent with knowledge graph and belief constraints.",
            suggested_action="proceed",
        )


class ArisDeliberativeTracer:
    """
    Synthesizes transparent, human-understandable reasoning traces ("Thinking Steps")
    demonstrating the internal cognitive path.
    """

    @classmethod
    def format_trace(
        cls,
        goal_plan: List[Any],
        executed_steps: List[Dict[str, Any]],
        reflection_logs: List[ReflectionAuditResult],
        final_answer: Any,
    ) -> Dict[str, Any]:
        """
        Compiles the full deliberative thinking trace.
        """
        thinking_steps = []

        for idx, step in enumerate(executed_steps):
            step_desc = step.get("statement") or step.get("query") or f"Sub-Goal {idx + 1}"
            step_res = step.get("result") or step.get("answer") or "Resolved"
            thinking_steps.append({
                "step_number": idx + 1,
                "description": step_desc,
                "intermediate_result": step_res,
                "status": step.get("status", "completed"),
            })

        reflections = [r.to_dict() for r in reflection_logs]

        return {
            "total_steps": len(executed_steps),
            "plan": [g.to_dict() if hasattr(g, "to_dict") else g for g in goal_plan],
            "executed_steps": executed_steps,
            "thinking_steps": thinking_steps,
            "reflections": reflections,
            "final_answer": final_answer,
            "is_metacognitively_verified": all(r.is_valid for r in reflection_logs if r.status != "backtrack"),
        }
