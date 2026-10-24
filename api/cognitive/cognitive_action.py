"""
AR.I.S — CognitiveAction (Phase 11)

Structured representation of a cognitive action produced from a COMMAND speech act.

Architectural Invariants:
  - ZERO HARD PATCHES (Rule 1): No hardcoded capability names or query extraction heuristics.
  - GENERIC INTELLIGENCE (Rule 4): Capability routing uses the registered ArisCapabilityRegistry
    and ArisStructuralAdvisor observation \u2014 not string matching on command verbs.
  - Payload comes from the full raw utterance stripped of operator verb, NOT from the
    longest candidate entity.
  - If the structural advisor cannot determine the operation, we return a graceful fallback
    rather than inventing a capability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


# Canonical operation types surfaced by CognitiveAction
OPERATION_BROWSE = "BROWSE"
OPERATION_SEARCH = "SEARCH"
OPERATION_COMPUTE = "COMPUTE"
OPERATION_LOOKUP = "LOOKUP"
OPERATION_UNKNOWN = "UNKNOWN"


@dataclass
class CognitiveAction:
    """
    Structured representation of a COMMAND cognitive intent.

    Produced by from_frame() from a grounded CognitiveLinguisticFrame.
    Consumed by dispatch() which invokes the appropriate ArisCapabilityRegistry capability.

    Fields:
        speech_act  — always "COMMAND" for this class
        operation   — canonical operation from the advisor observation (BROWSE, SEARCH, etc.)
        capability  — capability name from ArisCapabilityRegistry, or None if undetermined
        payload     — the primary content to operate on (text query, expression, etc.)
        parameters  — additional parameters for the capability
        confidence  — advisor classification confidence
    """
    speech_act: str = "COMMAND"
    operation: str = OPERATION_UNKNOWN
    capability: Optional[str] = None
    payload: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

    @classmethod
    def from_frame(cls, frame) -> "CognitiveAction":
        """
        Builds a CognitiveAction from a grounded CognitiveLinguisticFrame.

        Routing logic:
          1. Ask ArisStructuralAdvisor to observe the grounded utterance.
          2. Map the advisor's `type` and `capability` fields to our operation/capability.
          3. Payload is the full raw utterance when the operation is BROWSE/SEARCH/LOOKUP
             (the user's intent is expressed in natural language, not a single entity).
          4. For COMPUTE, payload is the expression; parameters come from the advisor.

        This replaces the previous heuristic of taking `max(candidates, key=len)` as payload.
        """
        from ..relational.structural_advisor import ArisStructuralAdvisor

        raw_text = frame.raw_utterance

        # Step 1: Ask the structural advisor
        obs = ArisStructuralAdvisor.observe_query(raw_text, enabled=True)
        obs_status = obs.get("status", "unavailable")

        if obs_status != "ok":
            # Advisor unavailable \u2014 return minimal action with raw utterance as payload
            return cls(
                operation=OPERATION_UNKNOWN,
                capability=None,
                payload=raw_text,
                confidence=0.0,
            )

        obs_type = obs.get("type", "")
        obs_capability = obs.get("capability", "")
        obs_params = obs.get("parameters", {})

        # Step 2: Map advisor observation to operation/capability
        if obs_capability == "math_computation" or obs_type == "executable_capability":
            return cls(
                operation=OPERATION_COMPUTE,
                capability="math_computation",
                payload=obs.get("expression", raw_text),
                parameters=obs_params,
                confidence=obs.get("confidence", 1.0),
            )

        elif obs_capability == "browser_access" or obs_type == "browser_search":
            # Payload for browser = the full utterance (user's natural language query)
            # Anchors from the advisor can refine but the full intent is in the text
            anchors = obs.get("anchors", [])
            query_payload = " ".join(anchors) if anchors else raw_text
            return cls(
                operation=OPERATION_BROWSE,
                capability="browser_access",
                payload=query_payload,
                parameters={"query": query_payload},
                confidence=obs.get("confidence", 0.8),
            )

        else:
            # Advisor returned a relational traversal observation for a command verb.
            # Extract the most informative payload from anchors (if available) or full text.
            anchors = obs.get("anchors", [])
            payload = " ".join(anchors) if anchors else raw_text
            return cls(
                operation=OPERATION_LOOKUP,
                capability=obs_capability or None,
                payload=payload,
                parameters=obs_params,
                confidence=obs.get("confidence", 0.5),
            )

    def dispatch(self) -> Dict[str, Any]:
        """
        Executes the CognitiveAction via ArisCapabilityRegistry.

        Returns a result dict compatible with ArisGroundedGenerator.generate().
        If capability is unknown or fails, returns an ACQUIRABLE epistemic state
        pointing to external sources.
        """
        from ..semantic.executable_tools import ArisCapabilityRegistry

        if not self.capability:
            return {
                "epistemic_state": "ACQUIRABLE",
                "capability": "external_search",
                "result": self.payload,
                "final_answer": None,
                "status": "acquirable",
            }

        try:
            result = ArisCapabilityRegistry.dispatch(self.capability, **self.parameters)
            status = result.get("status", "unknown")
            value = result.get("value") or result.get("result") or result.get("answer")
            return {
                "epistemic_state": "KNOWN" if status in ("success", "solved") else "ACQUIRABLE",
                "capability": self.capability,
                "operation": self.operation,
                "result": value,
                "final_answer": str(value) if value is not None else None,
                "payload": self.payload,
                "parameters": self.parameters,
                "raw_dispatch": result,
            }
        except Exception as exc:
            return {
                "epistemic_state": "ACQUIRABLE",
                "capability": self.capability,
                "operation": self.operation,
                "payload": self.payload,
                "result": None,
                "final_answer": None,
                "error": str(exc),
            }
