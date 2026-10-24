# pyright: ignore
"""
AR.I.S — Cognitive Semantic Frame & Intermediate Representation (IR)

Provides the domain-agnostic, language-independent cognitive interface between
natural language linguistics and the reasoning brain.

Architectural Invariants:
  1. ZERO HARD PATCHES (Rule 1): Continuous prototype scoring over communicative speech acts.
  2. LANGUAGE-INDEPENDENT IR: Brain operates strictly on CognitiveSemanticFrame & QueryFrame IRs.
  3. EXPLICIT OPEN-VARIABLE DECOMPOSITION: Interrogative operators (?x, ?location, ?person)
     specify the unknown role without polluting entity candidate space.
  4. DISCOURSE STATE SEPARATION: Working memory anaphora is cleanly separated from world KG facts.
"""

from typing import Dict, Any, List, Optional, Tuple
from ..semantic.semantic_representation import ArisSemanticEncoder
from ..db import get_mongo_db


from .pragmatic_tensor_engine import ArisPragmaticTensorEngine


class SpeechActType:
    PHATIC = "PHATIC"
    DECLARATIVE = "DECLARATIVE"
    INTERROGATIVE = "INTERROGATIVE"
    DENIAL = "DENIAL"
    CORRECTION = "CORRECTION"
    THANKING = "THANKING"
    COMMAND = "COMMAND"


class ArisSpeechActClassifier:
    """
    Compositional continuous speech act classifier.
    Powered directly by in-memory PyTorch tensor matrix projections.
    """

    @classmethod
    def classify(cls, utterance: str) -> Dict[str, Any]:
        """
        Computes continuous vector similarity distribution across all communicative speech acts.
        """
        return ArisPragmaticTensorEngine.classify_speech_act(utterance)


class ArisInterrogativeRoleLearner:
    """
    Continuous Semantic Role Prototype Classifier for Interrogative Queries.
    Powered directly by in-memory PyTorch tensor matrix projections.
    """

    ROLE_LOCATION = "location"
    ROLE_PERSON = "person"
    ROLE_TIME = "time"
    ROLE_CAUSE = "cause"
    ROLE_ENTITY = "entity"
    ROLE_QUANTITY = "quantity"

    @classmethod
    def infer_target_role(cls, operator_token: str) -> Tuple[str, float]:
        """
        Projects question operator token into continuous PyTorch role prototype tensor space.
        """
        return ArisPragmaticTensorEngine.infer_interrogative_role(operator_token)


class QueryFrame:
    """
    Intermediate Representation for an open-variable interrogative query.
    """

    def __init__(
        self,
        variable: str = "?x",
        anchors: Optional[List[str]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        expected_role: str = "entity",
        raw_query: str = "",
    ):
        self.variable = variable
        self.anchors = anchors or []
        self.constraints = constraints or []
        self.expected_role = expected_role
        self.raw_query = raw_query

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variable": self.variable,
            "anchors": self.anchors,
            "constraints": self.constraints,
            "expected_role": self.expected_role,
            "raw_query": self.raw_query,
        }


class CognitiveSemanticFrame:
    """
    Language-independent intermediate semantic representation produced
    from linguistic processing and consumed by the cognitive reasoning brain.
    """

    def __init__(
        self,
        raw_utterance: str,
        speech_act: str,
        speech_act_scores: Dict[str, float],
        propositions: Optional[List[Dict[str, Any]]] = None,
        anchors: Optional[List[str]] = None,
        query_frame: Optional[QueryFrame] = None,
        anaphora_bindings: Optional[List[Dict[str, Any]]] = None,
        discourse_updates: Optional[List[Dict[str, Any]]] = None,
    ):
        self.raw_utterance = raw_utterance
        self.speech_act = speech_act
        self.speech_act_scores = speech_act_scores
        self.propositions = propositions or []
        self.anchors = anchors or []
        self.query_frame = query_frame
        self.anaphora_bindings = anaphora_bindings or []
        self.discourse_updates = discourse_updates or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_utterance": self.raw_utterance,
            "speech_act": self.speech_act,
            "speech_act_scores": self.speech_act_scores,
            "propositions": self.propositions,
            "anchors": self.anchors,
            "query_frame": self.query_frame.to_dict() if self.query_frame else None,
            "anaphora_bindings": self.anaphora_bindings,
            "discourse_updates": self.discourse_updates,
        }
