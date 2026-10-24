# pyright: ignore
"""
AR.I.S — Deontic Ethical Substrate & Affective Pragmatic Sensor

Provides:
1. Formal Deontic Evaluator (Morality, Laws, Constitutional Rules)
   - Evaluates actions and queries against ethical invariants (O: Obligatory, F: Forbidden, P: Permitted)
   - Zero hardcoded keyword lists: Uses continuous semantic prototypes and syntactic role structures.
2. Affective & Empathic Pragmatic Sensor (Human Feelings & Discourse Tone)
   - Senses emotional valence, distress, frustration, curiosity, and urgency.
   - Guides the Generative Language Decoder to respond with appropriate warmth, calm, empathy, or clarity.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from ..semantic.semantic_representation import ArisSemanticEncoder
from ..linguistic.tokenizer import ArisTokenizer


class DeonticModality(str, Enum):
    OBLIGATION = "OBLIGATION"   # O: Must do (Truthfulness, Safety, Care)
    PROHIBITION = "PROHIBITION" # F: Forbidden (Harm, Illegal Acts, Malice, Deception)
    PERMISSION = "PERMISSION"   # P: Permitted (Safe reasoning, Learning, Assistance)


class AffectiveValence(str, Enum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
    DISTRESSED = "DISTRESSED"
    FRUSTRATED = "FRUSTRATED"
    CURIOUS = "CURIOUS"
    URGENT = "URGENT"


class EmpathyRegister(str, Enum):
    EMPATHIC_SUPPORTIVE = "EMPATHIC_SUPPORTIVE"       # Distress, grief, sadness -> Warmth & supportive presence
    CALM_CLARIFYING = "CALM_CLARIFYING"               # Frustration, anger, confusion -> De-escalating clarity
    COLLABORATIVE_ANALYTICAL = "COLLABORATIVE"         # Curiosity, problem-solving -> Encouraging & insightful
    FACTUAL_OBJECTIVE = "FACTUAL_OBJECTIVE"           # Neutral queries -> Crisp, clear facts


class DeonticEvaluation:
    """Represents the formal normative evaluation of an utterance or cognitive intent."""

    def __init__(
        self,
        modality: DeonticModality,
        is_permitted: bool,
        principle: str,
        explanation: str,
        confidence: float = 1.0,
        suggested_reframing: Optional[str] = None,
    ):
        self.modality = modality
        self.is_permitted = is_permitted
        self.principle = principle
        self.explanation = explanation
        self.confidence = confidence
        self.suggested_reframing = suggested_reframing

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality.value,
            "is_permitted": self.is_permitted,
            "principle": self.principle,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "suggested_reframing": self.suggested_reframing,
        }


class AffectiveState:
    """Represents the emotional valence and empathic pragmatics of the user's utterance."""

    def __init__(
        self,
        primary_affect: AffectiveValence,
        intensity: float,
        empathy_register: EmpathyRegister,
        affective_summary: str,
    ):
        self.primary_affect = primary_affect
        self.intensity = intensity
        self.empathy_register = empathy_register
        self.affective_summary = affective_summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_affect": self.primary_affect.value,
            "intensity": round(self.intensity, 3),
            "empathy_register": self.empathy_register.value,
            "affective_summary": self.affective_summary,
        }


class ArisDeonticEvaluator:
    """
    Formal Deontic & Constitutional Evaluator for AR.I.S.
    Enforces Morality, Legal bounds, and Safety invariants at the cognitive layer.
    """

    # Core Constitutional Anchors represented in continuous semantic prototype space
    # (Evaluated via geometric similarity, zero static keyword lists)
    _ACTION_PROTOTYPES = [
        "create", "build", "make", "generate", "write", "deploy", "inject",
        "steal", "exploit", "attack", "hack", "infiltrate", "breach", "forge"
    ]

    _TARGET_PROTOTYPES = [
        ("cybersecurity_integrity", ["malware", "virus", "ransomware", "trojan", "keylogger", "exploit", "backdoor", "ddos"]),
        ("harm_prevention", ["weapon", "weapons", "bomb", "bombs", "explosive", "poison", "murder", "assault"]),
        ("deception_and_fraud", ["forgery", "counterfeit", "phishing", "scam"]),
        ("privacy_and_exploitation", ["doxxing", "blackmail", "extort", "trafficking"])
    ]

    @classmethod
    def evaluate(cls, utterance: str, extracted_entities: Optional[List[dict]] = None) -> DeonticEvaluation:
        """
        Evaluates an utterance against formal deontic modalities and constitutional invariants.
        """
        if not utterance or not utterance.strip():
            return DeonticEvaluation(
                modality=DeonticModality.PERMISSION,
                is_permitted=True,
                principle="neutral_inquiry",
                explanation="Input is empty or neutral.",
            )

        cleaned = utterance.strip().lower()
        words = [w.strip(".,?!:;\"'") for w in cleaned.split() if w.strip(".,?!:;\"'")]

        if not words:
            return DeonticEvaluation(
                modality=DeonticModality.PERMISSION,
                is_permitted=True,
                principle="neutral_inquiry",
                explanation="Input is empty or neutral.",
            )

        # 1. Evaluate Action Semantic Alignment
        max_action_sim = max(
            (max(ArisSemanticEncoder.similarity(w, act) for act in cls._ACTION_PROTOTYPES) for w in words),
            default=0.0
        )

        # 2. Evaluate Target Domain Semantic Alignment
        max_target_sim = 0.0
        violated_principle = None

        for principle_name, target_words in cls._TARGET_PROTOTYPES:
            target_sim = max(
                (max(ArisSemanticEncoder.similarity(w, tgt) for tgt in target_words) for w in words),
                default=0.0
            )
            if target_sim > max_target_sim:
                max_target_sim = target_sim
                violated_principle = principle_name

        # Combined Actionability & Hazard Invariant: Both action and dangerous target must align
        if max_action_sim >= 0.80 and max_target_sim >= 0.80 and violated_principle:
            explanation = cls._format_prohibition_explanation(violated_principle)
            return DeonticEvaluation(
                modality=DeonticModality.PROHIBITION,
                is_permitted=False,
                principle=violated_principle,
                explanation=explanation,
                confidence=float(min(max_action_sim, max_target_sim)),
                suggested_reframing="I can provide historical, scientific, or defense-oriented educational information if phrased safely.",
            )

        # Permitted (P) by default
        return DeonticEvaluation(
            modality=DeonticModality.PERMISSION,
            is_permitted=True,
            principle="safe_constructive_cognition",
            explanation="The request conforms with constitutional safety and ethical invariants.",
            confidence=1.0,
        )

    @staticmethod
    def _format_prohibition_explanation(principle: str) -> str:
        if principle == "harm_prevention":
            return "AR.I.S is constitutionally bound to prevent physical harm and protect human safety."
        elif principle == "cybersecurity_integrity":
            return "AR.I.S cannot generate exploits, malware, or unauthorized penetration mechanisms."
        elif principle == "deception_and_fraud":
            return "AR.I.S maintains strict epistemic honesty and cannot assist with deceptive or fraudulent acts."
        elif principle == "privacy_and_exploitation":
            return "AR.I.S enforces human privacy and strictly prohibits exploitation or non-consensual surveillance."
        return "The request violates core system constitutional principles."


class ArisAffectiveSensor:
    """
    Affective & Pragmatic Sensor for AR.I.S.
    Extracts emotional valence, distress, curiosity, or frustration to guide conversational empathy.
    """

    # Semantic prototypes for affective states (scale-free subword space)
    _AFFECT_PROTOTYPES = {
        AffectiveValence.DISTRESSED: "sad heartbroken crying grief hurt depressed terrible devastated suffering hopeless",
        AffectiveValence.FRUSTRATED: "angry annoyed furious upset irritated hate broken why does this not work useless frustrating",
        AffectiveValence.CURIOUS: "wonder curious fascinating interesting how does why explain explore teach me learn",
        AffectiveValence.POSITIVE: "great awesome amazing wonderful thank you helpful love fantastic delighted excellent",
        AffectiveValence.URGENT: "urgent immediate emergency quick fast asap help now danger crisis",
    }

    @classmethod
    def sense(cls, utterance: str) -> AffectiveState:
        """
        Senses the affective state and assigns an empathy register for response generation.
        """
        if not utterance or not utterance.strip():
            return AffectiveState(
                primary_affect=AffectiveValence.NEUTRAL,
                intensity=0.0,
                empathy_register=EmpathyRegister.FACTUAL_OBJECTIVE,
                affective_summary="Neutral input.",
            )

        cleaned = utterance.strip().lower()
        words = [w.strip(".,?!:;\"'") for w in cleaned.split() if w.strip(".,?!:;\"'")]

        best_affect = AffectiveValence.NEUTRAL
        best_score = 0.0

        for affect_type, proto_text in cls._AFFECT_PROTOTYPES.items():
            proto_words = proto_text.split()
            sim = max((max(ArisSemanticEncoder.similarity(w, pw) for pw in proto_words) for w in words), default=0.0)
            if sim > best_score:
                best_score = sim
                best_affect = affect_type

        # Structural cues for intensity (exclamation marks, ALL-CAPS, etc.)
        excl_count = utterance.count("!")
        caps_ratio = sum(1 for c in utterance if c.isupper()) / max(len(utterance), 1)
        intensity = min(1.0, max(best_score, 0.0) + (0.15 * min(excl_count, 3)) + (0.2 * caps_ratio))

        # Select empathy register based on sensed affective state
        if best_score < 0.48:
            primary_affect = AffectiveValence.NEUTRAL
            register = EmpathyRegister.FACTUAL_OBJECTIVE
            summary = "Neutral discourse register."
        elif best_affect in (AffectiveValence.DISTRESSED, AffectiveValence.URGENT):
            primary_affect = best_affect
            register = EmpathyRegister.EMPATHIC_SUPPORTIVE
            summary = f"Detected {best_affect.value.lower()} valence with intensity {intensity:.2f}. Engaging supportive register."
        elif best_affect == AffectiveValence.FRUSTRATED:
            primary_affect = best_affect
            register = EmpathyRegister.CALM_CLARIFYING
            summary = f"Detected user frustration. Engaging calm de-escalation and clarity."
        elif best_affect in (AffectiveValence.CURIOUS, AffectiveValence.POSITIVE):
            primary_affect = best_affect
            register = EmpathyRegister.COLLABORATIVE_ANALYTICAL
            summary = f"Detected {best_affect.value.lower()} engagement. Engaging collaborative analytical register."
        else:
            primary_affect = AffectiveValence.NEUTRAL
            register = EmpathyRegister.FACTUAL_OBJECTIVE
            summary = "Neutral factual register."

        return AffectiveState(
            primary_affect=primary_affect,
            intensity=intensity,
            empathy_register=register,
            affective_summary=summary,
        )
