"""
AR.I.S — Cognitive Linguistic Frame (Unified IR)

The single canonical Intermediate Representation produced by ArisLinguisticPipeline
and consumed by all downstream cognitive components.

Architectural Invariants:
  - ZERO HARD PATCHES (Rule 1): No string/punctuation checks anywhere in this module.
  - GENERIC INTELLIGENCE (Rule 4): All fields are produced by the trained tensor
    substrate; none are inferred from raw text spelling or character patterns.
  - SINGLE AUTHORITY: Exactly one authoritative speech_act value. Derived boolean
    properties are computed lazily from that single value.
  - OBSERVABLE STAGES: surface_frame (what the user said) and grounded_frame
    (after coreference resolution) are kept as separate, distinguishable objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Sentence-level span — produced by sentence segmentation
# ---------------------------------------------------------------------------

@dataclass
class SentenceSpan:
    """
    A sentence-segmented slice of the utterance together with the linguistic
    representations that belong to that sentence.

    Used by ArisCoreferenceResolver to perform incremental intra-turn
    anaphora resolution: entity anchors discovered in sentence i are
    immediately available to resolve pro-forms in sentence i+1.
    """
    sentence_index: int
    raw_text: str
    tokens: List[Dict[str, Any]] = field(default_factory=list)
    # UPOS-filtered entity candidates belonging to this sentence
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    # Structural S-P-O bindings within this sentence
    bindings: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dependency token — represents one token's position in the dependency graph
# ---------------------------------------------------------------------------

@dataclass
class DependencyToken:
    """
    Represents a single token's role in the sentence dependency structure.
    Produced by ArisLinguisticPipeline from the UD tensor engine.

    Example tree for "Arunbharathi founded Axinix Technologies.":
        founded  (root)
          nsubj → Arunbharathi
          obj   → Axinix Technologies
    """
    token: str
    raw: str
    sequence_index: int
    upos: str                        # PROPN, NOUN, VERB, AUX, INTJ, …
    upos_confidence: float
    deprel: Optional[str] = None     # nsubj, obj, obl, root, cop, aux, …
    deprel_confidence: float = 0.0
    head_sequence_index: Optional[int] = None  # index of the syntactic head token


# ---------------------------------------------------------------------------
# Core Cognitive Linguistic Frame
# ---------------------------------------------------------------------------

@dataclass
class CognitiveLinguisticFrame:
    """
    The authoritative Intermediate Representation for one utterance, produced
    entirely by the trained linguistic tensor substrate.

    Downstream components — coreference resolver, syntactic normalizer,
    brain router — must read from this frame exclusively.
    They must NOT independently re-classify the utterance from raw text.

    Stages:
        surface_frame  — what the user linguistically produced (immutable after
                         ArisLinguisticPipeline.process())
        grounded_frame — same type, produced after coreference resolution with
                         resolved_text replacing raw_utterance and updated
                         candidates/bindings derived from the resolved text.
    """

    # ---- Raw input (preserved) -------------------------------------------
    raw_utterance: str

    # ---- Speech act (sole authoritative source) --------------------------
    # Produced by ArisPragmaticTensorEngine.classify_speech_act().
    # No downstream component may override or re-derive this value.
    speech_act: str
    speech_act_scores: Dict[str, float]

    # ---- Linguistic representations (tensor-derived) ---------------------
    # UPOS-enriched token list (each token carries .upos, .upos_confidence)
    tokens: List[Dict[str, Any]]

    # UPOS-filtered entity candidates (AUX, INTJ, VERB, etc. excluded)
    candidates: List[Dict[str, Any]]

    # Per-token dependency structure from the UD tensor engine
    dependency_structure: List[DependencyToken]

    # Structural S-P-O relational bindings (Tier 3)
    bindings: List[Dict[str, Any]]

    # Sentence-segmented spans for incremental coreference resolution
    sentence_spans: List[SentenceSpan]

    # ---- Derived boolean properties (computed, NOT stored separately) -----
    @property
    def is_question(self) -> bool:
        """True when the tensor-classified speech act is INTERROGATIVE."""
        return self.speech_act == "INTERROGATIVE"

    @property
    def is_declarative(self) -> bool:
        """True when the tensor-classified speech act is DECLARATIVE."""
        return self.speech_act == "DECLARATIVE"

    @property
    def is_command(self) -> bool:
        """True when the tensor-classified speech act is COMMAND."""
        return self.speech_act == "COMMAND"

    @property
    def is_phatic(self) -> bool:
        """True when the tensor-classified speech act is PHATIC or THANKING."""
        return self.speech_act in ("PHATIC", "THANKING")

    @property
    def is_denial(self) -> bool:
        """True when the tensor-classified speech act is DENIAL."""
        return self.speech_act == "DENIAL"

    @property
    def is_correction(self) -> bool:
        """True when the tensor-classified speech act is CORRECTION."""
        return self.speech_act == "CORRECTION"


# ---------------------------------------------------------------------------
# Grounded Cognitive Linguistic Frame
# ---------------------------------------------------------------------------

@dataclass
class GroundedCognitiveLinguisticFrame:
    """
    The discourse-resolved representation produced after coreference grounding.

    Keeps the original surface_frame for full observability and debugging.
    The grounded content (resolved utterance, updated candidates, bindings)
    is stored as a new CognitiveLinguisticFrame rather than overwriting.

    Usage:
        surface  = ArisLinguisticPipeline.process(raw_utterance)
        coref    = ArisCoreferenceResolver.resolve_utterance(surface, context)
        grounded = ArisLinguisticPipeline.process(coref["resolved_text"])
        gframe   = GroundedCognitiveLinguisticFrame(
                       surface_frame=surface,
                       grounded_frame=grounded,
                       resolved_coreferences=coref["resolved_coreferences"],
                       speaker_bindings=coref["speaker_bindings"],
                   )
    """
    surface_frame: CognitiveLinguisticFrame
    grounded_frame: CognitiveLinguisticFrame
    resolved_coreferences: List[Dict[str, Any]] = field(default_factory=list)
    speaker_bindings: Dict[str, Any] = field(default_factory=dict)

    # Convenience delegates to the grounded frame
    @property
    def speech_act(self) -> str:
        return self.grounded_frame.speech_act

    @property
    def speech_act_scores(self) -> Dict[str, float]:
        return self.grounded_frame.speech_act_scores

    @property
    def candidates(self) -> List[Dict[str, Any]]:
        return self.grounded_frame.candidates

    @property
    def bindings(self) -> List[Dict[str, Any]]:
        return self.grounded_frame.bindings

    @property
    def tokens(self) -> List[Dict[str, Any]]:
        return self.grounded_frame.tokens

    @property
    def dependency_structure(self) -> List[DependencyToken]:
        return self.grounded_frame.dependency_structure

    @property
    def resolved_text(self) -> str:
        return self.grounded_frame.raw_utterance
