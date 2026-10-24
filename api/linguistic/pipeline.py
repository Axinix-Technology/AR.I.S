"""
AR.I.S — Unified Linguistic Pipeline

Single entry point for all linguistic interpretation.
Produces a CognitiveLinguisticFrame that represents the authoritative
linguistic reading of an utterance, derived entirely from the trained
tensor substrate.

Architectural Invariants:
  - ZERO HARD PATCHES (Rule 1): No string spelling checks, no capitalization
    heuristics, no punctuation character guards.
  - GENERIC INTELLIGENCE (Rule 4): Every linguistic decision (UPOS, candidate
    status, dependency role, speech act) comes from trained tensor matrices.
  - SINGLE ENTRY POINT: All cognitive components call process() once and read
    from the returned CognitiveLinguisticFrame. They must NOT independently
    re-tokenize or re-classify the same utterance.
  - LANGUAGE-INDEPENDENT: Works for any language represented in the UD v2.18
    tensor bank (English, Tamil, French, Hindi, German, and others).
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Set

from .tokenizer import ArisTokenizer
from .ud_tensor_engine import ArisUniversalLinguisticEncoder
from .cognitive_frame import (
    CognitiveLinguisticFrame,
    GroundedCognitiveLinguisticFrame,
    DependencyToken,
    SentenceSpan,
)
from ..relational.relational import ArisRelationalBinder
from ..cognitive.pragmatic_tensor_engine import ArisPragmaticTensorEngine


# ---------------------------------------------------------------------------
# UPOS tags whose tokens must never become entity candidates.
# This list is the boundary of an architectural contract, not a linguistic
# heuristic: the tags were assigned by the trained UD tensor bank.
# The decision of whether a token is AUX, INTJ, VERB, etc. is the brain's —
# this filter merely enforces the contract that non-nominals are not entities.
# ---------------------------------------------------------------------------
_NON_ENTITY_UPOS: Set[str] = {
    "AUX",    # auxiliary modals: Could, Can, Would, Will, Should, Have, Be
    "INTJ",   # interjections / greetings: Hi, Hello, Hey, Vanakkam, Bonjour
    "PRON",   # pronouns: I, my, it, she, they, he (handled by deictic learner)
    "SCONJ",  # subordinating conjunctions: that, because, if
    "ADP",    # adpositions / prepositions: of, in, at, by
    "DET",    # determiners: the, a, an, this, that
    "PUNCT",  # punctuation marks
    "CCONJ",  # coordinating conjunctions: and, or, but
    "PART",   # particles: not, to (infinitive marker)
    "ADV",    # adverbs: where, when, how — kept here; interrogative roles
              # are handled downstream by ArisInterrogativeRoleLearner
}


class ArisLinguisticPipeline:
    """
    Unified linguistic pipeline: tokenize → enrich → extract → bind → classify → frame.

    Every cognitive component that needs linguistic interpretation calls
    ArisLinguisticPipeline.process(text) and reads from the returned frame.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def process(cls, text: str) -> CognitiveLinguisticFrame:
        """
        Produces a complete CognitiveLinguisticFrame from raw text.

        Pipeline sequence (all tensor-driven, no raw-string heuristics):
          1. Tokenization (Tier 1) — exact positional coordinates.
          2. UD/tensor linguistic enrichment — assigns UPOS + confidence.
          3. Dependency structure — assigns deprel + head per token.
          4. Structural candidate extraction (Tier 2) — base anchors.
          5. Relational binding (Tier 3) — S-P-O triples over structural anchors.
          6. UPOS candidate filtering — nominal entity candidates for frame.
          7. Speech-act classification — from ArisPragmaticTensorEngine.
          8. Sentence segmentation — builds SentenceSpan list for coreference.
          9. Frame construction — assembles authoritative CognitiveLinguisticFrame.
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return cls._empty_frame(cleaned)

        # Step 1 — Tokenize (Tier 1)
        tokens = ArisTokenizer.tokenize(cleaned)

        # Step 2 — UD UPOS enrichment (calls trained tensor bank)
        tokens = ArisTokenizer.enrich_tokens_with_syntax(tokens)

        # Step 3 — Dependency structure per token
        dependency_structure = cls._build_dependency_structure(tokens)

        # Step 4 — Structural candidate extraction (Tier 2)
        raw_candidates = ArisTokenizer.extract_candidates(cleaned, tokens)

        # Step 5 — Relational bindings over structural anchors (Tier 3)
        bindings = ArisRelationalBinder.bind_relations(cleaned, tokens, raw_candidates)

        # Step 6 — UPOS-filtered entity candidates (Tier 2 refined)
        candidates = cls._extract_upos_filtered_candidates(cleaned, tokens, raw_candidates=raw_candidates, bindings=bindings)

        # Step 7 — Speech act classification (tensor — no char overrides)
        speech_act_result = ArisPragmaticTensorEngine.classify_speech_act(cleaned)
        speech_act = speech_act_result["dominant_act"]
        speech_act_scores = speech_act_result["scores"]

        # Step 8 — Sentence segmentation
        sentence_spans = cls._segment_sentences(cleaned, tokens, candidates, bindings)

        # Step 9 — Assemble frame
        return CognitiveLinguisticFrame(
            raw_utterance=cleaned,
            speech_act=speech_act,
            speech_act_scores=speech_act_scores,
            tokens=tokens,
            candidates=candidates,
            dependency_structure=dependency_structure,
            bindings=bindings,
            sentence_spans=sentence_spans,
        )

    @classmethod
    def build_grounded_frame(
        cls,
        surface_frame: CognitiveLinguisticFrame,
        coref_result: Dict[str, Any],
    ) -> GroundedCognitiveLinguisticFrame:
        """
        Produces a GroundedCognitiveLinguisticFrame from a coreference result.

        Keeps the original surface_frame intact for full observability.
        Re-processes the resolved text through the full pipeline to produce
        the grounded frame's tokens, candidates, and bindings.

        Args:
            surface_frame: The original frame before coreference resolution.
            coref_result:  Output of ArisCoreferenceResolver.resolve_utterance().

        Returns:
            GroundedCognitiveLinguisticFrame with both stages preserved.
        """
        resolved_text = coref_result.get("resolved_text", surface_frame.raw_utterance)
        grounded_frame = cls.process(resolved_text)
        return GroundedCognitiveLinguisticFrame(
            surface_frame=surface_frame,
            grounded_frame=grounded_frame,
            resolved_coreferences=coref_result.get("resolved_coreferences", []),
            speaker_bindings=coref_result.get("speaker_bindings", {}),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _extract_upos_filtered_candidates(
        cls,
        text: str,
        tokens: List[Dict[str, Any]],
        raw_candidates: Optional[List[Dict[str, Any]]] = None,
        bindings: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracts entity candidates using the base structural extractor, then
        removes candidates whose constituent tokens are entirely non-nominal
        UPOS tags as determined by the trained UD tensor bank.
        """
        # Nominal UPOS tags — presence of any of these admits the candidate
        _NOMINAL_UPOS: Set[str] = {"PROPN", "NOUN", "X"}

        if raw_candidates is None:
            raw_candidates = ArisTokenizer.extract_candidates(text, tokens)

        # Ingest tensor-identified nominal tokens (UPOS == PROPN or NOUN) from trained tensor engine.
        consumed_indices = set()
        for cand in raw_candidates:
            consumed_indices.update(cand.get("token_indices", []))
        if bindings:
            for b in bindings:
                rel = b.get("relation", {})
                if isinstance(rel, dict):
                    consumed_indices.update(rel.get("token_indices", []))

        n_tok = len(tokens)
        i = 0
        while i < n_tok:
            if i in consumed_indices:
                i += 1
                continue
            tok = tokens[i]
            if tok.get("type") == "word" and tok.get("upos") in ("PROPN", "NOUN"):
                seq_start = i
                sent_idx = tok["position"]["sentence_index"]
                while (
                    i + 1 < n_tok
                    and (i + 1) not in consumed_indices
                    and tokens[i + 1].get("type") == "word"
                    and tokens[i + 1].get("upos") in ("PROPN", "NOUN")
                    and tokens[i + 1]["position"]["sentence_index"] == sent_idx
                ):
                    i += 1
                seq_end = i
                propn_tokens = tokens[seq_start:seq_end + 1]
                t_indices = [t["sequence_index"] for t in propn_tokens]
                s_char = propn_tokens[0]["position"]["char_start"]
                e_char = propn_tokens[-1]["position"]["char_end"]
                raw_candidates.append({
                    "text": text[s_char:e_char],
                    "normalized": text[s_char:e_char].lower(),
                    "candidate_type": "entity_candidate",
                    "token_indices": t_indices,
                    "char_start": s_char,
                    "char_end": e_char,
                    "sentence_index": sent_idx,
                })
                consumed_indices.update(t_indices)
            i += 1

        # Build a map: sequence_index → (upos, upos_confidence) for quick lookup
        upos_map: Dict[int, str] = {
            t["sequence_index"]: t.get("upos", "X")
            for t in tokens
        }

        filtered = []
        for cand in raw_candidates:
            cand_token_indices = set(cand.get("token_indices", []))
            if not cand_token_indices:
                # No positional information — trust structural extractor
                filtered.append(cand)
                continue

            # Collect UPOS tags of all constituent tokens
            upos_tags = {upos_map.get(idx, "X") for idx in cand_token_indices}
            is_pure_number = all(tokens[idx].get("type") == "number" for idx in cand_token_indices if idx < len(tokens))
            if is_pure_number:
                continue

            cand_type = cand.get("candidate_type", "")

            # Structural entity candidates (e.g. mid-sentence capitalized proper nouns like 'Arun', 'Axinix')
            if cand_type == "entity_candidate":
                if upos_tags.issubset({"PUNCT", "SYM", "PRON"}):
                    continue
                filtered.append(cand)
                continue

            # Ambiguous sentence-initial capitalized candidates (e.g. 'Where', 'Hi', 'Could', 'Does')
            if upos_tags.issubset(_NON_ENTITY_UPOS):
                continue

            # Sentence-initial nominal tokens (e.g. 'Arun works at Axinix')
            if upos_tags & _NOMINAL_UPOS:
                filtered.append(cand)
                continue

        return filtered

    @classmethod
    def _build_dependency_structure(
        cls,
        tokens: List[Dict[str, Any]],
    ) -> List[DependencyToken]:
        """
        Builds a per-token dependency representation using the trained UD
        tensor engine's pairwise dependency relation predictor.

        Current approach: for each word token, predicts its dependency
        relation with respect to its immediate left and right neighbours,
        then selects the highest-confidence assignment as the primary deprel.

        Note: The UD tensor engine currently exposes a pairwise predictor
        (`predict_dependency_relation(head, dependent)`). A full sentence-level
        dependency parse would require extending the engine to produce a
        complete dependency tree (head per token). This is Phase 5's boundary.
        For now, we record the best pairwise deprel available so downstream
        components have SOME structural signal. The syntactic normalizer
        (Phase 9) consumes this structure; if it is insufficient for a given
        language, Phase 5 must extend the engine rather than fall back to
        heuristics.
        """
        dep_tokens: List[DependencyToken] = []
        word_tokens = [t for t in tokens if t.get("type") == "word"]
        n = len(word_tokens)

        for i, tok in enumerate(word_tokens):
            best_deprel: Optional[str] = None
            best_conf: float = 0.0
            best_head_idx: Optional[int] = None

            # Compare with left neighbour (head candidate)
            if i > 0:
                left = word_tokens[i - 1]
                rel, conf = ArisUniversalLinguisticEncoder.predict_dependency_relation(
                    left["raw"], tok["raw"]
                )
                if conf > best_conf:
                    best_conf = conf
                    best_deprel = rel
                    best_head_idx = left["sequence_index"]

            # Compare with right neighbour (head candidate)
            if i < n - 1:
                right = word_tokens[i + 1]
                rel, conf = ArisUniversalLinguisticEncoder.predict_dependency_relation(
                    right["raw"], tok["raw"]
                )
                if conf > best_conf:
                    best_conf = conf
                    best_deprel = rel
                    best_head_idx = right["sequence_index"]

            dep_tokens.append(DependencyToken(
                token=tok.get("token", tok["raw"].lower()),
                raw=tok["raw"],
                sequence_index=tok["sequence_index"],
                upos=tok.get("upos", "X"),
                upos_confidence=tok.get("upos_confidence", 0.0),
                deprel=best_deprel,
                deprel_confidence=round(best_conf, 4),
                head_sequence_index=best_head_idx,
            ))

        return dep_tokens

    @classmethod
    def _segment_sentences(
        cls,
        text: str,
        tokens: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        bindings: List[Dict[str, Any]],
    ) -> List[SentenceSpan]:
        """
        Segments the utterance into sentence-level spans using the
        sentence_index field already produced by ArisTokenizer (Tier 1).

        Each SentenceSpan groups the tokens, candidates, and bindings that
        belong to one sentence, enabling sentence-by-sentence incremental
        anaphora resolution in ArisCoreferenceResolver.
        """
        sentence_map: Dict[int, Dict[str, Any]] = {}

        # Group tokens by sentence_index
        for tok in tokens:
            s_idx = tok["position"]["sentence_index"]
            if s_idx not in sentence_map:
                sentence_map[s_idx] = {
                    "tokens": [],
                    "candidates": [],
                    "bindings": [],
                    "char_start": tok["position"]["char_start"],
                    "char_end": tok["position"]["char_end"],
                }
            sentence_map[s_idx]["tokens"].append(tok)
            sentence_map[s_idx]["char_end"] = max(
                sentence_map[s_idx]["char_end"], tok["position"]["char_end"]
            )

        # Assign candidates and bindings to their sentence
        for cand in candidates:
            s_idx = cand.get("sentence_index", 0)
            if s_idx in sentence_map:
                sentence_map[s_idx]["candidates"].append(cand)

        for b in bindings:
            s_idx = b.get("sentence_index", 0)
            if s_idx in sentence_map:
                sentence_map[s_idx]["bindings"].append(b)

        # Build SentenceSpan objects
        spans: List[SentenceSpan] = []
        for s_idx in sorted(sentence_map.keys()):
            entry = sentence_map[s_idx]
            char_start = entry["char_start"]
            char_end = entry["char_end"]
            raw_text = text[char_start:char_end].strip()
            spans.append(SentenceSpan(
                sentence_index=s_idx,
                raw_text=raw_text,
                tokens=entry["tokens"],
                candidates=entry["candidates"],
                bindings=entry["bindings"],
            ))

        return spans

    @classmethod
    def _empty_frame(cls, text: str) -> CognitiveLinguisticFrame:
        """Returns a minimal frame for an empty or whitespace-only utterance."""
        return CognitiveLinguisticFrame(
            raw_utterance=text,
            speech_act="PHATIC",
            speech_act_scores={"PHATIC": 1.0},
            tokens=[],
            candidates=[],
            dependency_structure=[],
            bindings=[],
            sentence_spans=[],
        )
