"""
AR.I.S — Dynamic Coreference Resolver & Speaker Grounding Engine (Phase 1)

Resolves:
  1. Speaker Deictic Grounding: Binds speaker-referent frames to conversational participant (USER).
  2. Dynamic Cross-Turn Anaphora: Resolves pro-forms to antecedent entities via continuous
     prototype vector projection and recency-weighted salience ranking.

Key Invariants:
  - ZERO HARD PATCHES (Rule 1): Zero hardcoded word lists or regex dictionaries.
  - GENERIC INTELLIGENCE (Rule 4): Works across any natural language (English, Tamil, French)
    or alien vocabulary by projecting into continuous deictic prototype coordinate space.
"""

from typing import Dict, Any, List, Optional, Set
from ..linguistic.tokenizer import ArisTokenizer
from ..relational.relational import ArisRelationalBinder
from .dialogue_context import ArisDialogueContext
from .deictic_learner import ArisDeicticLearner
from .pragmatic_tensor_engine import ArisPragmaticTensorEngine


class ArisCoreferenceResolver:
    """
    Dynamic coreference and deictic resolution engine for multi-turn dialogue.
    Operates strictly via continuous coordinate vector prototypes and structural relational triples.

    Phases 6+7 architectural contract:
      - resolve_utterance() accepts a CognitiveLinguisticFrame as its primary argument.
      - speech_act is read from frame.speech_act — no punctuation char check.
      - Anaphora is resolved incrementally: entities discovered in sentence i are
        injected into the working salience state before processing sentence i+1.
      - The original raw-text signature is kept as a backward-compatible fallback
        that builds a minimal frame internally.
    """

    @classmethod
    def resolve_utterance(
        cls,
        text_or_frame,
        context: ArisDialogueContext,
        speaker_role: str = "USER",
    ) -> Dict[str, Any]:
        """
        Analyzes the utterance sentence-by-sentence, projects tokens into deictic
        prototype space, resolves anaphoric pro-forms against the active salience
        graph (updated incrementally per sentence), and binds speaker-referent relations.

        Accepts either:
          - A CognitiveLinguisticFrame (preferred — frame-based path)
          - A raw text string (deprecated — builds a minimal frame internally)
        """
        from ..linguistic.cognitive_frame import CognitiveLinguisticFrame as CLF
        from ..linguistic.pipeline import ArisLinguisticPipeline

        # --- Normalise input --------------------------------------------------
        if isinstance(text_or_frame, CLF):
            frame = text_or_frame
            cleaned_text = frame.raw_utterance
            tokens = frame.tokens
            candidates = frame.candidates
            raw_bindings = frame.bindings
            sentence_spans = frame.sentence_spans
            # Speech act from tensor (Violation 4 fix — no ? char check)
            is_question = frame.is_question
        else:
            # Backward-compatible: build frame from raw text
            frame = ArisLinguisticPipeline.process(str(text_or_frame))
            cleaned_text = frame.raw_utterance
            tokens = frame.tokens
            candidates = frame.candidates
            raw_bindings = frame.bindings
            sentence_spans = frame.sentence_spans
            is_question = frame.is_question

        # --- Shared state initialisation --------------------------------------

        resolved_coreferences: List[Dict[str, Any]] = []
        speaker_bindings: Dict[str, Any] = {}
        user_name = context.get_speaker_property("USER", "name")
        # Working salience: initialised from cross-turn context; updated per sentence
        working_salience: List[Dict[str, Any]] = list(context.get_entity_salience_ranking())

        # Build word token list and tracking structures
        words = [t for t in tokens if t.get("type") != "symbol"]
        resolved_word_map: Dict[int, str] = {}   # sequence_index → replacement text
        identity_token_indices: Set[int] = set()

        # Step 0: Pre-scan for speaker identity declarations
        # Guard: only perform identity binding for non-interrogative utterances.
        # Speech act comes from the tensor classifier via the frame — no ? char check.
        if not is_question:
            for binding in raw_bindings:
                subj = binding.get("subject", "")
                pred = binding.get("relation") or binding.get("predicate") or ""
                obj = binding.get("object", "")

                subj_str = subj.get("text") if isinstance(subj, dict) else str(subj)
                pred_str = (pred.get("text") if isinstance(pred, dict) else (pred.get("normalized") if isinstance(pred, dict) else str(pred))).lower()
                obj_str = (obj.get("text") if isinstance(obj, dict) else str(obj)).strip(" .,!?:;\"'")

                # Check each word in the subject span for deictic speaker role (handling fillers/denials like 'No My', 'Hey I')
                subj_words = [w.strip() for w in subj_str.split() if w.strip()]
                subj_role = None
                for sw in subj_words:
                    r = ArisDeicticLearner.classify_token_role(sw, threshold=0.90)
                    if r and r[0] in (ArisDeicticLearner.ROLE_SPEAKER_SUBJECT, ArisDeicticLearner.ROLE_SPEAKER_POSSESSIVE):
                        subj_role = r
                        break

                if subj_role:
                    # Generic Intelligence (Rule 1 & Rule 4): Competitive manifold projection (no scalar threshold)
                    pred_res = ArisPragmaticTensorEngine.classify_predicate_role(pred_str)
                    is_identity_predicate = (pred_res.get("dominant_role") == "identity_declaration")

                    if is_identity_predicate and obj_str and obj_str != "?" and not ArisDeicticLearner.is_speaker_role(obj_str):
                        context.bind_speaker_property("USER", "name", obj_str)
                        speaker_bindings["name"] = obj_str
                        user_name = obj_str
                        resolved_coreferences.append({
                            "referent": subj_str,
                            "resolved_entity": obj_str,
                            "role_id": subj_role[0],
                            "confidence": round(subj_role[1], 4),
                            "resolution_type": "speaker_identity_declaration",
                        })
                        # Protect all tokens in the identity declaration clause from premature replacement
                        if isinstance(subj, dict) and "token_indices" in subj:
                            identity_token_indices.update(subj["token_indices"])
                        if isinstance(pred, dict) and "token_indices" in pred:
                            identity_token_indices.update(pred["token_indices"])
                        if isinstance(obj, dict) and "token_indices" in obj:
                            identity_token_indices.update(obj["token_indices"])

        # -----------------------------------------------------------------------
        # Step 1: Process each sentence span incrementally (Violation 5 fix)
        # After resolving deictic/anaphoric tokens in sentence i, inject newly
        # discovered entity anchors from that sentence into working_salience so
        # that sentence i+1 can correctly resolve pro-forms referring to them.
        # -----------------------------------------------------------------------
        for span in sentence_spans:
            span_word_set = {t["sequence_index"] for t in span.tokens if t.get("type") != "symbol"}
            span_words = [t for t in words if t["sequence_index"] in span_word_set]

            for t in span_words:
                raw_val = t.get("raw", "")
                seq_idx = t.get("sequence_index", 0)
                if seq_idx in identity_token_indices:
                    continue

                role_match = ArisDeicticLearner.classify_token_role(raw_val, threshold=0.90)

                if not role_match:
                    continue

                role_id, confidence = role_match

                # CASE A: Speaker Possessive / Subject (e.g. 'my', 'I', 'naan', 'je')
                if role_id == ArisDeicticLearner.ROLE_SPEAKER_POSSESSIVE:
                    if user_name:
                        resolved_word_map[seq_idx] = f"{user_name}'s"
                        resolved_coreferences.append({
                            "referent": raw_val,
                            "resolved_entity": user_name,
                            "role_id": role_id,
                            "confidence": round(confidence, 4),
                            "resolution_type": "speaker_possessive_grounding",
                        })
                elif role_id == ArisDeicticLearner.ROLE_SPEAKER_SUBJECT:
                    if user_name:
                        resolved_word_map[seq_idx] = user_name
                        resolved_coreferences.append({
                            "referent": raw_val,
                            "resolved_entity": user_name,
                            "role_id": role_id,
                            "confidence": round(confidence, 4),
                            "resolution_type": "speaker_subject_grounding",
                        })

                # CASE B: Anaphoric Pro-Form (e.g. 'it', 'she', 'they', 'adhu')
                elif role_id in (ArisDeicticLearner.ROLE_ANAPHORA_INANIMATE, ArisDeicticLearner.ROLE_ANAPHORA_ANIMATE):
                    current_cand_texts = [c.get("text", "") for c in span.candidates if c.get("text")]
                    antecedent = cls._select_antecedent_from_salience(
                        role_id=role_id,
                        salience_ranking=working_salience,
                        user_name=user_name,
                        exclude_entities=current_cand_texts,
                    )
                    if antecedent:
                        resolved_word_map[seq_idx] = antecedent
                        resolved_coreferences.append({
                            "referent": raw_val,
                            "resolved_entity": antecedent,
                            "role_id": role_id,
                            "confidence": round(confidence, 4),
                            "resolution_type": "anaphora_salience_resolution",
                        })

            # After processing this sentence, inject its new entity anchors
            # into working_salience for the next sentence to resolve against.
            existing_entities = {e["entity"].strip().lower() for e in working_salience}
            for cand in span.candidates:
                cand_text = cand.get("text", "").strip()
                if cand_text and cand_text.lower() not in existing_entities:
                    # Structural role: use relational binding if available
                    cand_role = "object"
                    for b in span.bindings:
                        subj = b.get("subject", {})
                        subj_text = subj.get("text", "") if isinstance(subj, dict) else ""
                        if subj_text.lower() == cand_text.lower():
                            cand_role = "subject"
                            break
                    # Derive salience activation from the trained tensor substrate (Rule 1 & 4: Zero Hard Patches)
                    token_confs = [
                        t.get("upos_confidence")
                        for t in span.tokens
                        if t.get("sequence_index") in set(cand.get("token_indices", []))
                        and t.get("upos_confidence") is not None
                    ]
                    if token_confs:
                        learned_conf = sum(token_confs) / len(token_confs)
                    elif "confidence" in cand:
                        learned_conf = float(cand["confidence"])
                    else:
                        from ..linguistic.ud_tensor_engine import ArisUniversalLinguisticEncoder
                        _, learned_conf = ArisUniversalLinguisticEncoder.predict_upos_contextual(cand_text)

                    # Syntactic prominence weighting: subject argument has higher grammatical salience
                    role_weight = 1.0 if cand_role == "subject" else 0.85
                    salience_val = round(float(learned_conf) * role_weight, 4)

                    working_salience.insert(0, {
                        "entity": cand_text,
                        "salience_score": salience_val,
                        "structural_role": cand_role,
                        "mention_count": 1,
                    })
                    existing_entities.add(cand_text.lower())

        # 2. Reconstruct resolved text if any tokens were grounded
        resolved_tokens = []
        for t in tokens:
            seq_idx = t.get("sequence_index", 0)
            if seq_idx in resolved_word_map:
                resolved_tokens.append(resolved_word_map[seq_idx])
            else:
                resolved_tokens.append(t.get("raw", ""))

        resolved_text = " ".join(resolved_tokens)
        # Normalize punctuation spacing
        for punct in (",", ".", "?", "!", ":", ";"):
            resolved_text = resolved_text.replace(f" {punct}", punct)
        resolved_text = resolved_text.replace(" 's", "'s")

        return {
            "raw_text": cleaned_text,
            "resolved_text": resolved_text,
            "resolved_coreferences": resolved_coreferences,
            "speaker_bindings": speaker_bindings,
        }

    @classmethod
    def _select_antecedent_from_salience(
        cls,
        role_id: str,
        salience_ranking: List[Dict[str, Any]],
        user_name: Optional[str] = None,
        exclude_entities: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Selects the top-ranked candidate antecedent from the discourse salience ranking,
        enforcing structural co-argument disjointness and deictic role constraints (Centering Theory).
        """
        if not salience_ranking:
            return None

        candidates = list(salience_ranking)
        if user_name:
            # Universal Deictic Disjointness Invariant:
            # 3rd-person pro-forms ('it', 'she', 'he', 'they', 'its') cannot refer to the 1st-person speaker
            user_base = user_name.strip(" .,!?:;\"'").removesuffix("'s").removesuffix("’s").strip().lower()
            candidates = [
                c for c in candidates
                if user_base not in c.get("entity", "").strip(" .,!?:;\"'").lower()
            ]

        if exclude_entities:
            excluded_lowers = {e.strip(" .,!?:;\"'").removesuffix("'s").removesuffix("’s").strip().lower() for e in exclude_entities if e}
            candidates = [
                c for c in candidates
                if c.get("entity", "").strip(" .,!?:;\"'").removesuffix("'s").removesuffix("’s").strip().lower() not in excluded_lowers
            ]

        if not candidates:
            return None

        if role_id == ArisDeicticLearner.ROLE_ANAPHORA_INANIMATE:
            # Inanimate pro-form ("it", "adhu", "this") preferentially binds to Semantic Patient/Theme
            theme_cands = [c for c in candidates if c.get("structural_role") == "object"]
            if theme_cands:
                return theme_cands[0]["entity"]
            return candidates[0]["entity"]

        elif role_id == ArisDeicticLearner.ROLE_ANAPHORA_ANIMATE:
            # Animate 3rd-person pro-form ("she", "he", "they") binds to the most salient 3rd-person entity,
            # prioritizing the syntactic subject / agent over patient / theme (Centering Theory)
            subject_cands = [c for c in candidates if c.get("structural_role") == "subject"]
            if subject_cands:
                return subject_cands[0]["entity"]
            return candidates[0]["entity"]

        return candidates[0]["entity"]
