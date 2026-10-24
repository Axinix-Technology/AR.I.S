"""
AR.I.S — Interrogative Syntactic Frame Normalizer

Applies domain-independent, structural-linguistic invariants to normalize
syntactic frames into canonical semantic roles (Subject, Object, Target Variable,
Answer Slot) without hardcoded question dictionaries, benchmark IDs, or exact strings.

Strict Non-Negotiable Directives:
  - ZERO benchmark-specific hardcoded strings or question patterns.
  - Sentence Mood Invariant:
      * Interrogative (?) vs Declarative (.)
      * Polar / Boolean queries require interrogative clause-opener + sentence termination with '?'
      * Declarative assertions assert a factual relation: answer_slot = 'object'
  - Universal Passive Voice Invariant:
      * The agentive preposition 'by' preceding an entity designates the semantic Agent (Subject),
        while the preceding entity designates the semantic Patient (Object).
  - Genitive Attribute Invariants:
      * Prepositional genitive: '... [attribute] of [Entity]?' -> Subject = Entity, Object = ?
      * Saxon genitive: '... [Entity]'s [attribute]?' -> Subject = Entity, Object = ?
  - Prepositional Fronting Invariant:
      * '[prep] which [noun] ... [Entity] ... [predicate]?' -> Subject = Entity, Object = ?
"""

from typing import Dict, Any, List, Optional, Tuple
from ..cognitive.semantic_frame import (
    ArisSpeechActClassifier,
    SpeechActType,
    CognitiveSemanticFrame,
    QueryFrame,
    ArisInterrogativeRoleLearner,
)
from ..cognitive.deictic_learner import ArisDeicticLearner


class ArisSyntacticNormalizer:
    """
    Structural-linguistic normalizer for syntactic frames, semantic argument roles,
    and language-independent CognitiveSemanticFrame IRs.
    """

    @classmethod
    def filter_non_entity_candidates(cls, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters out grammatical auxiliaries, deictic markers, and question operators
        from entity candidates using continuous coordinate prototypes.
        """
        filtered = []
        for e in entities:
            text = (e.get("text") if isinstance(e, dict) else str(e)).strip()
            if not text:
                continue
            # Check if token projects onto deictic or interrogative prototype
            if ArisDeicticLearner.is_speaker_role(text) or ArisDeicticLearner.is_anaphoric_role(text):
                continue
            role, conf = ArisInterrogativeRoleLearner.infer_target_role(text)
            if conf >= 0.88:
                continue
            filtered.append(e)
        return filtered

    @classmethod
    def normalize_roles(
        cls,
        text: str,
        tokens: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        predicate_span: str = "",
        is_question: bool = False,
        target_role: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Normalizes grammatical roles into canonical semantic arguments:
        (Subject, Object, Answer Slot) using pure Universal Dependencies structural invariants.
        """
        raw_text = text.strip()
        # Direct derivation from cognitive linguistic frame: no punctuation slicing guards (Rule 7)
        is_declarative = not is_question
        words_in_text = [t["token"] for t in tokens if t.get("upos") != "PUNCT"]

        # Clean entities by removing interrogative operators
        valid_entities = cls.filter_non_entity_candidates(entities)
        entities = valid_entities

        # UD Dependency tree detection for passive voice: nsubj:pass, aux:pass, obl:agent
        has_passive_by = any(
            "pass" in t.get("deprel", "") or t.get("deprel") == "obl:agent"
            for t in tokens
        )

        # UD Dependency tree detection for Saxon genitive: nmod:poss or possessive PART
        has_saxon_genitive = any(
            t.get("deprel") == "nmod:poss" or t.get("upos") == "PART"
            for t in tokens
        )

        # UD Dependency tree detection for prepositional genitive: nmod/case adpositions
        has_prepositional_genitive = any(
            t.get("deprel") in ("nmod", "case") and t.get("upos") == "ADP"
            for t in tokens
        )

        # UD Dependency tree detection for prepositional fronting: sentence-initial adposition
        has_prep_fronting = any(
            t.get("position", {}).get("token_in_sentence") == 0 and t.get("upos") == "ADP"
            for t in tokens
        )

        subject = None
        obj = None
        answer_slot = "object"
        role_frame = "standard"

        # CASE 1: 2 or more entities bound
        if len(entities) >= 2:
            e1 = entities[0]["text"]
            e2 = entities[1]["text"]

            if has_passive_by:
                # Universal Passive Voice Inversion: [Patient] [Verb] by [Agent]
                # E.g. 'Arun is managed by Ravi.' -> Agent = Ravi, Patient = Arun
                # E.g. 'Earth is orbited by Moon.' -> Agent = Moon, Patient = Earth
                # The agentive particle 'by' establishes e2 as semantic Agent (Subject)
                # and e1 as semantic Patient (Object).
                subject = e2
                obj = e1
                role_frame = "passive_inversion"
                answer_slot = "boolean" if is_question else "object"
            elif is_declarative:
                # Declarative assertion: [Subject] [Verb] [Object].
                # E.g. 'Arun works at Axinix.' -> Subject = Arun, Object = Axinix, answer_slot = 'object'
                subject = e1
                obj = e2
                answer_slot = "object"
                role_frame = "declarative_assertion"
            elif is_question:
                # Polar / Boolean question: 'Does Arun work at Axinix?' -> answer_slot = 'boolean'
                subject = e1
                obj = e2
                answer_slot = "boolean"
                role_frame = "polar_question"
            else:
                subject = e1
                obj = e2
                answer_slot = "object"
                role_frame = "standard"

        # CASE 2: Exactly 1 entity bound
        elif len(entities) == 1:
            ent = entities[0]
            ent_text = ent["text"]
            ent_start = ent.get("char_start", 0)
            ent_end = ent.get("char_end", 0)

            # Check tokens before and after the entity
            tokens_before = [
                t for t in tokens
                if t["position"]["char_end"] <= ent_start
                and t["sequence_index"] > 0
                and t["type"] != "symbol"
            ]
            tokens_after = [
                t for t in tokens
                if t["position"]["char_start"] >= ent_end
                and t.get("upos") not in ("PUNCT", "SYM", "PART")
            ]

            ent_indices = ent.get("token_indices", [])
            ent_tokens = [t for t in tokens if t.get("sequence_index") in ent_indices]
            ent_deprels = {t.get("deprel", "") for t in ent_tokens}

            if has_saxon_genitive or any("poss" in d for d in ent_deprels):
                # Saxon genitive: possessor anchor is subject
                subject = ent_text
                obj = "?"
                answer_slot = "object"
                role_frame = "saxon_genitive"

            elif has_prepositional_genitive or any("nmod" in d for d in ent_deprels):
                # Genitive modification: derived from UD tree head relations
                if any("nsubj" in t.get("deprel", "") for t in tokens if t.get("sequence_index") not in ent_indices):
                    subject = ent_text
                    obj = "?"
                    answer_slot = "object"
                    role_frame = "prepositional_genitive_object"
                else:
                    subject = "?"
                    obj = ent_text
                    answer_slot = "subject"
                    role_frame = "prepositional_genitive_subject"

            elif any("nsubj" in d for d in ent_deprels):
                # Entity is nominal subject in UD dependency tree: queries object/target
                subject = ent_text
                obj = "?"
                answer_slot = "object"
                role_frame = "ud_subject_query"

            elif any(d in ("obj", "obl", "iobj") for d in ent_deprels):
                # Entity is object/oblique in UD dependency tree: queries subject/agent
                subject = "?"
                obj = ent_text
                answer_slot = "subject"
                role_frame = "ud_object_query"

            elif tokens_after:
                # Entity precedes main predicate in sequence: Subject role
                subject = ent_text
                obj = "?"
                answer_slot = "object"
                role_frame = "entity_precedes_predicate"

            else:
                # Entity follows main predicate in sequence: Object role
                subject = "?"
                obj = ent_text
                answer_slot = "subject"
                role_frame = "entity_follows_predicate"

        # CASE 3: 0 entities bound
        else:
            subject = "?"
            obj = "?"
            answer_slot = "object"
            role_frame = "unbound"

        return {
            "subject": subject,
            "object": obj,
            "answer_slot": answer_slot,
            "role_frame": role_frame,
            "is_declarative": is_declarative,
        }

    @classmethod
    def construct_semantic_frame(
        cls,
        text: str,
        tokens: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        bindings: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Any] = None,
    ) -> CognitiveSemanticFrame:
        """
        Synthesizes a domain-agnostic CognitiveSemanticFrame IR from tokens,
        clean entity anchors, relational bindings, and continuous speech act scores.
        """
        speech_act_info = ArisSpeechActClassifier.classify(text)
        act = speech_act_info["dominant_act"]
        scores = speech_act_info["scores"]

        valid_entities = cls.filter_non_entity_candidates(entities)
        anchor_names = [
            (e.get("text") if isinstance(e, dict) else str(e)).strip()
            for e in valid_entities
            if (e.get("text") if isinstance(e, dict) else str(e)).strip()
        ]

        query_frame = None
        propositions = []

        if act == SpeechActType.INTERROGATIVE:
            norm = cls.normalize_roles(
                text=text,
                tokens=tokens,
                entities=valid_entities,
                is_question=True
            )
            # Infer target variable role dynamically via continuous vector prototype projection
            words = [t["raw"] for t in tokens if t.get("type") != "symbol"]
            first_operator = words[0] if words else ""
            var_role, _ = ArisInterrogativeRoleLearner.infer_target_role(first_operator)

            query_frame = QueryFrame(
                variable=f"?{var_role}",
                anchors=anchor_names,
                constraints=[{
                    "subject": norm.get("subject"),
                    "relation": norm.get("role_frame"),
                    "object": norm.get("object"),
                    "answer_slot": norm.get("answer_slot"),
                }],
                expected_role=var_role,
                raw_query=text,
            )

        elif act == SpeechActType.DECLARATIVE:
            if bindings:
                for b in bindings:
                    s = b.get("subject", {}).get("text") if isinstance(b.get("subject"), dict) else str(b.get("subject", ""))
                    p = b.get("predicate", {}).get("text") if isinstance(b.get("predicate"), dict) else str(b.get("predicate", ""))
                    o = b.get("object", {}).get("text") if isinstance(b.get("object"), dict) else str(b.get("object", ""))
                    if s and p and o and s != "?" and o != "?":
                        propositions.append({"subject": s, "relation": p, "object": o})

        return CognitiveSemanticFrame(
            raw_utterance=text,
            speech_act=act,
            speech_act_scores=scores,
            propositions=propositions,
            anchors=anchor_names,
            query_frame=query_frame,
            anaphora_bindings=[],
            discourse_updates=[{"entity": a, "salience": 1.0} for a in anchor_names],
        )

