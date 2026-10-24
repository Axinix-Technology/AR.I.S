# pyright: ignore
"""
AR.I.S — Question to Semantic Query Representation & Binding Primitive

Transforms natural-language questions into structured semantic queries:
    Query(
        subject = entity | variable,
        relation = canonical_relation,
        object = entity | variable,
        answer_slot = subject | object | boolean
    )

Strict Architectural Invariants (Zero Hard Patches — Rule 1 & Rule 4):
  - NO hardcoded word lists (zero WH lists, zero auxiliary lists).
  - Sentence-initial single capitalized words are structurally ambiguous;
    in questions, they function as clause-initial operators, not entities.
  - Variable slot determination is purely structural and positional:
      - 2 Bound Entities -> Verification Query (answer_slot = "boolean")
      - 1 Bound Entity:
          * If content tokens follow Entity: Entity precedes main predicate ->
            Entity is Subject, Object is variable (?) (answer_slot = "object")
          * If no content tokens follow Entity: Entity follows main predicate ->
            Entity is Object, Subject is variable (?) (answer_slot = "subject")
  - Relation canonicalization operates in continuous coordinate space
    using ArisSemanticEncoder similarity against registered ontology relations.
  - Decoupled pipeline:
      INPUT -> TOKENIZATION -> ENTITY CANDIDATES -> QUESTION STRUCTURE ->
      SEMANTIC QUERY -> GRAPH SEARCH -> REASONING PATH -> ANSWER
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from ..linguistic.tokenizer import ArisTokenizer
from .semantic_representation import ArisSemanticEncoder
from ..linguistic.syntactic_normalizer import ArisSyntacticNormalizer
from ..cognitive.semantic_frame import (
    ArisSpeechActClassifier,
    ArisInterrogativeRoleLearner,
    SpeechActType,
)


@dataclass
class ArisQuery:
    subject: Optional[str]
    relation: str
    object: Optional[str]
    answer_slot: str  # "subject", "object", or "boolean"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "answer_slot": self.answer_slot,
        }


class ArisQueryBinder:
    """
    Cognitive Query Understanding and Semantic Binding Engine.
    Operates through pure structural positioning and continuous coordinate embeddings.
    """

    # Default registered ontology relations (gathered from Brain memory facts)
    DEFAULT_REGISTERED_RELATIONS = [
        "works_at",
        "located_in",
        "colleague_of",
        "manages",
        "uses",
        "supports",
        "orbits",
        "is_a",
        "brother_of",
    ]

    @classmethod
    def extract_entity_anchors(
        cls,
        text: str,
        tokens: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        is_question: bool,
        known_entities: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracts genuine entity anchors without dictionary blacklists.

        Structural Invariants:
          1. In a question, sentence-initial capitalized token (sequence_index 0)
             is an interrogative/modal clause opener, not an entity anchor.
          2. Proper noun candidates inside the sentence body (sequence_index > 0)
             are retained.
          3. Multi-word candidates spanning sequence 0 (e.g. 'Does Arun') have their
             sentence-initial operator decoupled from the internal entity.
          4. Known entities from the Knowledge Brain memory are matched.
        """
        anchors = []

        for c in candidates:
            token_indices = c.get("token_indices", [])
            cand_type = c.get("candidate_type", "")

            # If this is a question, sequence_index 0 is a clause-initial operator
            if is_question and 0 in token_indices:
                if len(token_indices) == 1 and cand_type == "ambiguous_capitalized":
                    # Sentence-initial single capitalized word is clause operator
                    continue
                elif len(token_indices) > 1:
                    # Multi-token started at sequence 0 (e.g. 'Does Arun')
                    # Decouple the initial token from subsequent entity tokens
                    sub_indices = [idx for idx in token_indices if idx != 0]
                    if sub_indices:
                        sub_tokens = [t for t in tokens if t["sequence_index"] in sub_indices]
                        start_c = sub_tokens[0]["position"]["char_start"]
                        end_c = sub_tokens[-1]["position"]["char_end"]
                        span_text = text[start_c:end_c]
                        anchors.append({
                            "text": span_text,
                            "normalized": span_text.lower(),
                            "candidate_type": "entity_candidate",
                            "char_start": start_c,
                            "char_end": end_c,
                            "token_indices": sub_indices,
                        })
                    continue

            anchors.append(c)

        # Clean possessives from candidate text (e.g., "Arun's" -> "Arun")
        cleaned_anchors = []
        for a in anchors:
            raw_text = a["text"]
            if raw_text.endswith("'s") or raw_text.endswith("’s"):
                clean_name = raw_text[:-2]
            else:
                clean_name = raw_text
            cleaned_anchors.append({
                **a,
                "text": clean_name,
                "normalized": clean_name.lower(),
            })

        # Grounding against known entity referents stored in Brain memory
        if known_entities:
            existing_texts = {a["normalized"] for a in cleaned_anchors}
            for ent in known_entities:
                ent_norm = ent.lower()
                if ent_norm not in existing_texts:
                    # Match across canonical token sequence
                    for t in tokens:
                        if t.get("raw", "").lower() == ent_norm:
                            cleaned_anchors.append({
                                "text": ent,
                                "normalized": ent_norm,
                                "candidate_type": "entity_candidate",
                                "char_start": t["position"]["char_start"],
                                "char_end": t["position"]["char_end"],
                                "token_indices": [t["sequence_index"]],
                            })
                            existing_texts.add(ent_norm)
                            break

        # Deduplicate anchors covering overlapping or identical entities
        deduped = []
        for a in cleaned_anchors:
            is_dup = False
            for d in deduped:
                if d["normalized"] == a["normalized"]:
                    is_dup = True
                    break
                # Span overlap check
                if max(d.get("char_start", 0), a.get("char_start", 0)) < min(d.get("char_end", 0), a.get("char_end", 0)):
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(a)

        deduped.sort(key=lambda x: x.get("char_start", 0))
        return deduped

    @classmethod
    def infer_interrogative_target_role(
        cls,
        tokens: List[Dict[str, Any]],
        is_question: bool,
    ) -> Tuple[Optional[str], float]:
        """
        Infers the target semantic role (?x, ?location, ?person) from the interrogative clause.
        Projects into continuous semantic role prototype space via ArisInterrogativeRoleLearner.
        """
        if not is_question or not tokens:
            return None, 0.0

        first_tok = tokens[0]["token"].lower()
        role, conf = ArisInterrogativeRoleLearner.infer_target_role(first_tok)
        if conf >= 0.80:
            return role, conf

        # Check token 1 if token 0 was a preposition (e.g. 'In which...')
        if len(tokens) > 1:
            second_tok = tokens[1]["token"].lower()
            role, conf = ArisInterrogativeRoleLearner.infer_target_role(second_tok)
            if conf >= 0.80:
                return role, conf

        return "entity", 0.50

    @classmethod
    def detect_question_operator(
        cls,
        tokens: List[Dict[str, Any]],
        is_question: bool,
    ) -> Optional[str]:
        """Backward compatibility helper for target role inference."""
        role, _ = cls.infer_interrogative_target_role(tokens, is_question)
        return role

    @classmethod
    def canonicalize_relation(
        cls,
        predicate_span: str,
        registered_relations: Optional[List[str]] = None,
        learned_mappings: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Maps the extracted predicate span into a canonical relation using
        continuous coordinate similarity (ArisSemanticEncoder).
        """
        relations = registered_relations or cls.DEFAULT_REGISTERED_RELATIONS
        if not predicate_span:
            return relations[0] if relations else ""

        cleaned_pred = predicate_span.strip().lower()

        # Check learned cognitive memory mappings if available
        if learned_mappings and cleaned_pred in learned_mappings:
            return learned_mappings[cleaned_pred]

        relations = registered_relations or cls.DEFAULT_REGISTERED_RELATIONS

        # Encode predicate span into continuous coordinate vector
        best_relation = relations[0]
        best_similarity = -1.0

        for rel in relations:
            rel_phrase = rel.replace("_", " ")

            # Subword cosine similarity in continuous coordinate space
            sim = ArisSemanticEncoder.similarity(cleaned_pred, rel_phrase)

            # Direct substring presence provides strong semantic alignment
            if rel_phrase in cleaned_pred or cleaned_pred in rel_phrase:
                sim = max(sim, 0.9)

            if sim > best_similarity:
                best_similarity = sim
                best_relation = rel

        return best_relation

    @classmethod
    def parse_question(
        cls,
        text: str,
        known_entities: Optional[List[str]] = None,
        registered_relations: Optional[List[str]] = None,
        learned_mappings: Optional[Dict[str, str]] = None,
        abstraction_engine: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Full Question Understanding & Semantic Query Generation Pipeline.

        Emits structured diagnostics matching the user specification:
        {
            "input": text,
            "query_understanding": {
                "entities": [...],
                "wh_type": ...,
                "subject": ...,
                "relation_surface": ...,
                "relation_canonical": ...,
                "object": ...,
                "answer_slot": ...
            },
            "semantic_query": {
                "subject": ...,
                "relation": ...,
                "object": ...,
                "answer_slot": ...
            },
            "query_object": ArisQuery(...)
        }
        """
        tokens = ArisTokenizer.tokenize(text)
        candidates = ArisTokenizer.extract_candidates(text, tokens)

        # Continuous speech act classification from neural tensor engine
        speech_act_info = ArisSpeechActClassifier.classify(text)
        is_question = (speech_act_info["dominant_act"] == SpeechActType.INTERROGATIVE)

        # 1. Entity Extraction without dictionary blacklists
        entities = cls.extract_entity_anchors(
            text=text,
            tokens=tokens,
            candidates=candidates,
            is_question=is_question,
            known_entities=known_entities,
        )

        # 2. Interrogative Target Role Inference via Neural Prototype Projections
        target_role, role_conf = cls.infer_interrogative_target_role(tokens, is_question)

        entity_names = [e["text"] for e in entities]

        # 3. Predicate Span Extraction via Positional Intervals
        predicate_span = ""
        if len(entities) >= 2:
            e1_end = entities[0]["char_end"]
            e2_start = entities[1]["char_start"]
            between_tokens = [
                t["raw"].lower() for t in tokens
                if t["position"]["char_start"] >= e1_end
                and t["position"]["char_end"] <= e2_start
                and t["type"] != "symbol"
            ]
            predicate_span = " ".join(between_tokens) if between_tokens else ""

        elif len(entities) == 1:
            ent = entities[0]
            ent_start = ent.get("char_start", 0)
            ent_end = ent.get("char_end", 0)

            tokens_before = [
                t for t in tokens
                if t["position"]["char_end"] <= ent_start
                and t["sequence_index"] > 0
                and t["type"] != "symbol"
            ]
            tokens_after = [
                t for t in tokens
                if t["position"]["char_start"] >= ent_end
                and t["type"] != "symbol"
                and t["raw"].lower() not in {"'s", "’s", "s"}
            ]

            if tokens_after:
                predicate_span = " ".join(t["raw"].lower() for t in tokens_after)
            elif tokens_before:
                predicate_span = " ".join(t["raw"].lower() for t in tokens_before)

        else:
            non_symbol = [t["raw"].lower() for t in tokens if t["sequence_index"] > 0 and t["type"] != "symbol"]
            predicate_span = " ".join(non_symbol)

        # 4. Canonical Relation Mapping & Abstraction via Continuous Coordinate Space
        candidate_relations = []
        confidence = 1.0

        if abstraction_engine:
            abs_res = abstraction_engine.abstract_relation(
                surface_phrase=predicate_span,
                question_text=text,
                entities=entity_names,
                target_role=target_role,
            )
            canonical_rel = abs_res["predicted_relation"]
            confidence = abs_res["confidence"]
            candidate_relations = abs_res["candidate_relations"]
        else:
            canonical_rel = cls.canonicalize_relation(
                predicate_span=predicate_span,
                registered_relations=registered_relations,
                learned_mappings=learned_mappings,
            )

        # 5. Syntactic Frame Normalization (Universal Dependencies — Zero Hard Patches)
        norm_res = ArisSyntacticNormalizer.normalize_roles(
            text=text,
            tokens=tokens,
            entities=entities,
            predicate_span=predicate_span,
            is_question=is_question,
            target_role=target_role,
        )

        subject = norm_res["subject"]
        obj = norm_res["object"]
        answer_slot = norm_res["answer_slot"]

        query_obj = ArisQuery(
            subject=subject,
            relation=canonical_rel,
            object=obj,
            answer_slot=answer_slot,
        )

        return {
            "input": text,
            "query_understanding": {
                "entities": entity_names,
                "target_role": target_role,
                "expected_role": target_role,
                "wh_type": target_role,
                "subject": subject if subject != "?" else None,
                "relation_surface": predicate_span,
                "relation_canonical": canonical_rel,
                "object": obj if obj != "?" else None,
                "answer_slot": answer_slot,
                "confidence": confidence,
                "candidate_relations": candidate_relations,
            },
            "semantic_query": {
                "subject": subject,
                "relation": canonical_rel,
                "object": obj,
                "answer_slot": answer_slot,
            },
            "query_object": query_obj,
        }
