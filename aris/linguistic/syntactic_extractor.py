"""
AR.I.S — Linguistic Syntactic Feature Extractor
Purpose:
  Extracts syntactic governor relations, subject-verb-object (SVO) candidate triples,
  and Universal POS annotations from parsed UDSentence objects without hardcoded word rules.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from .ud_parser import UDSentence, UDToken


class ArisSyntacticExtractor:
    """
    Extracts grammatical argument structures and syntactic dependencies from UD trees.
    """

    @classmethod
    def extract_svo_triples(cls, sentence: UDSentence) -> List[Dict[str, Any]]:
        """
        Extracts structural Subject-Verb-Object triples from the dependency tree.
        Uses pure head-dependent relation tags (nsubj, root/verb, obj/obl).
        """
        triples = []
        token_by_id = {t.id: t for t in sentence.tokens}

        for token in sentence.tokens:
            # Look for verbal predicate heads
            if token.upos in ("VERB", "AUX"):
                verb_id = token.id
                verb_text = token.lemma or token.form

                subjects = [t for t in sentence.tokens if t.head == verb_id and "subj" in t.deprel]
                objects = [t for t in sentence.tokens if t.head == verb_id and ("obj" in t.deprel or "obl" in t.deprel)]

                for subj in subjects:
                    # Resolve full compound nominal span for subject (e.g. "Dr Martin Keller")
                    subj_tokens = sorted([t for t in sentence.tokens if t.head == subj.id and t.deprel == "compound"] + [subj], key=lambda x: x.id)
                    subj_text = " ".join(t.form for t in subj_tokens)

                    if objects:
                        for obj in objects:
                            obj_tokens = sorted([t for t in sentence.tokens if t.head == obj.id and t.deprel == "compound"] + [obj], key=lambda x: x.id)
                            obj_text = " ".join(t.form for t in obj_tokens)
                            triples.append({
                                "subject": subj_text,
                                "predicate": verb_text,
                                "object": obj_text,
                                "sentence": sentence.text,
                            })
                    else:
                        triples.append({
                            "subject": subj_text,
                            "predicate": verb_text,
                            "object": None,
                            "sentence": sentence.text,
                        })

        return triples

