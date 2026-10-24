"""
AR.I.S — Relational Belief & Human Feedback Manager (Pillar 4 Foundation)

Manages dynamic belief revision, human feedback loops (confirmations and denials),
zero-confidence inhibition matrices, and learned reciprocal/symmetric rules in MongoDB.

Key Architectural Invariants:
  - Zero Hard Patches: Rejection sets confidence to 0.0 and permanently inhibits invalid hypotheses.
  - Symmetrical & Reciprocal Induction: Discovers whether relations are symmetric (R <-> R)
    or reciprocal (R1 <-> R2) dynamically from user feedback.
  - Domain-Agnostic: Treats all relations and entities uniformly without predefined lists.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import re


class ArisRelationalBeliefManager:
    """
    Manages relational hypotheses, user feedback inhibition (confidence=0.0),
    and reciprocal/symmetric relationship rules in MongoDB.
    """

    BELIEFS_COLLECTION = "brain_relational_beliefs"
    RULES_COLLECTION = "brain_relational_rules"

    # In-memory stores for testing / offline isolation
    _in_memory_beliefs: Dict[str, Dict[str, Any]] = {}
    _in_memory_rules: Dict[str, List[str]] = {}
    _in_memory_symmetric: set = set()

    @classmethod
    def _get_db(cls):
        try:
            from ..db import get_mongo_db
            return get_mongo_db()
        except Exception:
            return None

    @classmethod
    def _normalize_rel(cls, rel: str) -> str:
        r = (rel or "").strip().lower()
        for prefix in ("was ", "is ", "are ", "were "):
            if r.startswith(prefix):
                return r[len(prefix):].strip()
        return r

    @classmethod
    def _make_key(cls, subject: str, relation: str, obj: str) -> str:
        s = (subject or "").strip().lower()
        r = cls._normalize_rel(relation)
        o = (obj or "").strip().lower()
        return f"{s}||{r}||{o}"

    @classmethod
    def record_denial(
        cls,
        subject: str,
        relation: str,
        obj: str,
        reason: str = "user_feedback",
        in_memory_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Records a negative constraint (denial) setting confidence to 0.0.
        Permanently inhibits this hypothesis from being suggested.
        """
        s_clean = subject.strip()
        r_clean = relation.strip()
        o_clean = obj.strip()
        now_iso = datetime.now(timezone.utc).isoformat()
        key = cls._make_key(s_clean, r_clean, o_clean)

        doc = {
            "key": key,
            "subject": s_clean,
            "relation": r_clean,
            "object": o_clean,
            "status": "denied",
            "confidence": 0.0,
            "reason": reason,
            "updated_at": now_iso,
        }

        cls._in_memory_beliefs[key] = doc

        if not in_memory_only:
            db = cls._get_db()
            if db is not None:
                try:
                    coll = db[cls.BELIEFS_COLLECTION]
                    coll.update_one({"key": key}, {"$set": doc}, upsert=True)
                except Exception:
                    pass

        return doc

    @classmethod
    def record_confirmation(
        cls,
        subject: str,
        relation: str,
        obj: str,
        in_memory_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Records an explicit confirmation from user feedback setting confidence to 1.0.
        """
        s_clean = subject.strip()
        r_clean = relation.strip()
        o_clean = obj.strip()
        now_iso = datetime.now(timezone.utc).isoformat()
        key = cls._make_key(s_clean, r_clean, o_clean)

        doc = {
            "key": key,
            "subject": s_clean,
            "relation": r_clean,
            "object": o_clean,
            "status": "confirmed",
            "confidence": 1.0,
            "reason": "user_confirmation",
            "updated_at": now_iso,
        }

        cls._in_memory_beliefs[key] = doc

        if not in_memory_only:
            db = cls._get_db()
            if db is not None:
                try:
                    coll = db[cls.BELIEFS_COLLECTION]
                    coll.update_one({"key": key}, {"$set": doc}, upsert=True)
                except Exception:
                    pass

        return doc

    @classmethod
    def is_denied(cls, subject: str, relation: str, obj: str) -> bool:
        """
        Checks whether this relation assertion has been explicitly refuted by the user.
        Fast O(1) in-memory check falling back to MongoDB.
        """
        key = cls._make_key(subject, relation, obj)

        if key in cls._in_memory_beliefs:
            return cls._in_memory_beliefs[key].get("confidence", 1.0) == 0.0

        db = cls._get_db()
        if db is not None:
            try:
                coll = db[cls.BELIEFS_COLLECTION]
                doc = coll.find_one({"key": key})
                if doc:
                    cls._in_memory_beliefs[key] = doc
                    return doc.get("confidence", 1.0) == 0.0
            except Exception:
                pass

        return False

    @classmethod
    def is_connection_denied(cls, subject: str, obj: str) -> bool:
        """
        Checks whether the connection between subject and obj has been globally inhibited (confidence=0.0).
        Matches if relation was generic (e.g. 'connected', 'connected_to', 'any', 'inhibited_connection').
        """
        s_clean = (subject or "").strip().lower()
        o_clean = (obj or "").strip().lower()
        generic_rels = {"connected", "connected_to", "any", "inhibited_connection"}

        for b in cls._in_memory_beliefs.values():
            if (b.get("subject", "").lower() == s_clean and b.get("object", "").lower() == o_clean) or (
                b.get("subject", "").lower() == o_clean and b.get("object", "").lower() == s_clean
            ):
                if b.get("confidence", 1.0) == 0.0 and b.get("relation", "").lower() in generic_rels:
                    return True

        db = cls._get_db()
        if db is not None:
            try:
                coll = db[cls.BELIEFS_COLLECTION]
                rgx_s = f"^{re.escape(s_clean)}$"
                rgx_o = f"^{re.escape(o_clean)}$"
                doc = coll.find_one({
                    "$or": [
                        {"subject": {"$regex": rgx_s, "$options": "i"}, "object": {"$regex": rgx_o, "$options": "i"}},
                        {"subject": {"$regex": rgx_o, "$options": "i"}, "object": {"$regex": rgx_s, "$options": "i"}},
                    ],
                    "confidence": 0.0,
                    "relation": {"$in": list(generic_rels)},
                })
                if doc:
                    return True
            except Exception:
                pass
        return False

    @classmethod
    def is_pair_denied(cls, subject: str, obj: str) -> bool:
        """
        Checks whether ANY relationship between subject and obj has been inhibited (confidence=0.0).
        """
        s_clean = (subject or "").strip().lower()
        o_clean = (obj or "").strip().lower()

        # Check in-memory beliefs
        for b in cls._in_memory_beliefs.values():
            if (b.get("subject", "").lower() == s_clean and b.get("object", "").lower() == o_clean) or (
                b.get("subject", "").lower() == o_clean and b.get("object", "").lower() == s_clean
            ):
                if b.get("confidence", 1.0) == 0.0:
                    return True

        db = cls._get_db()
        if db is not None:
            try:
                coll = db[cls.BELIEFS_COLLECTION]
                # Check normalized key prefix or case-insensitive regex
                rgx_s = f"^{re.escape(s_clean)}$"
                rgx_o = f"^{re.escape(o_clean)}$"
                doc = coll.find_one({
                    "$or": [
                        {"subject": {"$regex": rgx_s, "$options": "i"}, "object": {"$regex": rgx_o, "$options": "i"}, "confidence": 0.0},
                        {"subject": {"$regex": rgx_o, "$options": "i"}, "object": {"$regex": rgx_s, "$options": "i"}, "confidence": 0.0},
                        {"key": {"$regex": f"^{re.escape(s_clean)}\\|\\|.*\\|\\|{re.escape(o_clean)}$"}, "confidence": 0.0},
                        {"key": {"$regex": f"^{re.escape(o_clean)}\\|\\|.*\\|\\|{re.escape(s_clean)}$"}, "confidence": 0.0},
                    ]
                })
                if doc:
                    return True
            except Exception:
                pass

        return False

    @classmethod
    def get_belief(cls, subject: str, relation: str, obj: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the belief state for a specific triple if recorded.
        """
        key = cls._make_key(subject, relation, obj)
        if key in cls._in_memory_beliefs:
            return cls._in_memory_beliefs[key]

        db = cls._get_db()
        if db is not None:
            try:
                coll = db[cls.BELIEFS_COLLECTION]
                doc = coll.find_one({"key": key})
                if doc:
                    cls._in_memory_beliefs[key] = doc
                    return doc
            except Exception:
                pass

        return None

    @classmethod
    def learn_reciprocal_rule(cls, r1: str, r2: str, in_memory_only: bool = False):
        """
        Records an abstract reciprocal relation pair: (r1 <-> r2).
        e.g. 'is brother of' <-> 'is sister of' or 'parent of' <-> 'child of'.
        """
        r1_norm = r1.strip().lower()
        r2_norm = r2.strip().lower()

        if r1_norm not in cls._in_memory_rules:
            cls._in_memory_rules[r1_norm] = []
        if r2_norm not in cls._in_memory_rules[r1_norm]:
            cls._in_memory_rules[r1_norm].append(r2_norm)

        if r2_norm not in cls._in_memory_rules:
            cls._in_memory_rules[r2_norm] = []
        if r1_norm not in cls._in_memory_rules[r2_norm]:
            cls._in_memory_rules[r2_norm].append(r1_norm)

        if not in_memory_only:
            db = cls._get_db()
            if db is not None:
                try:
                    coll = db[cls.RULES_COLLECTION]
                    coll.update_one(
                        {"relation": r1_norm},
                        {"$addToSet": {"reciprocals": r2_norm}, "$set": {"rule_type": "reciprocal"}},
                        upsert=True,
                    )
                    coll.update_one(
                        {"relation": r2_norm},
                        {"$addToSet": {"reciprocals": r1_norm}, "$set": {"rule_type": "reciprocal"}},
                        upsert=True,
                    )
                except Exception:
                    pass

    @classmethod
    def get_reciprocal_relations(cls, relation: str) -> List[str]:
        """Returns known reciprocal relations for the given relation."""
        r_norm = relation.strip().lower()
        recs = cls._in_memory_rules.get(r_norm, [])
        if recs:
            return recs

        db = cls._get_db()
        if db is not None:
            try:
                coll = db[cls.RULES_COLLECTION]
                doc = coll.find_one({"relation": r_norm})
                if doc and "reciprocals" in doc:
                    cls._in_memory_rules[r_norm] = doc["reciprocals"]
                    return doc["reciprocals"]
            except Exception:
                pass

        return []

    @classmethod
    def learn_symmetric_rule(cls, relation: str, in_memory_only: bool = False):
        """
        Records that a relation exhibits symmetry: A --R--> B implies B --R--> A.
        """
        r_norm = relation.strip().lower()
        cls._in_memory_symmetric.add(r_norm)

        if not in_memory_only:
            db = cls._get_db()
            if db is not None:
                try:
                    coll = db[cls.RULES_COLLECTION]
                    coll.update_one(
                        {"relation": r_norm},
                        {"$set": {"is_symmetric": True, "rule_type": "symmetric"}},
                        upsert=True,
                    )
                except Exception:
                    pass

    @classmethod
    def is_symmetric(cls, relation: str) -> bool:
        """Returns True if the relation is registered as symmetric."""
        r_norm = relation.strip().lower()
        if r_norm in cls._in_memory_symmetric:
            return True

        db = cls._get_db()
        if db is not None:
            try:
                coll = db[cls.RULES_COLLECTION]
                doc = coll.find_one({"relation": r_norm, "is_symmetric": True})
                if doc:
                    cls._in_memory_symmetric.add(r_norm)
                    return True
            except Exception:
                pass

        return False

    @classmethod
    def clear_in_memory_cache(cls):
        """Resets the in-memory cache for test isolation."""
        cls._in_memory_beliefs.clear()
        cls._in_memory_rules.clear()
        cls._in_memory_symmetric.clear()
