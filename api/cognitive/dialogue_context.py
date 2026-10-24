"""
AR.I.S — Conversational Dialogue Context Manager (Phase 1)

Maintains stateful multi-turn conversational context windows, entity salience graphs,
and session history for AR.I.S dialogue interactions.

Key Invariants:
  - ZERO HARD PATCHES: Structural recency and salience weighting over entities.
  - ELASTIC CONTEXT BUFFER: Configurable sliding window (default 32 turns) with automatic compaction.
  - PERSISTENCE: Session-isolated conversational storage in MongoDB collection 'brain_conversations'.
"""

import time
import uuid
from typing import Dict, Any, List, Optional
from ..db import get_mongo_db


class ArisDialogueTurn:
    """
    Encapsulates a single conversational interaction turn.
    """

    def __init__(
        self,
        turn_id: str,
        turn_index: int,
        speaker_role: str,  # "USER" | "AGENT"
        raw_text: str,
        tokens: Optional[List[Dict[str, Any]]] = None,
        extracted_entities: Optional[List[Dict[str, Any]]] = None,
        relational_bindings: Optional[List[Dict[str, Any]]] = None,
        resolved_coreferences: Optional[List[Dict[str, Any]]] = None,
        epistemic_state: Optional[str] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.turn_id = turn_id or str(uuid.uuid4())
        self.turn_index = turn_index
        self.speaker_role = speaker_role.upper()
        self.raw_text = raw_text
        self.tokens = tokens or []
        self.extracted_entities = extracted_entities or []
        self.relational_bindings = relational_bindings or []
        self.resolved_coreferences = resolved_coreferences or []
        self.epistemic_state = epistemic_state or "UNKNOWN"
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "turn_index": self.turn_index,
            "speaker_role": self.speaker_role,
            "raw_text": self.raw_text,
            "tokens": self.tokens,
            "extracted_entities": self.extracted_entities,
            "relational_bindings": self.relational_bindings,
            "resolved_coreferences": self.resolved_coreferences,
            "epistemic_state": self.epistemic_state,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArisDialogueTurn":
        return cls(
            turn_id=data.get("turn_id", ""),
            turn_index=data.get("turn_index", 0),
            speaker_role=data.get("speaker_role", "USER"),
            raw_text=data.get("raw_text", ""),
            tokens=data.get("tokens", []),
            extracted_entities=data.get("extracted_entities", []),
            relational_bindings=data.get("relational_bindings", []),
            resolved_coreferences=data.get("resolved_coreferences", []),
            epistemic_state=data.get("epistemic_state", "UNKNOWN"),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


class ArisDialogueContext:
    """
    Maintains the state of an active conversation session, including sliding turns,
    speaker identities, and dynamic entity salience weights.
    """

    DEFAULT_MAX_TURNS = 32

    def __init__(
        self,
        session_id: str,
        user_id: str = "default_user",
        max_turns: int = DEFAULT_MAX_TURNS,
        turns: Optional[List[ArisDialogueTurn]] = None,
        speaker_identity: Optional[Dict[str, Any]] = None,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.user_id = user_id
        self.max_turns = max_turns
        self.turns: List[ArisDialogueTurn] = turns or []
        # Structural speaker registry (e.g. USER identity bindings)
        self.speaker_identity: Dict[str, Any] = speaker_identity or {
            "USER": {"name": None, "entities": []},
            "AGENT": {"name": "AR.I.S", "entities": []},
        }

    def add_turn(self, turn: ArisDialogueTurn) -> None:
        """Appends a turn and enforces sliding context window."""
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_turn_count(self) -> int:
        return len(self.turns)

    def get_recent_turns(self, n: int = 5) -> List[ArisDialogueTurn]:
        return self.turns[-n:]

    def bind_speaker_property(self, speaker: str, property_name: str, value: Any) -> None:
        """Binds a structural property to a speaker (e.g. USER name)."""
        speaker_key = speaker.upper()
        if speaker_key not in self.speaker_identity:
            self.speaker_identity[speaker_key] = {"name": None, "entities": []}
        self.speaker_identity[speaker_key][property_name] = value

    def get_speaker_property(self, speaker: str, property_name: str) -> Optional[Any]:
        speaker_key = speaker.upper()
        return self.speaker_identity.get(speaker_key, {}).get(property_name)

    def get_entity_salience_ranking(self, recency_decay: float = 0.85) -> List[Dict[str, Any]]:
        """
        Calculates dynamic entity salience across the active context window.
        Entities mentioned in recent turns receive higher salience.
        """
        salience_map: Dict[str, Dict[str, Any]] = {}
        total_turns = len(self.turns)

        from .deictic_learner import ArisDeicticLearner

        for idx, turn in enumerate(reversed(self.turns)):
            distance_from_now = idx  # 0 is the most recent turn
            recency_multiplier = recency_decay ** distance_from_now

            # Phase 8 — Phatic immunity:
            # Purely phatic or acknowledgement turns carry no semantic entity content.
            # Including their extracted_entities in the salience graph would pollute
            # coreference resolution with greeting words (e.g. "Hi", "buddy").
            # We skip the turn's entity contributions only if the epistemic_state is
            # exclusively PHATIC or THANKING AND the turn contains no relational bindings
            # (which would indicate meaningful declarative/identity content was also present).
            if turn.epistemic_state in ("PHATIC", "THANKING") and not turn.relational_bindings:
                continue

            # Entities explicitly extracted in this turn

            for ent in turn.extracted_entities:
                ent_name = ent.get("text") if isinstance(ent, dict) else str(ent)
                if not ent_name or ent_name == "?":
                    continue
                # Skip pro-forms and deictic markers via vector classifier
                if ArisDeicticLearner.is_speaker_role(ent_name) or ArisDeicticLearner.is_anaphoric_role(ent_name):
                    continue

                if ent_name not in salience_map:
                    salience_map[ent_name] = {
                        "entity": ent_name,
                        "salience_score": 0.0,
                        "last_seen_turn": turn.turn_index,
                        "speaker": turn.speaker_role,
                        "mention_count": 0,
                    }

                salience_map[ent_name]["salience_score"] += (1.0 * recency_multiplier)
                salience_map[ent_name]["mention_count"] += 1

            # Relational bindings in this turn
            for rel in turn.relational_bindings:
                subj = rel.get("subject")
                obj = rel.get("object")

                if subj:
                    subj_name = subj.get("text") if isinstance(subj, dict) else str(subj)
                    if subj_name and subj_name not in ("?", "USER", "AGENT"):
                        if not (ArisDeicticLearner.is_speaker_role(subj_name) or ArisDeicticLearner.is_anaphoric_role(subj_name)):
                            if subj_name not in salience_map:
                                salience_map[subj_name] = {
                                    "entity": subj_name,
                                    "salience_score": 0.0,
                                    "last_seen_turn": turn.turn_index,
                                    "speaker": turn.speaker_role,
                                    "mention_count": 0,
                                    "structural_role": "subject",
                                }
                            salience_map[subj_name]["salience_score"] += (1.2 * recency_multiplier)
                            salience_map[subj_name]["mention_count"] += 1
                            salience_map[subj_name]["structural_role"] = "subject"

                if obj:
                    obj_name = obj.get("text") if isinstance(obj, dict) else str(obj)
                    if obj_name and obj_name not in ("?", "USER", "AGENT"):
                        if not (ArisDeicticLearner.is_speaker_role(obj_name) or ArisDeicticLearner.is_anaphoric_role(obj_name)):
                            if obj_name not in salience_map:
                                salience_map[obj_name] = {
                                    "entity": obj_name,
                                    "salience_score": 0.0,
                                    "last_seen_turn": turn.turn_index,
                                    "speaker": turn.speaker_role,
                                    "mention_count": 0,
                                    "structural_role": "object",
                                }
                            salience_map[obj_name]["salience_score"] += (0.6 * recency_multiplier)
                            salience_map[obj_name]["mention_count"] += 1
                            salience_map[obj_name]["structural_role"] = "object"

        # Sort by computed salience descending
        ranked = sorted(salience_map.values(), key=lambda x: x["salience_score"], reverse=True)
        return ranked

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "max_turns": self.max_turns,
            "speaker_identity": self.speaker_identity,
            "turns": [turn.to_dict() for turn in self.turns],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArisDialogueContext":
        turns = [ArisDialogueTurn.from_dict(t) for t in data.get("turns", [])]
        return cls(
            session_id=data.get("session_id", str(uuid.uuid4())),
            user_id=data.get("user_id", "default_user"),
            max_turns=data.get("max_turns", cls.DEFAULT_MAX_TURNS),
            turns=turns,
            speaker_identity=data.get("speaker_identity"),
        )


class ArisDialogueContextManager:
    """
    Session persistence and context lifecycle manager.
    Stores and retrieves dialogue contexts from MongoDB 'brain_conversations'.
    """

    COLLECTION_NAME = "brain_conversations"

    # In-memory session cache for low-latency multi-turn interactions
    _active_contexts: Dict[str, ArisDialogueContext] = {}

    @classmethod
    def _get_collection(cls):
        db = get_mongo_db()
        return db[cls.COLLECTION_NAME]

    @classmethod
    def get_or_create_context(
        cls, session_id: Optional[str] = None, user_id: str = "default_user", max_turns: int = 32
    ) -> ArisDialogueContext:
        """Retrieves active session context from memory or MongoDB, or initializes a new one."""
        sid = session_id or str(uuid.uuid4())

        # Check in-memory active cache first
        if sid in cls._active_contexts:
            return cls._active_contexts[sid]

        # Check MongoDB collection
        try:
            coll = cls._get_collection()
            doc = coll.find_one({"session_id": sid})
            if doc:
                ctx = ArisDialogueContext.from_dict(doc)
                cls._active_contexts[sid] = ctx
                return ctx
        except Exception:
            pass

        # Create new context
        new_ctx = ArisDialogueContext(session_id=sid, user_id=user_id, max_turns=max_turns)
        cls._active_contexts[sid] = new_ctx
        cls.save_context(new_ctx)
        return new_ctx

    @classmethod
    def save_context(cls, context: ArisDialogueContext) -> None:
        """Persists context to MongoDB collection."""
        cls._active_contexts[context.session_id] = context
        try:
            coll = cls._get_collection()
            coll.update_one(
                {"session_id": context.session_id},
                {"$set": context.to_dict()},
                upsert=True
            )
        except Exception:
            # Tolerates memory-only mode if DB is in mock/test setup
            pass

    @classmethod
    def clear_context(cls, session_id: str) -> bool:
        """Clears an active session context."""
        cls._active_contexts.pop(session_id, None)
        try:
            coll = cls._get_collection()
            coll.delete_one({"session_id": session_id})
            return True
        except Exception:
            return False
