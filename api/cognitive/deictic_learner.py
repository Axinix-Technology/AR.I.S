"""
AR.I.S — Generic Deictic & Pro-Form Cognitive Prototype Learner

Core Directives:
  - ZERO HARD PATCHES (Rule 1): Zero hardcoded English pronoun dictionaries or regex string matches.
  - GENERIC COGNITION (Rule 4): Discovers deictic participant roles (Speaker, Addressee, Pro-Forms)
    across any natural language (English, Tamil, French, Hindi) or synthetic alien vocabulary
    via continuous coordinate embedding prototypes and structural teaching demonstrations.
"""

from typing import Dict, Any, List, Optional, Tuple
from ..semantic.semantic_representation import ArisSemanticEncoder
from ..db import get_mongo_db


class ArisDeicticPrototype:
    """
    Abstract continuous prototype vector representing a communicative role
    via an ensemble of demonstrated exemplar vectors.
    """

    def __init__(
        self,
        role_id: str,
        name: str,
        centroid_vector: Optional[List[float]] = None,
        exemplar_vectors: Optional[List[List[float]]] = None,
        demonstration_count: int = 1,
    ):
        self.role_id = role_id
        self.name = name
        self.exemplar_vectors: List[List[float]] = exemplar_vectors or []
        if centroid_vector and not self.exemplar_vectors:
            self.exemplar_vectors.append(centroid_vector)
        self.demonstration_count = demonstration_count

    def similarity(self, token_vector: List[float]) -> float:
        """Computes maximum cosine similarity across all learned exemplar vectors."""
        if not self.exemplar_vectors or not token_vector:
            return 0.0
        max_sim = -1.0
        for ex in self.exemplar_vectors:
            dot = sum(a * b for a, b in zip(ex, token_vector))
            if dot > max_sim:
                max_sim = dot
        return max(-1.0, min(1.0, max_sim))

    def update_with_vector(self, new_vector: List[float]) -> None:
        """Adds a new exemplar vector to the communicative role."""
        self.exemplar_vectors.append(new_vector)
        self.demonstration_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role_id": self.role_id,
            "name": self.name,
            "exemplar_vectors": self.exemplar_vectors,
            "demonstration_count": self.demonstration_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArisDeicticPrototype":
        return cls(
            role_id=data.get("role_id", ""),
            name=data.get("name", ""),
            exemplar_vectors=data.get("exemplar_vectors"),
            centroid_vector=data.get("centroid_vector"),
            demonstration_count=data.get("demonstration_count", 1),
        )


class ArisDeicticLearner:
    """
    Learns and maintains continuous deictic and anaphoric prototype spaces.
    Contains ZERO hardcoded word dictionaries.
    """

    COLLECTION_NAME = "brain_deictic_prototypes"

    # In-memory prototype registry initialized via continuous embedding vectors
    _prototypes: Dict[str, ArisDeicticPrototype] = {}

    # Canonical deictic communicative roles
    ROLE_SPEAKER_SUBJECT = "role_speaker_subject"       # 1st-person subject (e.g., 'I', 'naan', 'je', 'main')
    ROLE_SPEAKER_POSSESSIVE = "role_speaker_possessive" # 1st-person genitive (e.g., 'my', 'ennudaiya', 'mon', 'mera')
    ROLE_ADDRESSEE = "role_addressee"                   # 2nd-person (e.g., 'you', 'your', 'nee', 'vous')
    ROLE_ANAPHORA_INANIMATE = "role_anaphora_inanimate" # Pro-form entity (e.g., 'it', 'adhu', 'cela')
    ROLE_ANAPHORA_ANIMATE = "role_anaphora_animate"     # Pro-form person (e.g., 'he', 'she', 'they', 'avan', 'aval')

    @classmethod
    def _get_collection(cls):
        db = get_mongo_db()
        return db[cls.COLLECTION_NAME]

    @classmethod
    def teach_marker(cls, marker: str, role_id: str, role_name: Optional[str] = None) -> ArisDeicticPrototype:
        """
        Teaches AR.I.S that a specific token/phrase marker embodies a deictic communicative role.
        Updates the continuous prototype centroid in vector space.
        """
        if not marker or not marker.strip():
            raise ValueError("Marker cannot be empty.")

        vec = ArisSemanticEncoder.encode(marker.strip())

        if role_id in cls._prototypes:
            proto = cls._prototypes[role_id]
            proto.update_with_vector(vec)
        else:
            proto = ArisDeicticPrototype(
                role_id=role_id,
                name=role_name or role_id,
                centroid_vector=vec,
                demonstration_count=1,
            )
            cls._prototypes[role_id] = proto

        # Persist to database if available
        try:
            coll = cls._get_collection()
            coll.update_one(
                {"role_id": role_id},
                {"$set": proto.to_dict()},
                upsert=True
            )
        except Exception:
            pass

        return proto

    @classmethod
    def classify_token_role(cls, token: Any, threshold: float = 0.90) -> Optional[Tuple[str, float]]:
        """
        Classifies a token against learned deictic prototypes in continuous vector space
        using in-memory PyTorch tensor matrices.
        """
        if not token:
            return None

        if isinstance(token, dict):
            token_str = token.get("text") or token.get("raw") or ""
        else:
            token_str = str(token)

        if not token_str.strip():
            return None

        from .pragmatic_tensor_engine import ArisPragmaticTensorEngine
        return ArisPragmaticTensorEngine.classify_deictic_token(token_str.strip(), threshold=threshold)

    @classmethod
    def is_speaker_role(cls, token: str, threshold: float = 0.90) -> bool:
        """Returns True if token projects onto speaker subject or possessive prototypes."""
        res = cls.classify_token_role(token, threshold=threshold)
        if not res:
            return False
        return res[0] in (cls.ROLE_SPEAKER_SUBJECT, cls.ROLE_SPEAKER_POSSESSIVE)

    @classmethod
    def is_anaphoric_role(cls, token: str, threshold: float = 0.90) -> bool:
        """Returns True if token projects onto anaphoric pro-form prototypes."""
        res = cls.classify_token_role(token, threshold=threshold)
        if not res:
            return False
        return res[0] in (cls.ROLE_ANAPHORA_INANIMATE, cls.ROLE_ANAPHORA_ANIMATE)

    _initialized = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Initializes foundational prototype space if empty via dynamic seed representations."""
        if cls._initialized:
            return

        cls._initialized = True

        import json
        import os

        # Load deictic role exemplars from cognitive pragmatic dataset
        dataset_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cognitive_pragmatic_training.json")
        if os.path.exists(dataset_path):
            try:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    deictic_map = data.get("deictic_roles", {})
                    for role_id, markers in deictic_map.items():
                        for marker in markers:
                            cls.teach_marker(marker, role_id, role_id)
            except Exception:
                pass

        # Attempt to load saved custom prototypes from MongoDB
        try:
            coll = cls._get_collection()
            docs = list(coll.find({}))
            if docs:
                for d in docs:
                    proto = ArisDeicticPrototype.from_dict(d)
                    if proto.role_id in cls._prototypes:
                        for ex in proto.exemplar_vectors:
                            cls._prototypes[proto.role_id].update_with_vector(ex)
                    else:
                        cls._prototypes[proto.role_id] = proto
        except Exception:
            pass
