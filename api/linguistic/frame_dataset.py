"""
AR.I.S — Generic De-Lexicalized Frame Dataset & Vectorizer

Non-Negotiable Architecture Invariants:
  - ZERO ARITHMETIC IN TRAINING: Training dataset contains ZERO numbers,
    zero arithmetic operations (+, -, *, /), zero arithmetic expressions.
  - Domain-Independent Structural Parameter Relationships:
    Learns abstract frames:
      parameter_1 = bound, parameter_2 = variable
      parameter_1 = variable, parameter_2 = bound
  - Randomizes symbols, alien relations, casing, lengths, variable markers (?, _, Q, Z, Who, etc.).
  - Pure structural perception: The network receives NO semantic knowledge of words.
"""

import random
from typing import List, Dict, Any, Tuple


class ArisFrameVectorizer:
    """
    Vectorizes structural frames (Subject, Relation, Object) into pure numerical tensors.
    Contains zero lexical identity mappings or word dictionaries.
    """

    FEATURE_DIM = 14

    @classmethod
    def vectorize_frame(
        cls,
        subject: str,
        relation: str,
        obj: str,
        subj_token_idx: int = 0,
        obj_token_idx: int = 2,
        total_tokens: int = 3,
    ) -> List[float]:
        """
        Extracts 14 structural invariants from the parameter frame:
          [0]: Subject relative position in sentence (0.0 - 1.0)
          [1]: Subject token index normalized
          [2]: Subject length normalized (min(len/15, 1.0))
          [3]: Subject is uppercase (1.0 / 0.0)
          [4]: Subject is titlecase (1.0 / 0.0)
          [5]: Subject is variable placeholder symbol (?, _, $, #) (1.0 / 0.0)
          [6]: Subject is single character (1.0 / 0.0)
          [7]: Object relative position in sentence (0.0 - 1.0)
          [8]: Object token index normalized
          [9]: Object length normalized (min(len/15, 1.0))
          [10]: Object is uppercase (1.0 / 0.0)
          [11]: Object is titlecase (1.0 / 0.0)
          [12]: Object is variable placeholder symbol (?, _, $, #) (1.0 / 0.0)
          [13]: Object is single character (1.0 / 0.0)
        """
        s_clean = str(subject).strip()
        o_clean = str(obj).strip()
        tot = max(total_tokens, 1)

        s_sym = 1.0 if any(c in "?_-$#@" for c in s_clean) else 0.0
        o_sym = 1.0 if any(c in "?_-$#@" for c in o_clean) else 0.0

        return [
            float(subj_token_idx) / tot,
            min(float(subj_token_idx) / 10.0, 1.0),
            min(len(s_clean) / 15.0, 1.0),
            1.0 if s_clean.isupper() else 0.0,
            1.0 if s_clean.istitle() else 0.0,
            s_sym,
            1.0 if len(s_clean) == 1 else 0.0,
            float(obj_token_idx) / tot,
            min(float(obj_token_idx) / 10.0, 1.0),
            min(len(o_clean) / 15.0, 1.0),
            1.0 if o_clean.isupper() else 0.0,
            1.0 if o_clean.istitle() else 0.0,
            o_sym,
            1.0 if len(o_clean) == 1 else 0.0,
        ]


class ArisStructuralCurriculumGenerator:
    """
    Generates domain-independent synthetic structural curricula.
    STRICT RULE: Contains ZERO arithmetic, zero numbers, zero math operators.
    """

    # Pure alien / arbitrary structural entities (ZERO arithmetic)
    ENTITIES = [
        "A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "M", "N",
        "Veltra", "Zeta", "Solar", "Protocols", "Alpha", "Beta", "Gamma",
        "Orion", "Lyra", "Cygnus", "Vesper", "Apex", "Nexus", "Vertex",
    ]

    # Pure alien / arbitrary relational verbs (ZERO arithmetic)
    RELATIONS = [
        "daxels", "mivens", "qevron", "vortax", "plonks", "zephyrs", "frelx",
        "klystron", "norvax", "glides over", "anchors to", "links with",
        "resonates with", "binds to", "transmits to",
    ]

    # Structural variable placeholder representations
    VARIABLE_MARKERS = [
        "?", "_", "Who", "who", "What", "what", "Which", "which",
        "Q", "Z", "X", "var", "target", "unknown",
    ]

    @classmethod
    def generate_curriculum(cls, num_samples: int = 2000, seed: int = 42) -> List[Dict[str, Any]]:
        """
        Generates balanced synthetic structural parameter frames.
        Label 0: Subject is Variable.
        Label 1: Object is Variable.
        """
        rng = random.Random(seed)
        dataset = []

        for _ in range(num_samples // 2):
            # Case 1: Subject is Variable (Reverse Traversal Frame: ? --R--> Object)
            var_token = rng.choice(cls.VARIABLE_MARKERS)
            bound_entity = rng.choice(cls.ENTITIES)
            rel = rng.choice(cls.RELATIONS)

            features = ArisFrameVectorizer.vectorize_frame(
                subject=var_token,
                relation=rel,
                obj=bound_entity,
                subj_token_idx=0,
                obj_token_idx=2,
                total_tokens=3,
            )
            dataset.append({
                "features": features,
                "label": 0,  # 0 = subject is variable
                "subject": var_token,
                "relation": rel,
                "object": bound_entity,
            })

            # Case 2: Object is Variable (Forward Traversal Frame: Subject --R--> ?)
            bound_entity2 = rng.choice(cls.ENTITIES)
            var_token2 = rng.choice(cls.VARIABLE_MARKERS)
            rel2 = rng.choice(cls.RELATIONS)

            features2 = ArisFrameVectorizer.vectorize_frame(
                subject=bound_entity2,
                relation=rel2,
                obj=var_token2,
                subj_token_idx=0,
                obj_token_idx=2,
                total_tokens=3,
            )
            dataset.append({
                "features": features2,
                "label": 1,  # 1 = object is variable
                "subject": bound_entity2,
                "relation": rel2,
                "object": var_token2,
            })

        rng.shuffle(dataset)
        return dataset
