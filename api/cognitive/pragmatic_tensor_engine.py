# pyright: ignore
"""
AR.I.S — PyTorch Pragmatic & Deictic Metric Tensor Engine

Computes maximum continuous cosine similarity projections over pre-compiled
PyTorch exemplar tensor matrices in GPU/CPU memory.

Constitutional Invariants:
  1. ZERO HARD PATCHES (Rule 1): Zero string keyword checks or language-specific if-clauses.
  2. PURE METRIC LEARNING: Evaluates continuous coordinate vector projections over learned exemplar matrices.
  3. ZERO RUNTIME DISK I/O: Loads pre-compiled tensor bank into active PyTorch memory at startup.
"""

import os
import json
import torch
from typing import Dict, Any, List, Optional, Tuple
from ..semantic.semantic_representation import ArisSemanticEncoder


class ArisPragmaticTensorEngine:
    """
    In-memory PyTorch Tensor engine for continuous pragmatic and deictic inferences.
    """

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "aris_pragmatic_tensor_bank.pt")

    _speech_matrices: Dict[str, torch.Tensor] = {}
    _deictic_matrices: Dict[str, torch.Tensor] = {}
    _interrogative_matrices: Dict[str, torch.Tensor] = {}
    _predicate_matrices: Dict[str, torch.Tensor] = {}

    _initialized = False

    @classmethod
    def train_and_compile(cls, dataset_path: Optional[str] = None, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Compiles raw JSON pragmatic training exemplars into normalized PyTorch tensor matrices.
        """
        data_file = dataset_path or os.path.join(os.path.dirname(__file__), "..", "..", "data", "cognitive_pragmatic_training.json")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Training dataset not found: {data_file}")

        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        speech_matrices = {}
        for act, phrases in data.get("speech_acts", {}).items():
            vecs = torch.tensor([ArisSemanticEncoder.encode(p) for p in phrases], dtype=torch.float32)
            vecs = vecs / (torch.norm(vecs, dim=-1, keepdim=True) + 1e-8)
            speech_matrices[act] = vecs

        deictic_matrices = {}
        for role, markers in data.get("deictic_roles", {}).items():
            vecs = torch.tensor([ArisSemanticEncoder.encode(m) for m in markers], dtype=torch.float32)
            vecs = vecs / (torch.norm(vecs, dim=-1, keepdim=True) + 1e-8)
            deictic_matrices[role] = vecs

        interrogative_matrices = {}
        for role, words in data.get("interrogative_roles", {}).items():
            vecs = torch.tensor([ArisSemanticEncoder.encode(w) for w in words], dtype=torch.float32)
            vecs = vecs / (torch.norm(vecs, dim=-1, keepdim=True) + 1e-8)
            interrogative_matrices[role] = vecs

        predicate_matrices = {}
        for role, phrases in data.get("predicate_roles", {}).items():
            vecs = torch.tensor([ArisSemanticEncoder.encode(p) for p in phrases], dtype=torch.float32)
            vecs = vecs / (torch.norm(vecs, dim=-1, keepdim=True) + 1e-8)
            predicate_matrices[role] = vecs

        target_file = save_path or cls.MODEL_PATH
        os.makedirs(os.path.dirname(target_file), exist_ok=True)

        payload = {
            "speech_matrices": speech_matrices,
            "deictic_matrices": deictic_matrices,
            "interrogative_matrices": interrogative_matrices,
            "predicate_matrices": predicate_matrices,
        }
        torch.save(payload, target_file)

        cls._speech_matrices = speech_matrices
        cls._deictic_matrices = deictic_matrices
        cls._interrogative_matrices = interrogative_matrices
        cls._predicate_matrices = predicate_matrices
        cls._initialized = True

        return {
            "status": "compiled_tensor_bank",
            "saved_to": target_file,
            "speech_acts": list(speech_matrices.keys()),
            "deictic_roles": list(deictic_matrices.keys()),
            "interrogative_roles": list(interrogative_matrices.keys()),
            "predicate_roles": list(predicate_matrices.keys()),
        }

    @classmethod
    def _ensure_loaded(cls, force_reload: bool = False) -> None:
        if cls._initialized and cls._speech_matrices and cls._predicate_matrices and not force_reload:
            return

        if os.path.exists(cls.MODEL_PATH):
            payload = torch.load(cls.MODEL_PATH, weights_only=False)
            cls._speech_matrices = payload.get("speech_matrices", {})
            cls._deictic_matrices = payload.get("deictic_matrices", {})
            cls._interrogative_matrices = payload.get("interrogative_matrices", {})
            cls._predicate_matrices = payload.get("predicate_matrices", {})
            if not cls._predicate_matrices:
                cls.train_and_compile()
                return
            cls._initialized = True
        else:
            cls.train_and_compile()


    @classmethod
    def _pool_sequence_vector(cls, sequence: Any, hidden_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extracts a normalized continuous representation vector in R^64 from sequence and/or hidden_states.
        """
        if hidden_states is not None:
            hs = hidden_states[0] if hidden_states.dim() == 3 else hidden_states
            # Mean pool over sequence length
            pooled = hs.mean(dim=0)
            norm = torch.norm(pooled)
            if norm > 1e-8:
                return pooled / norm
            return pooled

        # Fallback to tokens from sequence
        tokens = getattr(sequence, "tokens", [])
        content_tokens = [t for t in tokens if t not in {"[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"}]
        phrase = " ".join(content_tokens) if content_tokens else " ".join(tokens)
        vec = torch.tensor(ArisSemanticEncoder.encode(phrase), dtype=torch.float32)
        return vec / (torch.norm(vec) + 1e-8)

    @classmethod
    def classify_predicate_role_sequence(
        cls,
        sequence: Any,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Classifies the semantic communicative role directly from a canonical TokenizedSequence.
        """
        cls._ensure_loaded()
        if not sequence or not getattr(sequence, "tokens", None):
            return {"dominant_role": None, "confidence": 0.0, "margin": 0.0, "scores": {}}

        norm_vec = cls._pool_sequence_vector(sequence, hidden_states)

        scores: Dict[str, float] = {}
        for role, mat in cls._predicate_matrices.items():
            sims = torch.matmul(mat, norm_vec)
            max_sim = float(torch.max(sims).item())
            scores[role] = round(max(-1.0, min(1.0, max_sim)), 4)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        dominant_role, dominant_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        competitive_margin = dominant_score - runner_up

        return {
            "dominant_role": dominant_role,
            "confidence": dominant_score,
            "margin": round(competitive_margin, 4),
            "scores": scores,
        }

    @classmethod
    def classify_predicate_role(cls, predicate: str) -> Dict[str, Any]:
        """
        Classifies the semantic communicative role of a predicate.
        Routes through the canonical serialization boundary.
        """
        cleaned = (predicate or "").strip()
        if not cleaned:
            return {"dominant_role": None, "confidence": 0.0, "margin": 0.0, "scores": {}}

        try:
            from ..linguistic.canonical_tokenizer import ArisCanonicalTokenizer
            seq = ArisCanonicalTokenizer().serialize(cleaned, source="predicate", add_special_tokens=False)
            return cls.classify_predicate_role_sequence(seq)
        except Exception:
            cls._ensure_loaded()
            vec = torch.tensor(ArisSemanticEncoder.encode(cleaned), dtype=torch.float32)
            norm_vec = vec / (torch.norm(vec) + 1e-8)
            scores = {}
            for role, mat in cls._predicate_matrices.items():
                sims = torch.matmul(mat, norm_vec)
                scores[role] = round(max(-1.0, min(1.0, float(torch.max(sims).item()))), 4)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            dominant_role, dominant_score = ranked[0]
            runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
            return {
                "dominant_role": dominant_role,
                "confidence": dominant_score,
                "margin": round(dominant_score - runner_up, 4),
                "scores": scores,
            }

    @classmethod
    def classify_speech_act_sequence(
        cls,
        sequence: Any,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Classifies speech act directly from a canonical TokenizedSequence and its continuous hidden states.
        """
        cls._ensure_loaded()
        if not sequence or not getattr(sequence, "tokens", None):
            return {"dominant_act": "PHATIC", "scores": {"PHATIC": 1.0}, "confidence": 1.0}

        norm_vec = cls._pool_sequence_vector(sequence, hidden_states)

        scores: Dict[str, float] = {}
        for act, mat in cls._speech_matrices.items():
            sims = torch.matmul(mat, norm_vec)
            max_sim = float(torch.max(sims).item())
            scores[act] = round(max(-1.0, min(1.0, max_sim)), 4)

        # Check opening token against interrogative matrices
        tokens = [t for t in sequence.tokens if t not in {"[BOS]", "[EOS]", "[PAD]", "[UNK]"}]
        first_token = tokens[0] if tokens else ""
        if first_token and cls._interrogative_matrices:
            if hidden_states is not None and len(sequence.tokens) > 1:
                first_idx = 1 if sequence.tokens[0] == "[BOS]" else 0
                hs = hidden_states[0] if hidden_states.dim() == 3 else hidden_states
                if first_idx < hs.shape[0]:
                    f_vec = hs[first_idx]
                    f_norm = f_vec / (torch.norm(f_vec) + 1e-8)
                else:
                    f_vec = torch.tensor(ArisSemanticEncoder.encode(first_token), dtype=torch.float32)
                    f_norm = f_vec / (torch.norm(f_vec) + 1e-8)
            else:
                f_vec = torch.tensor(ArisSemanticEncoder.encode(first_token), dtype=torch.float32)
                f_norm = f_vec / (torch.norm(f_vec) + 1e-8)

            max_op_sim = max(float(torch.max(torch.matmul(mat, f_norm)).item()) for mat in cls._interrogative_matrices.values())
            # Continual score enhancement via maximum projection without magic number threshold
            if max_op_sim > scores.get("INTERROGATIVE", -1.0):
                scores["INTERROGATIVE"] = round(max(scores.get("INTERROGATIVE", 0.0), max_op_sim), 4)

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        dominant_act, dominant_val = sorted_scores[0]

        return {
            "dominant_act": dominant_act,
            "scores": scores,
            "confidence": dominant_val,
        }

    @classmethod
    def classify_speech_act(cls, utterance: str) -> Dict[str, Any]:
        """
        Classifies speech act of utterance, routing strictly through the canonical serialization boundary.
        Zero raw string splitting, zero hardcoded magic thresholds.
        """
        cleaned = utterance.strip() if utterance else ""
        if not cleaned:
            return {"dominant_act": "PHATIC", "scores": {"PHATIC": 1.0}, "confidence": 1.0}

        from ..linguistic.canonical_tokenizer import ArisCanonicalTokenizer
        seq = ArisCanonicalTokenizer().serialize(cleaned, source="speech_act_query")
        return cls.classify_speech_act_sequence(seq)

    @classmethod
    def classify_deictic_token(cls, token: str, threshold: float = 0.90) -> Optional[Tuple[str, float]]:
        cls._ensure_loaded()
        cleaned = token.strip()
        if not cleaned:
            return None

        vec = torch.tensor(ArisSemanticEncoder.encode(cleaned), dtype=torch.float32)
        norm_vec = vec / (torch.norm(vec) + 1e-8)

        best_role = None
        best_sim = -1.0

        for role, mat in cls._deictic_matrices.items():
            sims = torch.matmul(mat, norm_vec)
            max_sim = float(torch.max(sims).item())
            if max_sim > best_sim:
                best_sim = max_sim
                best_role = role

        if best_sim >= threshold and best_role:
            return (best_role, best_sim)
        return None

    @classmethod
    def infer_interrogative_role_sequence(
        cls,
        sequence: Any,
        hidden_states: Optional[torch.Tensor] = None,
        threshold: float = 0.60,
    ) -> Tuple[str, float]:
        """
        Infers interrogative question role directly from canonical TokenizedSequence.
        """
        cls._ensure_loaded()
        if not sequence or not getattr(sequence, "tokens", None):
            return ("entity", 1.0)

        norm_vec = cls._pool_sequence_vector(sequence, hidden_states)

        best_role = "entity"
        best_sim = -1.0

        for role, mat in cls._interrogative_matrices.items():
            sims = torch.matmul(mat, norm_vec)
            max_sim = float(torch.max(sims).item())
            if max_sim > best_sim:
                best_sim = max_sim
                best_role = role

        return (best_role, max(0.0, min(1.0, best_sim)))

    @classmethod
    def infer_interrogative_role(cls, operator_token: str, threshold: float = 0.60) -> Tuple[str, float]:
        cleaned = operator_token.strip() if operator_token else ""
        if not cleaned:
            return ("entity", 1.0)

        try:
            from ..linguistic.canonical_tokenizer import ArisCanonicalTokenizer
            seq = ArisCanonicalTokenizer().serialize(cleaned, source="interrogative_token", add_special_tokens=False)
            return cls.infer_interrogative_role_sequence(seq, threshold=threshold)
        except Exception:
            cls._ensure_loaded()
            vec = torch.tensor(ArisSemanticEncoder.encode(cleaned), dtype=torch.float32)
            norm_vec = vec / (torch.norm(vec) + 1e-8)
            best_role = "entity"
            best_sim = -1.0
            for role, mat in cls._interrogative_matrices.items():
                sims = torch.matmul(mat, norm_vec)
                max_sim = float(torch.max(sims).item())
                if max_sim > best_sim:
                    best_sim = max_sim
                    best_role = role
            return (best_role, max(0.0, min(1.0, best_sim)))
