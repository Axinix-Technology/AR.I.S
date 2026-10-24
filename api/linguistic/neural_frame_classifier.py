"""
AR.I.S — Local Neural Frame Classifier (PyTorch)

Core Directives:
  - Consumes the structural frame (Subject, Relation, Object), NOT character identity.
  - Predicts which slot is the open parameter variable (?) vs bound anchor.
  - Fully offline, local, deterministic, and fast (< 1 ms inference).
  - ZERO arithmetic in training.
"""

import os
from typing import Dict, Any, Optional, List
from .frame_dataset import ArisFrameVectorizer, ArisStructuralCurriculumGenerator


class ArisFrameClassifierNet:
    """
    Lightweight PyTorch Neural Network for Structural Frame Parameter Classification.
    Input: 14 structural invariants.
    Output: 2 logits [P(Subject is Var), P(Object is Var)].
    """

    def __init__(self):
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, input_dim: int = 14, hidden_dim: int = 32):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.relu1 = nn.ReLU()
                self.fc2 = nn.Linear(hidden_dim, 16)
                self.relu2 = nn.ReLU()
                self.out = nn.Linear(16, 2)

            def forward(self, x):
                x = self.relu1(self.fc1(x))
                x = self.relu2(self.fc2(x))
                return self.out(x)

        self._model = _Net(input_dim=ArisFrameVectorizer.FEATURE_DIM)


class ArisNeuralFrameLearner:
    """
    Trainer, persistence manager, and offline inference engine for the Frame Classifier.
    """

    DEFAULT_MODEL_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "models",
        "aris_frame_classifier.pt"
    )

    _loaded_model: Optional[Any] = None

    @classmethod
    def train(
        cls,
        curriculum: Optional[List[Dict[str, Any]]] = None,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 0.01,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Trains the local neural network on structural curricula using PyTorch.
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim

        if curriculum is None:
            curriculum = ArisStructuralCurriculumGenerator.generate_curriculum(num_samples=2000)

        # Build tensors
        X_list = [item["features"] for item in curriculum]
        y_list = [item["label"] for item in curriculum]

        X_tensor = torch.tensor(X_list, dtype=torch.float32)
        y_tensor = torch.tensor(y_list, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        net = ArisFrameClassifierNet()._model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        final_loss = 0.0
        for epoch in range(epochs):
            net.train()
            running_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = net(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_x.size(0)
            final_loss = running_loss / len(dataset)

        # Evaluate final training accuracy
        net.eval()
        with torch.no_grad():
            outputs = net(X_tensor)
            preds = torch.argmax(outputs, dim=1)
            correct = (preds == y_tensor).sum().item()
            accuracy = float(correct) / len(dataset)

        # Save weights
        target_path = save_path or cls.DEFAULT_MODEL_PATH
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        torch.save(net.state_dict(), target_path)

        cls._loaded_model = net

        return {
            "status": "trained",
            "epochs": epochs,
            "samples": len(dataset),
            "final_loss": round(final_loss, 4),
            "accuracy": round(accuracy, 4),
            "saved_to": target_path,
        }

    @classmethod
    def load_model(cls, model_path: Optional[str] = None):
        """
        Loads persisted model weights.
        """
        import torch

        target_path = model_path or cls.DEFAULT_MODEL_PATH
        if not os.path.exists(target_path):
            return None

        net = ArisFrameClassifierNet()._model
        state = torch.load(target_path, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        net.eval()
        cls._loaded_model = net
        return net

    @classmethod
    def predict(
        cls,
        subject: str,
        relation: str,
        obj: str,
        subj_token_idx: int = 0,
        obj_token_idx: int = 2,
        total_tokens: int = 3,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes local neural prediction in microseconds.
        """
        import torch

        if cls._loaded_model is None:
            cls._loaded_model = cls.load_model(model_path)

        if cls._loaded_model is None:
            # Train if no saved model exists yet
            cls.train()

        net = cls._loaded_model
        net.eval()

        feats = ArisFrameVectorizer.vectorize_frame(
            subject=subject,
            relation=relation,
            obj=obj,
            subj_token_idx=subj_token_idx,
            obj_token_idx=obj_token_idx,
            total_tokens=total_tokens,
        )

        with torch.no_grad():
            x = torch.tensor([feats], dtype=torch.float32)
            logits = net(x)
            probs = torch.softmax(logits, dim=1).squeeze().tolist()
            pred_class = int(torch.argmax(logits, dim=1).item())

        if pred_class == 0:
            var_slot = "subject"
            anc_slot = "object"
            conf = probs[0]
        else:
            var_slot = "object"
            anc_slot = "subject"
            conf = probs[1]

        return {
            "status": "ok",
            "variable_slot": var_slot,
            "anchor_slot": anc_slot,
            "confidence": round(conf, 4),
            "source": "local_neural_frame_classifier",
        }

    @classmethod
    def predict_from_relational_binding(
        cls,
        binding: Dict[str, Any],
        sequence: Optional[Any] = None,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes local neural frame classification directly on a canonical relational binding.
        """
        subj = binding.get("subject", {}).get("text", "")
        rel = binding.get("relation", {}).get("text", "")
        obj = binding.get("object", {}).get("text", "")

        subj_indices = binding.get("subject", {}).get("token_indices", [0])
        obj_indices = binding.get("object", {}).get("token_indices", [2])

        subj_idx = subj_indices[0] if subj_indices else 0
        obj_idx = obj_indices[0] if obj_indices else 2
        total_tokens = getattr(sequence, "sequence_length", max(subj_idx, obj_idx) + 1)

        return cls.predict(
            subject=subj,
            relation=rel,
            obj=obj,
            subj_token_idx=subj_idx,
            obj_token_idx=obj_idx,
            total_tokens=total_tokens,
            model_path=model_path,
        )
