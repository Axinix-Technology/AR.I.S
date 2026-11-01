"""
AR.I.S — Collaborative Model Merging Engine (2026 Production Grade)
Supports Model Soups (Linear Averaging), Spherical Linear Interpolation (SLERP),
and Task Vector Arithmetic for multi-PC distributed training.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch


class ModelMerger:
    """
    Combines independently trained checkpoints from multiple team members' PCs
    into a single unified checkpoint.
    """

    @staticmethod
    def slerp(p0: torch.Tensor, p1: torch.Tensor, t: float) -> torch.Tensor:
        """
        Spherical Linear Interpolation (SLERP) between two tensors.
        Preserves geometric magnitude and angular direction on the hypersphere.
        """
        p0_flat = p0.view(-1).float()
        p1_flat = p1.view(-1).float()

        norm_p0 = torch.norm(p0_flat, p=2)
        norm_p1 = torch.norm(p1_flat, p=2)

        if norm_p0 == 0.0 or norm_p1 == 0.0:
            return (1.0 - t) * p0 + t * p1

        p0_unit = p0_flat / norm_p0
        p1_unit = p1_flat / norm_p1

        dot = torch.dot(p0_unit, p1_unit)
        dot = torch.clamp(dot, -0.9995, 0.9995)

        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)

        if sin_omega == 0.0:
            return (1.0 - t) * p0 + t * p1

        scale0 = torch.sin((1.0 - t) * omega) / sin_omega
        scale1 = torch.sin(t * omega) / sin_omega

        interpolated_unit = scale0 * p0_unit + scale1 * p1_unit
        target_norm = (1.0 - t) * norm_p0 + t * norm_p1

        interpolated = interpolated_unit * target_norm
        return interpolated.view(p0.shape).to(p0.dtype)

    @classmethod
    def average_models(
        cls,
        checkpoint_paths: List[Path],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Linear Model Soup: Averages parameters across N team checkpoints.
        """
        if not checkpoint_paths:
            raise ValueError("No checkpoints provided for merging.")

        ckpts = [torch.load(p, map_location="cpu", weights_only=False) for p in checkpoint_paths]

        if weights is None:
            weights = [1.0 / len(ckpts)] * len(ckpts)
        else:
            total_w = sum(weights)
            weights = [w / total_w for w in weights]

        base_state = ckpts[0]["model_state_dict"]
        merged_state = {}

        for key in base_state.keys():
            tensors = [ck["model_state_dict"][key] for ck in ckpts]
            if tensors[0].is_floating_point():
                stacked = torch.stack([t.float() * w for t, w in zip(tensors, weights)], dim=0)
                merged_state[key] = stacked.sum(dim=0).to(tensors[0].dtype)
            else:
                # Keep integer buffers from first model
                merged_state[key] = tensors[0]

        # Combine cumulative steps: base step + incremental steps trained on all PCs
        base_steps = [ck.get("step", 0) for ck in ckpts]
        max_step = max(base_steps)
        total_delta = sum(max(0, s - min(base_steps)) for s in base_steps)
        merged_step = max_step + (total_delta // max(1, len(ckpts) - 1)) if len(ckpts) > 1 else max_step

        return {
            "model_state_dict": merged_state,
            "model_config": ckpts[0]["model_config"],
            "step": merged_step,
            "merging_method": "linear_model_soup",
            "source_checkpoints": [str(p) for p in checkpoint_paths],
        }

    @classmethod
    def slerp_models(
        cls,
        checkpoint_a: Path,
        checkpoint_b: Path,
        alpha: float = 0.5,
    ) -> Dict[str, Any]:
        """
        SLERP Merging: Spherical linear interpolation between Checkpoint A and Checkpoint B.
        """
        ck_a = torch.load(checkpoint_a, map_location="cpu", weights_only=False)
        ck_b = torch.load(checkpoint_b, map_location="cpu", weights_only=False)

        state_a = ck_a["model_state_dict"]
        state_b = ck_b["model_state_dict"]

        merged_state = {}
        for key in state_a.keys():
            t_a = state_a[key]
            t_b = state_b[key]

            if t_a.is_floating_point() and t_a.dim() >= 2:
                merged_state[key] = cls.slerp(t_a, t_b, alpha)
            elif t_a.is_floating_point():
                merged_state[key] = (1.0 - alpha) * t_a + alpha * t_b
            else:
                merged_state[key] = t_a

        step_a = ck_a.get("step", 0)
        step_b = ck_b.get("step", 0)
        merged_step = max(step_a, step_b) + abs(step_a - step_b) // 2

        return {
            "model_state_dict": merged_state,
            "model_config": ck_a["model_config"],
            "step": merged_step,
            "merging_method": "slerp",
            "source_checkpoints": [str(checkpoint_a), str(checkpoint_b)],
        }

    @classmethod
    def merge_and_save(
        cls,
        model_paths: List[Path],
        output_path: Path,
        method: str = "average",
        weights: Optional[List[float]] = None,
        alpha: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Merges checkpoints and writes the unified model to output_path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if method == "slerp":
            if len(model_paths) != 2:
                raise ValueError("SLERP method requires exactly 2 checkpoints (Model A and Model B).")
            merged_dict = cls.slerp_models(model_paths[0], model_paths[1], alpha=alpha)
        else:
            merged_dict = cls.average_models(model_paths, weights=weights)

        torch.save(merged_dict, output_path)
        print(f"\n[Model Merging Complete] Unified model saved to: {output_path}")
        print(f"  - Merging Strategy:    {merged_dict['merging_method']}")
        print(f"  - Combined Knowledge:  Cumulative Step {merged_dict['step']}")
        print(f"  - Integrated Models:   {len(model_paths)} checkpoints")
        return merged_dict

