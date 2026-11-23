"""
Empirical Verification of Loss Mechanics:
1. Token Shift Alignment Test (Proof of Zero Double-Shift).
2. NaN Guard on All-Masked Targets.
3. Single-Batch Overfitting Sanity Check (Loss -> ~0).
"""

import pytest
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from aris.configs.model_config import ArisConfig
from aris.models.aris_lm import ArisLM
from aris.tokenizer.bpe_tokenizer import ArisTokenizer
from aris.data.pipeline_v2 import ArisDataPipelineV2


def test_token_shift_alignment_empirical():
    """
    Empirical check on actual pipeline output:
    For every position t where shift_targets[t] != -100:
    shift_targets[t] MUST be exactly inp[t+1].
    This mathematically disproves double-shifting.
    """
    tok = ArisTokenizer.load("models/aris_bpe_tokenizer_v2.json")
    pipeline = ArisDataPipelineV2(tokenizer=tok, max_seq_len=128, seed=42)
    data_iter = iter(pipeline)

    checked_pairs = 0
    for _ in range(10):
        inp, tgt = next(data_iter)
        assert len(inp) == len(tgt)

        # Replicate model's internal shift
        shift_inputs = inp[:-1]
        shift_targets = tgt[1:]

        # Find all positions where target is supervised (not -100)
        supervised_indices = (shift_targets != -100).nonzero(as_tuple=True)[0]
        assert len(supervised_indices) > 0, "Pipeline sample must have supervised tokens"

        for idx in supervised_indices:
            pos = idx.item()
            # The token that predicts position pos+1 is inp[pos]
            predicting_token_id = shift_inputs[pos].item()
            target_token_id = shift_targets[pos].item()
            expected_next_token_id = inp[pos + 1].item()

            # STRICT INVARIANT: The target must be identical to the actual next token in input
            assert target_token_id == expected_next_token_id, (
                f"Double shift detected! At pos {pos}: target {target_token_id} != input {expected_next_token_id}"
            )
            checked_pairs += 1

    print(f"Verified {checked_pairs} supervised token prediction pairs with zero alignment discrepancy.")


def test_nan_guard_on_all_masked_targets():
    """
    Edge case: When all targets are -100, F.cross_entropy returns NaN (0/0).
    Verify that our forward pass returns 0.0 without NaN.
    """
    cfg = ArisConfig(
        vocab_size=8192,
        max_seq_len=64,
        embed_dim=128,
        num_layers=2,
        num_query_heads=4,
        num_kv_heads=2,
        ff_dim=384,
        tie_word_embeddings=True,
    )
    model = ArisLM(cfg)
    model.eval()

    x = torch.randint(10, 8000, (2, 32), dtype=torch.long)
    # All targets are -100
    y = torch.full((2, 32), -100, dtype=torch.long)

    logits, loss, _ = model(x, targets=y)
    assert loss is not None
    assert not torch.isnan(loss), "Loss must not be NaN when all targets are masked (-100)"
    assert loss.item() == 0.0


def test_single_batch_overfit_to_near_zero():
    """
    Gold Standard Sanity Check:
    A functioning autoregressive transformer MUST be able to memorize
    and overfit a single small batch down to near-zero loss (< 0.05).
    If there is any double-shift or broken loss calculation, loss will plateau.
    """
    torch.manual_seed(42)
    cfg = ArisConfig(
        vocab_size=8192,
        max_seq_len=32,
        embed_dim=128,
        num_layers=2,
        num_query_heads=4,
        num_kv_heads=2,
        ff_dim=384,
        tie_word_embeddings=True,
    )
    model = ArisLM(cfg)
    model.train()

    optimizer = AdamW(model.parameters(), lr=3e-3)

    # 2 synthetic sequences: prompt (masked -100) + response (supervised)
    # Seq 1: 8 prompt tokens, 16 response tokens
    s1_prompt = torch.randint(100, 2000, (8,))
    s1_resp = torch.randint(100, 2000, (16,))
    s1_x = torch.cat([s1_prompt, s1_resp])
    s1_y = torch.cat([torch.full((8,), -100, dtype=torch.long), s1_resp])

    # Seq 2: 6 prompt tokens, 18 response tokens
    s2_prompt = torch.randint(100, 2000, (6,))
    s2_resp = torch.randint(100, 2000, (18,))
    s2_x = torch.cat([s2_prompt, s2_resp])
    s2_y = torch.cat([torch.full((6,), -100, dtype=torch.long), s2_resp])

    # Pad to 32
    def pad(t, fill, length=32):
        return torch.cat([t, torch.full((length - len(t),), fill, dtype=torch.long)])

    batch_x = torch.stack([pad(s1_x, 0), pad(s2_x, 0)])
    batch_y = torch.stack([pad(s1_y, -100), pad(s2_y, -100)])

    # Initial loss
    _, initial_loss, _ = model(batch_x, targets=batch_y)
    assert initial_loss.item() > 6.0, f"Expected high initial loss, got {initial_loss.item()}"

    # Overfit for 120 steps
    final_loss = 0.0
    for step in range(120):
        optimizer.zero_grad()
        _, loss, _ = model(batch_x, targets=batch_y)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    # The loss MUST collapse to near-zero
    assert final_loss < 0.05, f"Model failed to overfit single batch! Final loss was {final_loss:.4f}"
    print(f"Overfitting verified: Loss dropped from {initial_loss.item():.4f} -> {final_loss:.4f} (< 0.05).")
