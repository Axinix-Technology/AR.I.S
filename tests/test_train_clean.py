"""
Unit Tests for AR.I.S Clean Training Runner V2
"""

import torch
import pytest
from aris.configs.model_config import ArisConfig
from aris.models.aris_lm import ArisLM
from aris.training.trainer import get_cosine_lr


def test_cosine_lr_schedule():
    max_lr = 6e-4
    min_lr = 6e-5
    warmup = 100
    total = 1000

    # Warmup phase: step 0 -> lr > 0, step 99 -> ~max_lr
    lr_0 = get_cosine_lr(0, warmup, total, max_lr, min_lr)
    assert lr_0 == max_lr * (1.0 / warmup)

    lr_warmup_end = get_cosine_lr(99, warmup, total, max_lr, min_lr)
    assert abs(lr_warmup_end - max_lr) < 1e-7

    # Midpoint
    lr_mid = get_cosine_lr(550, warmup, total, max_lr, min_lr)
    assert min_lr < lr_mid < max_lr

    # End
    lr_end = get_cosine_lr(1000, warmup, total, max_lr, min_lr)
    assert abs(lr_end - min_lr) < 1e-7


def test_batch_padding_and_forward_pass():
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

    # Create two variable-length sample tensors (e.g. len 20 and len 35)
    t1_in = torch.randint(10, 8000, (20,), dtype=torch.long)
    t1_tgt = torch.cat([torch.full((5,), -100, dtype=torch.long), t1_in[5:]])

    t2_in = torch.randint(10, 8000, (35,), dtype=torch.long)
    t2_tgt = torch.cat([torch.full((10,), -100, dtype=torch.long), t2_in[10:]])

    batch_inputs = [t1_in, t2_in]
    batch_targets = [t1_tgt, t2_tgt]

    target_seq_len = cfg.max_seq_len  # 64
    padded_inputs = []
    padded_targets = []

    for inp_t, tgt_t in zip(batch_inputs, batch_targets):
        cur_len = len(inp_t)
        if cur_len < target_seq_len:
            pad_len = target_seq_len - cur_len
            p_inp = torch.cat([inp_t, torch.full((pad_len,), 0, dtype=torch.long)])
            p_tgt = torch.cat([tgt_t, torch.full((pad_len,), -100, dtype=torch.long)])
        else:
            p_inp = inp_t[:target_seq_len]
            p_tgt = tgt_t[:target_seq_len]
        padded_inputs.append(p_inp)
        padded_targets.append(p_tgt)

    x = torch.stack(padded_inputs)
    y = torch.stack(padded_targets)

    assert x.shape == (2, 64)
    assert y.shape == (2, 64)

    logits, loss, _ = model(x, targets=y)
    assert logits.shape == (2, 64, 8192)
    assert loss is not None
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_trainer_load_model_supports_both_config_keys(tmp_path):
    import tempfile
    from aris.training.trainer import ArisTrainer

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

    # 1. Test loading checkpoint with 'config' key
    ckpt_file_1 = tmp_path / "test_ckpt_1.pt"
    torch.save({
        "step": 42,
        "config": cfg.__dict__,
        "model_state_dict": model.state_dict(),
    }, ckpt_file_1)

    loaded_m1, step1 = ArisTrainer.load_model(ckpt_file_1, device=torch.device("cpu"))
    assert step1 == 42
    assert loaded_m1.config.vocab_size == 8192

    # 2. Test loading checkpoint with 'model_config' key
    ckpt_file_2 = tmp_path / "test_ckpt_2.pt"
    torch.save({
        "step": 99,
        "model_config": cfg.__dict__,
        "model_state_dict": model.state_dict(),
    }, ckpt_file_2)

    loaded_m2, step2 = ArisTrainer.load_model(ckpt_file_2, device=torch.device("cpu"))
    assert step2 == 99
    assert loaded_m2.config.vocab_size == 8192


def test_budget_multiplier_calculations():
    import math

    # Small rung configuration
    cfg = ArisConfig(
        vocab_size=8192,
        max_seq_len=384,
        embed_dim=512,
        num_layers=8,
        num_query_heads=8,
        num_kv_heads=2,
        ff_dim=1408,
        tie_word_embeddings=True,
    )
    param_count = cfg.expected_param_count()
    assert param_count == 26_747_392

    batch_size = 16
    accum_steps = 2
    tokens_per_step = batch_size * cfg.max_seq_len * accum_steps
    assert tokens_per_step == 12_288

    # 1.0x Pilot budget
    pilot_tokens = int(param_count * 1.0)
    pilot_steps = math.ceil(pilot_tokens / tokens_per_step)
    assert pilot_tokens == 26_747_392
    assert pilot_steps == 2_177

    # 10.0x Scaling budget
    ten_x_tokens = int(param_count * 10.0)
    ten_x_steps = math.ceil(ten_x_tokens / tokens_per_step)
    assert ten_x_tokens == 267_473_920
    assert ten_x_steps == 21_768

    # 20.0x Full Chinchilla budget
    twenty_x_tokens = int(param_count * 20.0)
    twenty_x_steps = math.ceil(twenty_x_tokens / tokens_per_step)
    assert twenty_x_tokens == 534_947_840
    assert twenty_x_steps == 43_535

