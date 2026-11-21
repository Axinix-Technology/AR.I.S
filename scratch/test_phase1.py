import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from aris.configs.model_config import ArisConfig
from aris.models.aris_lm import ArisLM
from aris.models.rope import ArisRotaryEmbedding, apply_rotary_pos_emb
from aris.models.attention import ArisGQAWithKVCache
from aris.tokenizer.bpe_tokenizer import ArisTokenizer

def test_hf_llama_parity():
    print("=== TEST 1: HF LlamaForCausalLM END-TO-END PARITY ===")
    aris_cfg = ArisConfig(
        vocab_size=4096,
        max_seq_len=128,
        embed_dim=512,
        num_layers=4,
        num_query_heads=8,
        num_kv_heads=2,
        ff_dim=1408,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
    )
    llama_cfg = LlamaConfig(
        vocab_size=4096,
        max_position_embeddings=128,
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=1408,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
    )

    torch.manual_seed(42)
    aris_model = ArisLM(aris_cfg)
    aris_model.eval()

    llama_model = LlamaForCausalLM(llama_cfg)
    llama_model.eval()

    with torch.no_grad():
        llama_model.model.embed_tokens.weight.copy_(aris_model.tok_emb.weight)
        llama_model.model.norm.weight.copy_(aris_model.norm_f.weight)
        llama_model.lm_head.weight.copy_(aris_model.lm_head.weight)
        
        for i in range(4):
            llama_model.model.layers[i].input_layernorm.weight.copy_(aris_model.blocks[i].attn_norm.weight)
            llama_model.model.layers[i].post_attention_layernorm.weight.copy_(aris_model.blocks[i].mlp_norm.weight)
            
            llama_model.model.layers[i].self_attn.q_proj.weight.copy_(aris_model.blocks[i].attn.q_proj.weight)
            llama_model.model.layers[i].self_attn.k_proj.weight.copy_(aris_model.blocks[i].attn.k_proj.weight)
            llama_model.model.layers[i].self_attn.v_proj.weight.copy_(aris_model.blocks[i].attn.v_proj.weight)
            llama_model.model.layers[i].self_attn.o_proj.weight.copy_(aris_model.blocks[i].attn.out_proj.weight)
            
            llama_model.model.layers[i].mlp.gate_proj.weight.copy_(aris_model.blocks[i].mlp.w_gate.weight)
            llama_model.model.layers[i].mlp.up_proj.weight.copy_(aris_model.blocks[i].mlp.w_up.weight)
            llama_model.model.layers[i].mlp.down_proj.weight.copy_(aris_model.blocks[i].mlp.w_down.weight)

    test_input = torch.randint(0, 4096, (2, 16))
    with torch.no_grad():
        aris_logits, _, _ = aris_model(test_input)
        llama_logits = llama_model(test_input).logits

    max_diff = torch.max(torch.abs(aris_logits - llama_logits)).item()
    print(f"Max absolute logit diff vs HF LLaMA: {max_diff:.8e}")
    assert max_diff < 1e-4, f"Parity violated: {max_diff}"
    print("Result: PASS")

def test_causality():
    print("\n=== TEST 2: STRICT CAUSALITY PERTURBATION TEST ===")
    cfg = ArisConfig(embed_dim=256, num_layers=4, num_query_heads=4, num_kv_heads=2, ff_dim=704)
    model = ArisLM(cfg)
    model.eval()

    torch.manual_seed(101)
    seq_a = torch.randint(0, 4096, (1, 20))
    seq_b = seq_a.clone()
    # Perturb future tokens at position 12..19
    seq_b[0, 12:] = (seq_b[0, 12:] + 50) % 4096

    with torch.no_grad():
        logits_a, _, _ = model(seq_a)
        logits_b, _, _ = model(seq_b)

    # Earlier positions 0..11 must have EXACTLY 0.0 logit difference
    past_diff = torch.max(torch.abs(logits_a[:, :12, :] - logits_b[:, :12, :])).item()
    print(f"Max logit difference at unperturbed positions 0..11: {past_diff:.8e}")
    assert past_diff == 0.0, f"Causality leak detected! Difference: {past_diff}"
    print("Result: PASS (Strictly Causal, 0.0 leak)")

def test_rope_relative_invariance():
    print("\n=== TEST 3: RoPE RELATIVE POSITION INVARIANCE ===")
    dim = 64
    rope = ArisRotaryEmbedding(dim=dim, max_seq_len=256)
    
    torch.manual_seed(202)
    q = torch.randn(1, 1, 1, dim) # 1 token
    k = torch.randn(1, 1, 1, dim) # 1 token

    # Position pair (5, 2): relative distance = 3
    cos_5, sin_5 = rope(seq_len=1, offset=5)
    cos_2, sin_2 = rope(seq_len=1, offset=2)
    q_rot_5, _ = apply_rotary_pos_emb(q, q, cos_5, sin_5)
    _, k_rot_2 = apply_rotary_pos_emb(k, k, cos_2, sin_2)
    dot_dist_3_a = torch.matmul(q_rot_5, k_rot_2.transpose(-1, -2)).item()

    # Position pair (15, 12): relative distance = 3 (shifted by +10)
    cos_15, sin_15 = rope(seq_len=1, offset=15)
    cos_12, sin_12 = rope(seq_len=1, offset=12)
    q_rot_15, _ = apply_rotary_pos_emb(q, q, cos_15, sin_15)
    _, k_rot_12 = apply_rotary_pos_emb(k, k, cos_12, sin_12)
    dot_dist_3_b = torch.matmul(q_rot_15, k_rot_12.transpose(-1, -2)).item()

    rope_diff = abs(dot_dist_3_a - dot_dist_3_b)
    print(f"Attention dot product at (5, 2): {dot_dist_3_a:.8f}")
    print(f"Attention dot product at (15, 12) [same relative distance 3]: {dot_dist_3_b:.8f}")
    print(f"Absolute diff: {rope_diff:.8e}")
    assert rope_diff < 1e-5, f"RoPE relative invariance violated: {rope_diff}"
    print("Result: PASS (Relative position shift invariant)")

def test_gqa_grouping():
    print("\n=== TEST 4: GROUPED-QUERY ATTENTION (GQA) REPEAT & PROJECTION ===")
    embed_dim = 256
    num_q_heads = 8
    num_kv_heads = 2
    gqa = ArisGQAWithKVCache(embed_dim=embed_dim, num_query_heads=num_q_heads, num_kv_heads=num_kv_heads)
    
    assert gqa.num_queries_per_kv == 4, f"Expected 4 queries per KV head, got {gqa.num_queries_per_kv}"
    print(f"Verified GQA ratio: {num_q_heads} Q heads / {num_kv_heads} KV heads = {gqa.num_queries_per_kv}:1 ratio")
    
    x = torch.randn(2, 8, embed_dim)
    rope = ArisRotaryEmbedding(dim=gqa.head_dim, max_seq_len=64)
    cos, sin = rope(seq_len=8, offset=0)
    
    out, cache = gqa(x, cos=cos, sin=sin, use_cache=True)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    assert cache[0].shape == (2, num_kv_heads, 8, gqa.head_dim)
    print(f"KV Cache shape verified: {cache[0].shape}")
    print("Result: PASS")

def test_overfit_one_batch():
    print("\n=== TEST 5: OVERFIT ONE BATCH TO NEAR-ZERO LOSS ===")
    cfg = ArisConfig(embed_dim=256, num_layers=4, num_query_heads=4, num_kv_heads=2, ff_dim=704)
    model = ArisLM(cfg)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    torch.manual_seed(42)
    batch_in = torch.randint(0, 4096, (2, 16))
    batch_tgt = batch_in.clone()

    initial_loss = None
    for step in range(120):
        optimizer.zero_grad()
        _, loss, _ = model(batch_in, targets=batch_tgt)
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    print(f"Initial Loss (Random): {initial_loss:.4f}")
    print(f"Final Loss after 120 steps: {final_loss:.6f}")
    assert final_loss < 0.05, f"Overfit failed: final loss {final_loss}"
    print("Result: PASS (Model smoothly overfits to loss < 0.05)")

def test_tokenizer_round_trip():
    print("\n=== TEST 6: TOKENIZER EXACT ROUND-TRIP & SPECIAL TOKENS ===")
    tok = ArisTokenizer.load("models/aris_bpe_tokenizer.json")
    
    # Assert special tokens
    special_ids = {t: tok.get_token_id(t) for t in tok.SPECIAL_TOKENS}
    print("Special token IDs in vocab:", special_ids)
    
    test_samples = [
        "<|im_start|>user\nWhat is 15 + 27?<|im_end|>\n<|im_start|>assistant\n<|thought|>\nCalculating sum.\nAnswer: 42<|im_end|>\n",
        "def is_even(n: int) -> bool:\n    return n % 2 == 0\n",
        "Special chars & indentation:    leading spaces, \t tabs, \n newlines. 123 + 456 = 579.",
        "Non-ASCII utf-8: \u2192 \u2264 \u2265 \u00b1 \u03c0",
    ]
    
    for idx, s in enumerate(test_samples):
        encoded = tok.encode(s, add_special_tokens=False)
        decoded = tok.decode(encoded, skip_special_tokens=False)
        assert s == decoded, f"Roundtrip failed on sample {idx}!\nOriginal: {repr(s)}\nDecoded:  {repr(decoded)}"
    
    print(f"All {len(test_samples)} complex test strings roundtripped with 100% byte fidelity.")
    print("Result: PASS")

if __name__ == "__main__":
    print("=======================================================")
    print("  AR.I.S PHASE 1: ARCHITECTURAL PARITY & TEST SUITE")
    print("=======================================================\n")
    test_hf_llama_parity()
    test_causality()
    test_rope_relative_invariance()
    test_gqa_grouping()
    test_overfit_one_batch()
    test_tokenizer_round_trip()
    print("\n=======================================================")
    print("  ALL 6 ARCHITECTURAL TESTS PASSED CLEANLY (GATE: PASS)")
    print("=======================================================")
