import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import LlamaConfig, LlamaForCausalLM
import pyarrow.parquet as pq

from aris.training.trainer import ArisTrainer
from aris.configs.model_config import ArisConfig
from aris.models.aris_lm import ArisLM
from aris.inference.generation import ArisInferenceEngine
from aris.tokenizer.bpe_tokenizer import ArisTokenizer

def main():
    # 1. Load canonical checkpoint
    tok = ArisTokenizer.load("models/aris_bpe_tokenizer.json")
    aris_model, step = ArisTrainer.load_model("aris_lm_checkpoint.pt", device=torch.device("cpu"))
    aris_model.eval()

    cfg = aris_model.config
    print("=== AR.I.S CONFIG ===")
    print(f"d_model={cfg.embed_dim}, num_layers={cfg.num_layers}, num_query_heads={cfg.num_query_heads}, num_kv_heads={cfg.num_kv_heads}, head_dim={cfg.head_dim}")

    # 2. Build HF LlamaConfig and model
    llama_cfg = LlamaConfig(
        vocab_size=cfg.vocab_size,
        max_position_embeddings=cfg.max_seq_len,
        hidden_size=cfg.embed_dim,
        num_hidden_layers=cfg.num_layers,
        num_attention_heads=cfg.num_query_heads,
        num_key_value_heads=cfg.num_kv_heads,
        intermediate_size=cfg.ff_dim,
        tie_word_embeddings=cfg.tie_word_embeddings,
        rope_theta=cfg.rope_theta,
        rms_norm_eps=cfg.rms_norm_eps,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
    )
    hf_model = LlamaForCausalLM(llama_cfg)
    hf_model.eval()

    print("\n=== HF MODEL PROFILE ===")
    print("Type:", type(hf_model))
    print("Attn Implementation:", getattr(hf_model.config, "_attn_implementation", "default"))

    # Copy weights from aris_model to hf_model
    with torch.no_grad():
        hf_model.model.embed_tokens.weight.copy_(aris_model.tok_emb.weight)
        hf_model.model.norm.weight.copy_(aris_model.norm_f.weight)
        hf_model.lm_head.weight.copy_(aris_model.lm_head.weight)
        for i in range(cfg.num_layers):
            hf_model.model.layers[i].input_layernorm.weight.copy_(aris_model.blocks[i].attn_norm.weight)
            hf_model.model.layers[i].post_attention_layernorm.weight.copy_(aris_model.blocks[i].mlp_norm.weight)
            hf_model.model.layers[i].self_attn.q_proj.weight.copy_(aris_model.blocks[i].attn.q_proj.weight)
            hf_model.model.layers[i].self_attn.k_proj.weight.copy_(aris_model.blocks[i].attn.k_proj.weight)
            hf_model.model.layers[i].self_attn.v_proj.weight.copy_(aris_model.blocks[i].attn.v_proj.weight)
            hf_model.model.layers[i].self_attn.o_proj.weight.copy_(aris_model.blocks[i].attn.out_proj.weight)
            hf_model.model.layers[i].mlp.gate_proj.weight.copy_(aris_model.blocks[i].mlp.w_gate.weight)
            hf_model.model.layers[i].mlp.up_proj.weight.copy_(aris_model.blocks[i].mlp.w_up.weight)
            hf_model.model.layers[i].mlp.down_proj.weight.copy_(aris_model.blocks[i].mlp.w_down.weight)

    # 3. Mutation test: verify perturbation produces non-zero diff
    with torch.no_grad():
        orig_val = hf_model.lm_head.weight[0, 0].item()
        hf_model.lm_head.weight[0, 0] += 0.5
        dummy_in = torch.randint(0, cfg.vocab_size, (1, 8))
        diff_mut = torch.max(torch.abs(aris_model(dummy_in)[0] - hf_model(dummy_in).logits)).item()
        print("\n=== MUTATION TEST ===")
        print(f"Max diff after perturbing hf lm_head.weight[0, 0] by +0.5: {diff_mut:.6f}")
        assert diff_mut > 0.0, "Mutation test failed: diff is zero!"
        print("Result: PASS (Sensitivity verified)")
        # Restore weight
        hf_model.lm_head.weight[0, 0] = orig_val

    # 4. Real tokens test (length 384, padded batch)
    table = pq.read_table("data/gsm8k/train.parquet")
    text1 = str(table["question"][0].as_py())
    text2 = str(table["question"][1].as_py())
    toks1 = tok.encode(text1, add_special_tokens=False)
    toks2 = tok.encode(text2, add_special_tokens=False)

    pad_id = tok.pad_id
    def pad_to_384(t):
        if len(t) < 384:
            return t + [pad_id] * (384 - len(t)), [1] * len(t) + [0] * (384 - len(t))
        return t[:384], [1] * 384

    t1_pad, m1 = pad_to_384(toks1)
    t2_pad, m2 = pad_to_384(toks2)

    input_ids = torch.tensor([t1_pad, t2_pad], dtype=torch.long)
    attn_mask = torch.tensor([m1, m2], dtype=torch.long)

    with torch.no_grad():
        aris_logits, _, _ = aris_model(input_ids)
        hf_out = hf_model(input_ids, attention_mask=attn_mask, output_hidden_states=True)
        hf_logits = hf_out.logits

    # Unmasked positions logit comparison (batch item 0 valid tokens, batch item 1 valid tokens)
    valid_mask = attn_mask.unsqueeze(-1).expand_as(aris_logits).bool()
    diff_real = torch.max(torch.abs(aris_logits[valid_mask] - hf_logits[valid_mask])).item()
    mean_real = torch.mean(torch.abs(aris_logits[valid_mask] - hf_logits[valid_mask])).item()
    print("\n=== REAL TOKENS & REAL CHECKPOINT WEIGHTS (Len 384) ===")
    print(f"Max abs logit diff on unmasked tokens: {diff_real:.8e}")
    print(f"Mean abs logit diff on unmasked tokens: {mean_real:.8e}")

    # 5. Per-layer hidden states comparison
    print("\n=== PER-LAYER HIDDEN STATES COMPARISON ===")
    aris_hidden = []
    with torch.no_grad():
        x = aris_model.tok_emb(input_ids)
        aris_hidden.append(x)  # 0: input embeddings
        cos, sin = aris_model.rope(seq_len=input_ids.size(1), offset=0)
        for blk in aris_model.blocks[:-1]:
            x, _ = blk(x, cos=cos, sin=sin)
            aris_hidden.append(x)  # 1..7: intermediate blocks (before final norm)
        # Final block + norm (which matches HF hidden_states[-1])
        x, _ = aris_model.blocks[-1](x, cos=cos, sin=sin)
        x_normed = aris_model.norm_f(x)
        aris_hidden.append(x_normed)  # 8: final layer post-norm

    for layer_idx, (ah, hh) in enumerate(zip(aris_hidden, hf_out.hidden_states)):
        mask_expanded = attn_mask.unsqueeze(-1).expand_as(ah).bool()
        h_diff = torch.max(torch.abs(ah[mask_expanded] - hh[mask_expanded])).item()
        label = "emb" if layer_idx == 0 else (f"blk_{layer_idx-1}" if layer_idx < 8 else "final_normed")
        print(f"Layer {layer_idx:2} ({label:12}) hidden state max diff: {h_diff:.8e}")

    # 6. Greedy 64 tokens generation comparison
    print("\n=== 64 GREEDY TOKENS GENERATION COMPARISON ===")
    prompt = "Solve step by step: What is 12 * 15?"
    p_ids = tok.encode(prompt, add_special_tokens=False)
    p_tensor = torch.tensor([p_ids], dtype=torch.long)

    # ARIS KV-cache generation
    aris_gen_out = ArisInferenceEngine.generate(aris_model, p_tensor, max_new_tokens=64, temperature=0.0, eos_id=tok.im_end_id)
    aris_gen_tokens = aris_gen_out[0, len(p_ids):].tolist()

    # HF generate
    hf_gen_out = hf_model.generate(p_tensor, max_new_tokens=64, do_sample=False, eos_token_id=tok.im_end_id)
    hf_gen_tokens = hf_gen_out[0, len(p_ids):].tolist()

    tokens_match = (aris_gen_tokens == hf_gen_tokens)
    print("ARIS generated tokens (first 10):", aris_gen_tokens[:10], "...")
    print("HF   generated tokens (first 10):", hf_gen_tokens[:10], "...")
    print("Do all 64 generated tokens match exactly?", tokens_match)
    if not tokens_match:
        for idx, (a, b) in enumerate(zip(aris_gen_tokens, hf_gen_tokens)):
            if a != b:
                print(f"First mismatch at step {idx}: ARIS={a} ({repr(tok.decode([a]))}) vs HF={b} ({repr(tok.decode([b]))})")
                break

    # 7. Actual K/V Cache Shape in canonical model
    print("\n=== CACHE SHAPE ON CANONICAL MODEL ===")
    _, _, kv_caches = aris_model(p_tensor, use_cache=True)
    k_cache_0, v_cache_0 = kv_caches[0]
    print(f"Canonical K Cache Shape for Layer 0: {k_cache_0.shape}")
    print(f"Canonical V Cache Shape for Layer 0: {v_cache_0.shape}")
    print(f"Formula: (Batch={p_tensor.size(0)}, num_kv_heads={cfg.num_kv_heads}, SeqLen={p_tensor.size(1)}, head_dim={cfg.head_dim})")
    assert k_cache_0.shape == (p_tensor.size(0), cfg.num_kv_heads, p_tensor.size(1), cfg.head_dim)
    print("Result: PASS (Matches (B, 2, T, 64) exactly)")

if __name__ == "__main__":
    main()
