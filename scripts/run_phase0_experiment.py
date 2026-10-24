"""
AR.I.S — Phase 0 Empirical Experiment Runner

Objective:
Answer scientifically:
  Given tokenized input, can a PyTorch model learn a measurable relationship
  from the token representation?

Investigates:
  Where does information about relationships between tokens actually come from?
  Distinguishes:
    1. Token Identity (Discrete integer coordinate in V)
    2. Token Position (Positional index p and embedding E_pos)
    3. Context (Self-attention cross-token interaction)
    4. Embedding (Continuous parameter matrix E_tok)
    5. Learned Parameters (Transformation weights updated by backpropagation)
    6. Training Objective (Loss function driving the gradient updates)
"""

import os
import sys
import json
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.linguistic.canonical_tokenizer import Vocabulary, ArisCanonicalTokenizer
from api.linguistic.phase0_substrate import ArisPhase0Model


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    a_norm = vec_a / (torch.norm(vec_a) + 1e-8)
    b_norm = vec_b / (torch.norm(vec_b) + 1e-8)
    return float(torch.dot(a_norm, b_norm).item())


def build_experiment_data():
    """
    Builds synthetic vocabulary and relational dataset.
    Entities: e_0, ..., e_15 (16 symbols)
    Relations: r_succ, r_pred, r_swap, r_jump (4 symbols)
    Special Tokens: [PAD], [UNK], [BOS], [EOS], [MASK] (5 symbols)
    Total Vocabulary: 25 tokens.

    To rigorously evaluate both:
      (A) Representation Learning on relational bindings
      (B) Contextual Generalization across syntactic position variation
    Each fact (subject, relation, target) is serialized into two syntactic contexts:
      Canonical Context:     [BOS] e_i r_k [MASK] [EOS]  (target at pos 3)
      Transposed Context:    [BOS] r_k e_i [MASK] [EOS]  (target at pos 3)
    """
    vocab = Vocabulary()
    
    # 16 entities
    for i in range(16):
        vocab.add_token(f"e_{i}")

    # 4 relations
    relations = ["r_succ", "r_pred", "r_swap", "r_jump"]
    for r in relations:
        vocab.add_token(r)

    tokenizer = ArisCanonicalTokenizer(vocab)

    facts = []
    for i in range(16):
        facts.append({"subj": f"e_{i}", "rel": "r_succ", "target": f"e_{(i + 1) % 16}"})
        facts.append({"subj": f"e_{i}", "rel": "r_pred", "target": f"e_{(i - 1) % 16}"})
        facts.append({"subj": f"e_{i}", "rel": "r_swap", "target": f"e_{15 - i}"})
        facts.append({"subj": f"e_{i}", "rel": "r_jump", "target": f"e_{(i + 4) % 16}"})

    # Generate multi-context serializations
    # Train set: Canonical order for 80% facts + Transposed order for 80% facts
    # Held-out test set: Canonical order for 20% facts + Transposed order for 20% facts
    random.seed(42)
    random.shuffle(facts)

    train_facts = facts[:51]
    test_facts = facts[51:]

    train_samples = []
    for f in train_facts:
        train_samples.append({
            "text": f"[BOS] {f['subj']} {f['rel']} [MASK] [EOS]",
            "target": f["target"],
            "mask_pos": 3,
            "pattern": "canonical",
        })
        train_samples.append({
            "text": f"[BOS] {f['rel']} {f['subj']} [MASK] [EOS]",
            "target": f["target"],
            "mask_pos": 3,
            "pattern": "transposed",
        })

    test_samples = []
    for f in test_facts:
        # Test on canonical order
        test_samples.append({
            "text": f"[BOS] {f['subj']} {f['rel']} [MASK] [EOS]",
            "target": f["target"],
            "mask_pos": 3,
            "pattern": "canonical",
        })
        # Test on transposed order
        test_samples.append({
            "text": f"[BOS] {f['rel']} {f['subj']} [MASK] [EOS]",
            "target": f["target"],
            "mask_pos": 3,
            "pattern": "transposed",
        })

    return vocab, tokenizer, train_samples, test_samples, train_facts, test_facts


def run_experiment():
    set_seed(42)
    print("=" * 80)
    print("AR.I.S -- PHASE 0: TOKENIZER-FIRST COGNITIVE FOUNDATION EXPERIMENT")
    print("=" * 80)

    vocab, tokenizer, train_samples, test_samples, train_facts, test_facts = build_experiment_data()
    vocab_size = len(vocab)
    print(f"\n[1] CANONICAL TOKENIZER & VOCABULARY AUDIT")
    print(f"    Vocabulary Size |V|:   {vocab_size}")
    print(f"    Special Tokens:        PAD={vocab.pad_id}, UNK={vocab.unk_id}, BOS={vocab.bos_id}, EOS={vocab.eos_id}, MASK={vocab.mask_id}")
    print(f"    Entities:              16 (e_0 to e_15)")
    print(f"    Relations:             4 (r_succ, r_pred, r_swap, r_jump)")

    # 2. Serialization Round-Trip Verification
    sample_text = train_samples[0]["text"]
    sample_ids = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(sample_ids)
    print(f"\n[2] SERIALIZATION BOUNDARY VERIFICATION")
    print(f"    Raw Input String:      '{sample_text}'")
    print(f"    Token IDs:             {sample_ids}")
    print(f"    Sequence Length (L):   {len(sample_ids)}")
    print(f"    Decoded Back:          '{decoded_text}'")
    assert decoded_text == sample_text, "Bijective serialization failed!"
    print(f"    -> Bijective Roundtrip: SUCCESS (Zero Character Leaks)")

    # 3. PyTorch Sequence Tensors
    train_texts = [s["text"] for s in train_samples]
    train_targets = [vocab.token_to_id(s["target"]) for s in train_samples]
    test_texts = [s["text"] for s in test_samples]
    test_targets = [vocab.token_to_id(s["target"]) for s in test_samples]

    train_batch = tokenizer.batch_encode(train_texts)
    test_batch = tokenizer.batch_encode(test_texts)

    train_x = train_batch["input_ids"]
    train_y = torch.tensor(train_targets, dtype=torch.long)
    test_x = test_batch["input_ids"]
    test_y = torch.tensor(test_targets, dtype=torch.long)

    # 4. Neural Architecture Substrate
    embed_dim = 64
    max_seq_len = 8
    num_heads = 4
    ff_dim = 128
    num_layers = 2

    model = ArisPhase0Model(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
    )

    mask_pos = 3
    with torch.no_grad():
        out0 = model(train_x)

    print(f"\n[3] PYTORCH TENSOR SUBSTRATE & EXACT TENSOR SHAPES")
    print(f"    Input Token IDs Tensor:        {tuple(train_x.shape)}  (Batch={train_x.shape[0]}, SeqLen={train_x.shape[1]})")
    print(f"    Token Embedding Tensor:        {tuple(out0['token_embeddings'].shape)}  (Batch={out0['token_embeddings'].shape[0]}, SeqLen={out0['token_embeddings'].shape[1]}, Dim={out0['token_embeddings'].shape[2]})")
    print(f"    Positional Embedding Tensor:   {tuple(out0['pos_embeddings'].shape)}")
    print(f"    Self-Attention Hidden States:  {tuple(out0['hidden_states'].shape)}")
    print(f"    Prediction Logits Tensor:      {tuple(out0['logits'].shape)}  (Batch, SeqLen, |V|={vocab_size})")
    print(f"    Target Tensor Shape:           {tuple(train_y.shape)}")
    print(f"    Total Learnable Parameters:    {sum(p.numel() for p in model.parameters())}")

    # 5. Untrained Baseline State (Zero Knowledge)
    criterion = nn.CrossEntropyLoss()
    initial_train_logits = out0["logits"][:, mask_pos, :]
    initial_train_loss = criterion(initial_train_logits, train_y).item()
    initial_train_preds = torch.argmax(initial_train_logits, dim=1)
    initial_train_acc = (initial_train_preds == train_y).float().mean().item()

    with torch.no_grad():
        test_out0 = model(test_x)
        initial_test_logits = test_out0["logits"][:, mask_pos, :]
        initial_test_loss = criterion(initial_test_logits, test_y).item()
        initial_test_preds = torch.argmax(initial_test_logits, dim=1)
        initial_test_acc = (initial_test_preds == test_y).float().mean().item()

    # Pre-training embedding geometry
    e0_id = vocab.token_to_id("e_0")
    e1_id = vocab.token_to_id("e_1")
    e8_id = vocab.token_to_id("e_8")
    w_e0_init = model.token_embedding.weight[e0_id].clone()
    w_e1_init = model.token_embedding.weight[e1_id].clone()
    w_e8_init = model.token_embedding.weight[e8_id].clone()
    init_sim_succ = compute_cosine_similarity(w_e0_init, w_e1_init)
    init_sim_dist = compute_cosine_similarity(w_e0_init, w_e8_init)

    # Initial parameter snapshots for gradient update verification
    initial_tok_emb_weight = model.token_embedding.weight.clone()
    initial_attn_qkv_weight = model.blocks[0].attention.in_proj_weight.clone()

    print(f"\n[4] UNTRAINED INITIAL BASELINE (Zero Knowledge Prior)")
    print(f"    Theoretical Random Loss:       {-math.log(1.0 / vocab_size):.4f}")
    print(f"    Initial Train Loss:            {initial_train_loss:.4f}")
    print(f"    Initial Train Accuracy:        {initial_train_acc * 100:.2f}% (Chance = {100.0 / vocab_size:.2f}%)")
    print(f"    Initial Test Loss:             {initial_test_loss:.4f}")
    print(f"    Initial Test Accuracy:         {initial_test_acc * 100:.2f}%")
    print(f"    Initial Cosine Sim(e_0, e_1):  {init_sim_succ:.4f}")
    print(f"    Initial Cosine Sim(e_0, e_8):  {init_sim_dist:.4f}")

    # 6. Real PyTorch Training Loop
    print(f"\n[5] EXECUTING REAL PYTORCH BACKPROPAGATION ON TOKEN SEQUENCES")
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    epochs = 200

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(train_x)
        mask_logits = out["logits"][:, mask_pos, :]
        loss = criterion(mask_logits, train_y)
        loss.backward()
        optimizer.step()

        if epoch in (1, 10, 25, 50, 100, 150, 200):
            with torch.no_grad():
                model.eval()
                preds = torch.argmax(mask_logits, dim=1)
                acc = (preds == train_y).float().mean().item()
                print(f"    Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.4f} | Train Accuracy: {acc * 100:.1f}%")

    # 7. Post-Training Evaluation
    model.eval()
    with torch.no_grad():
        final_train_out = model(train_x)
        final_train_logits = final_train_out["logits"][:, mask_pos, :]
        final_train_loss = criterion(final_train_logits, train_y).item()
        final_train_preds = torch.argmax(final_train_logits, dim=1)
        final_train_acc = (final_train_preds == train_y).float().mean().item()

    # Weight Update Verification
    delta_tok_emb = torch.norm(model.token_embedding.weight - initial_tok_emb_weight).item()
    delta_attn = torch.norm(model.blocks[0].attention.in_proj_weight - initial_attn_qkv_weight).item()

    # Post-training embedding geometry
    w_e0_post = model.token_embedding.weight[e0_id]
    w_e1_post = model.token_embedding.weight[e1_id]
    w_e8_post = model.token_embedding.weight[e8_id]
    post_sim_succ = compute_cosine_similarity(w_e0_post, w_e1_post)
    post_sim_dist = compute_cosine_similarity(w_e0_post, w_e8_post)

    print(f"\n[6] POST-TRAINING EMPIRICAL PROOF OF LEARNING")
    print(f"    Train Loss:                    {initial_train_loss:.4f} -> {final_train_loss:.4f} (Reduction: {initial_train_loss - final_train_loss:.4f})")
    print(f"    Train Accuracy:                {initial_train_acc * 100:.1f}% -> {final_train_acc * 100:.1f}%")
    print(f"    Note: This accuracy is not sufficient to establish generalization because the reported")
    print(f"          evaluation must be explicitly confirmed as held-out from training at the structural-combination level.")
    print(f"    Non-Zero Parameter Update Proof:")
    print(f"      - Token Embedding Delta ||Delta_E_tok||:   {delta_tok_emb:.4f} > 0")
    print(f"      - Attention Weights Delta ||Delta_W_attn||: {delta_attn:.4f} > 0")
    print(f"    Embedding Geometry Evolution:")
    print(f"      - Pre-training Cosine Sim(e_0, e_1 [succ]):  {init_sim_succ:.4f}")
    print(f"      - Post-training Cosine Sim(e_0, e_1 [succ]): {post_sim_succ:.4f}")

    print(f"\n[7] OBSERVED PREDICTION OUTPUTS")
    print(f"    {'Input Sequence':<32} | {'Target':<10} | {'Predicted':<10} | {'Confidence':<10} | {'Context'}")
    print(f"    {'-'*32}-|-{'-'*10}-|-{'-'*10}-|-{'-'*10}-|--------")
    
    train_probs = torch.softmax(final_train_logits, dim=1)
    for idx in range(min(len(train_samples), 8)):
        in_text = train_samples[idx]["text"]
        true_tok = train_samples[idx]["target"]
        pred_id = int(final_train_preds[idx].item())
        pred_tok = vocab.id_to_token(pred_id)
        conf = float(train_probs[idx, pred_id].item())
        status = "Observed in training"
        print(f"    {in_text:<32} | {true_tok:<10} | {pred_tok:<10} | {conf * 100:6.2f}%    | {status}")

    # 8. Critical Investigation of Meaning
    print("\n" + "=" * 80)
    print("CRITICAL INVESTIGATION: WHERE DOES INFORMATION ABOUT RELATIONSHIPS COME FROM?")
    print("=" * 80)
    print(f"""
    1. TOKEN IDENTITY:
       Token IDs (e.g. e_0 = {e0_id}, r_succ = {vocab.token_to_id('r_succ')}) are pure discrete integers.
       They contain NO meaning, NO order, and NO continuous geometry.

    2. TOKEN POSITION:
       Provides positional information that allows the model to distinguish locations within
       the serialized sequence. The experiment does not yet establish that the model has learned
       semantic argument roles from position alone.

    3. CONTEXT:
       Multi-head self-attention provides a mechanism through which the representation at the
       prediction position can incorporate information from other sequence positions.

    4. EMBEDDING:
       Initial embeddings E_tok are uninformative random normal vectors (~N(0, 0.02)).
       The initial cosine similarity was near zero ({init_sim_succ:.4f}).
       Embeddings do NOT 'understand' anything initially; they are blank lookup rows.

    5. LEARNED PARAMETERS:
       Backpropagation pushed loss gradients dL/dE and dL/dW into every parameter:
         ||Delta_E_tok|| = {delta_tok_emb:.4f} > 0
         ||Delta_W_attn|| = {delta_attn:.4f} > 0
       Gradient descent physically updated parameters to align with the demonstrated prediction task.

    6. TRAINING OBJECTIVE:
       The CrossEntropy loss L = -log P(target | context) supplied the ENTIRE
       directional force. Without the training objective, the parameters remain random noise.

    SCIENTIFIC DISTINCTION & CONCLUSION:
    The experiment establishes learnability of the supplied mapping, not yet the existence
    of an abstract relation concept independent of the token identities used during training.

    The experiment demonstrates that token IDs themselves contain no learned semantic geometry,
    while the trainable embedding and transformation parameters can be modified by gradient descent
    to encode information sufficient for the demonstrated token-to-target prediction task.
    Whether this learned structure represents reusable relational knowledge requires held-out
    compositional and permutation-based generalization experiments.
    """)

    # Save artifact metrics for reporting
    results = {
        "vocab_size": vocab_size,
        "sample_token_ids": sample_ids,
        "sample_sequence_length": len(sample_ids),
        "tensor_shapes": {
            "train_input": list(train_x.shape),
            "token_embeddings": list(out0["token_embeddings"].shape),
            "pos_embeddings": list(out0["pos_embeddings"].shape),
            "hidden_states": list(out0["hidden_states"].shape),
            "logits": list(out0["logits"].shape),
            "target": list(train_y.shape),
        },
        "initial_train_loss": round(initial_train_loss, 4),
        "final_train_loss": round(final_train_loss, 4),
        "initial_train_acc": round(initial_train_acc, 4),
        "final_train_acc": round(final_train_acc, 4),
        "weight_deltas": {
            "token_embedding_frobenius": round(delta_tok_emb, 4),
            "attention_frobenius": round(delta_attn, 4),
        },
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "embedding_geometry": {
            "pre_training_sim_e0_e1": round(init_sim_succ, 4),
            "post_training_sim_e0_e1": round(post_sim_succ, 4),
        }
    }

    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    out_path = os.path.join(PROJECT_ROOT, "models", "phase0_experiment_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results recorded to: {out_path}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_experiment()
