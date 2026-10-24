"""
AR.I.S — Contextual Token Representation Experiment

Tests the single scientific question:
Does AR.I.S produce a different representation for the same token when its surrounding language context changes?

Strict Invariants:
- Uses existing ArisCanonicalTokenizer and ArisPhase0Model.
- Zero python heuristics or mock orchestration.
- Raw text -> Tokenizer -> Token IDs -> PyTorch Substrate -> Hidden states -> Token representation.
"""

import os
import sys
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.linguistic.canonical_tokenizer import Vocabulary, ArisCanonicalTokenizer
from api.linguistic.phase0_substrate import ArisPhase0Model


def compute_cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    a = vec_a.flatten().float()
    b = vec_b.flatten().float()
    dot = torch.dot(a, b)
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float((dot / (norm_a * norm_b)).item())


def compute_euclidean_distance(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    a = vec_a.flatten().float()
    b = vec_b.flatten().float()
    return float(torch.norm(a - b).item())


def locate_token(tokens, target="apple"):
    for idx, tok in enumerate(tokens):
        if tok.lower() == target.lower():
            return idx, tok
    raise ValueError(f"Target token '{target}' not found in tokens: {tokens}")


def run_experiment():
    torch.manual_seed(42)

    # 1. Sentences used in the benchmark
    sentences_pool = [
        "I ate an apple.",
        "Apple released a phone.",
        "The apple fell from the tree.",
        "She bought an apple.",
        "I bought an apple.",
        "I dropped an apple.",
        "I peeled an apple.",
    ]

    # Build canonical vocabulary from the input sentences
    raw_tokenizer = ArisCanonicalTokenizer()
    vocab = Vocabulary()
    for s in sentences_pool:
        for token in raw_tokenizer.tokenize_raw(s):
            vocab.add_token(token)

    tokenizer = ArisCanonicalTokenizer(vocab)

    # 2. Existing Phase 0 Substrate
    embed_dim = 64
    max_seq_len = 32
    num_heads = 2
    ff_dim = 128
    num_layers = 2

    model = ArisPhase0Model(
        vocab_size=len(tokenizer.vocab),
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=0.0,
    )
    model.eval()

    def extract_token_rep(text: str, target: str = "apple"):
        seq = tokenizer.serialize(text, add_special_tokens=True)
        pos, matched_tok = locate_token(seq.tokens, target)
        tid = int(seq.input_ids[0, pos].item())
        with torch.no_grad():
            out = model(seq.input_ids, attention_mask=seq.attention_mask)
            # Hidden states shape: (1, seq_len, embed_dim)
            hs = out["hidden_states"]
            vec = hs[0, pos, :].clone()
        return {
            "sentence": text,
            "token": matched_tok,
            "token_id": tid,
            "token_pos": pos,
            "hidden_dim": list(vec.shape),
            "norm": float(torch.norm(vec).item()),
            "vector": vec,
            "tokens": seq.tokens,
            "input_ids": seq.input_ids.tolist()[0],
        }

    # -------------------------------------------------------------
    # EXPERIMENT 1: Same sentence, same token
    # -------------------------------------------------------------
    exp1_a = extract_token_rep("I ate an apple.")
    exp1_b = extract_token_rep("I ate an apple.")
    cos_1 = compute_cosine_similarity(exp1_a["vector"], exp1_b["vector"])
    euc_1 = compute_euclidean_distance(exp1_a["vector"], exp1_b["vector"])

    # -------------------------------------------------------------
    # EXPERIMENT 2: Same token, different context
    # -------------------------------------------------------------
    exp2_a = extract_token_rep("I ate an apple.")
    exp2_b = extract_token_rep("Apple released a phone.")
    cos_2 = compute_cosine_similarity(exp2_a["vector"], exp2_b["vector"])
    euc_2 = compute_euclidean_distance(exp2_a["vector"], exp2_b["vector"])

    # -------------------------------------------------------------
    # EXPERIMENT 3: Same word, different context (pairwise matrix)
    # -------------------------------------------------------------
    exp3_cases = [
        ("A", "I ate an apple."),
        ("B", "Apple released a phone."),
        ("C", "The apple fell from the tree."),
        ("D", "She bought an apple."),
    ]
    exp3_reps = {lbl: extract_token_rep(txt) for lbl, txt in exp3_cases}
    labels = ["A", "B", "C", "D"]
    matrix = {l1: {l2: compute_cosine_similarity(exp3_reps[l1]["vector"], exp3_reps[l2]["vector"]) for l2 in labels} for l1 in labels}

    # -------------------------------------------------------------
    # EXPERIMENT 4: Context sensitivity across positions
    # -------------------------------------------------------------
    exp4_a = extract_token_rep("Apple released a phone.")
    exp4_b = extract_token_rep("I ate an apple.")

    # -------------------------------------------------------------
    # EXPERIMENT 5: Context perturbation
    # -------------------------------------------------------------
    exp5_sentences = [
        "I ate an apple.",
        "I bought an apple.",
        "I dropped an apple.",
        "I peeled an apple.",
    ]
    exp5_reps = [extract_token_rep(s) for s in exp5_sentences]
    base_rep = exp5_reps[0]

    # Print Clean Scientific Report
    print("============================================================")
    print("AR.I.S CONTEXTUAL REPRESENTATION EXPERIMENT")
    print("============================================================")
    print()
    print("Model: ArisPhase0Model (PyTorch TransformerEncoder, 2 layers, 2 heads)")
    print(f"Tokenizer: ArisCanonicalTokenizer (Vocabulary size: {len(tokenizer.vocab)})")
    print(f"Hidden dimension: {embed_dim}")
    print()
    print("Experiment 1:")
    print("Same sentence / same token")
    print(f"Cosine similarity: {cos_1:.6f}")
    print(f"Euclidean distance: {euc_1:.6f}")
    print()
    print("Experiment 2:")
    print("Same token / different context")
    print(f"Cosine similarity: {cos_2:.6f}")
    print(f"Euclidean distance: {euc_2:.6f}")
    print()
    print("Experiment 3:")
    print("Pairwise similarity matrix:")
    print(f"          {'A':<8}{'B':<8}{'C':<8}{'D':<8}")
    for r in labels:
        row_str = "".join(f"{matrix[r][c]:<8.4f}" for c in labels)
        print(f"{r:<10}{row_str}")
    print()
    print("Experiment 4:")
    print("Token-location verification:")
    for item in [exp4_a, exp4_b]:
        print(f"  Sentence:        \"{item['sentence']}\"")
        print(f"  Token:           \"{item['token']}\"")
        print(f"  Token ID:        {item['token_id']}")
        print(f"  Token position:  {item['token_pos']} in sequence {item['tokens']}")
        print(f"  Hidden dimension: {item['hidden_dim']}")
        print(f"  Hidden vector norm: {item['norm']:.4f}")
        print()
    print("Experiment 5:")
    print("Context perturbation results:")
    for rep in exp5_reps:
        c_sim = compute_cosine_similarity(base_rep["vector"], rep["vector"])
        e_dist = compute_euclidean_distance(base_rep["vector"], rep["vector"])
        print(f"  Sentence: \"{rep['sentence']}\"")
        print(f"    Token: \"{rep['token']}\" (pos: {rep['token_pos']})")
        print(f"    Cosine similarity to base (\"I ate an apple.\"): {c_sim:.6f}")
        print(f"    Euclidean distance to base: {e_dist:.6f}")
    print()
    print("============================================================")


if __name__ == "__main__":
    run_experiment()
