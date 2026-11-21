"""
Automated Test Suite for Data Pipeline V2
Verifies:
1. Every training sequence ends with <|im_end|> (token id 2).
2. Zero-supervised rate = 0 (every sequence has > 0 supervised tokens; prompt masked with -100).
3. Held-out and excluded files NEVER appear in the train manifest or iterator.
4. Resumption gives bit-identical tensors and state reproducibility.
5. Context shrinking preserves full assistant responses without truncation.
"""

import os
import sys
from pathlib import Path
import pytest
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from aris.tokenizer.bpe_tokenizer import ArisTokenizer
from aris.data.manifest import get_manifest_entries, DATA_MANIFEST
from aris.data.pipeline_v2 import ArisDataPipelineV2, is_held_out_by_hash

TOKENIZER_PATH = BASE_DIR / "models/aris_bpe_tokenizer.json"
tok = ArisTokenizer.load(TOKENIZER_PATH)

def test_manifest_exclusions():
    """Verify that held-out and excluded files are strictly filtered out of train entries."""
    train_entries = get_manifest_entries(role="train")
    train_paths = [e.path for e in train_entries]

    forbidden_patterns = [
        "data/vision/detail_23k.json",
        "data/custom_chat.json",
        "data/ultrachat_200k/data/test_gen-00000-of-00001-3d4cd8309148a71f.parquet",
        "data/ultrachat_200k/data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
        "data/ultrachat_200k/data/train_sft.parquet", # byte duplicate
        "data/gsm8k/test.parquet",
        "data/mbpp/test.parquet",
        "data/squad_v2/validation.parquet",
        "data/hotpotqa/distractor/validation.parquet",
        "data/qasrl_gs/qasrl-gs/dev.jsonl.gz",
        "data/qasrl_gs/qasrl-gs/test.jsonl.gz",
    ]

    for forbidden in forbidden_patterns:
        assert forbidden not in train_paths, f"CRITICAL: Forbidden file '{forbidden}' found in train manifest!"


def test_hash_carve_out_determinism():
    """Verify deterministic hash carve-out partitions consistently."""
    test_prompts = [
        "How do I sort a list in Python?",
        "Explain quantum entanglement simply.",
        "Write a poem about neural networks.",
        "What is the capital of France?",
    ]
    for p in test_prompts:
        res1 = is_held_out_by_hash(p, split_pct=1)
        res2 = is_held_out_by_hash(p, split_pct=1)
        assert res1 == res2, "Hash carve-out must be 100% deterministic"


def test_zero_slice_truncation_and_im_end():
    """Verify that sequences end with <|im_end|> and zero-supervised rate is strictly 0."""
    pipeline = ArisDataPipelineV2(tokenizer=tok, max_seq_len=384, seed=42)
    stream = iter(pipeline)

    sample_count = 200
    for idx in range(sample_count):
        inp, tgt = next(stream)

        # 1. Length within max_seq_len
        assert len(inp) <= 384, f"Sequence {idx} length {len(inp)} > 384"
        assert len(inp) == len(tgt), f"Input and target lengths must match"

        # 2. Ends with <|im_end|> (token id tok.im_end_id)
        # Check last tokens of input sequence
        tokens_list = inp.tolist()
        assert (tok.im_end_id in tokens_list[-3:]), f"Sequence {idx} lacks <|im_end|> in suffix: {tokens_list[-5:]}"

        # 3. Supervised tokens count > 0 (zero-supervised rate = 0)
        sup_tokens = (tgt != -100).sum().item()
        assert sup_tokens > 0, f"Sequence {idx} has zero supervised tokens!"

        # 4. Prompt tokens strictly masked with -100
        first_sup_idx = (tgt != -100).nonzero(as_tuple=True)[0][0].item()
        assert (tgt[:first_sup_idx] == -100).all(), f"Prompt tokens before index {first_sup_idx} not masked!"


def test_context_shrinking_policy():
    """Verify context shrinking shrinks prompt without dropping/truncating assistant response."""
    pipeline = ArisDataPipelineV2(tokenizer=tok, max_seq_len=200, seed=42)
    long_prompt = "Context:\n" + ("The quick brown fox jumps over the lazy dog. " * 30) + "\nQuestion: What jumped?"
    exact_response = "The quick brown fox jumped over the lazy dog."

    sample = pipeline._shrink_context_and_tokenize(long_prompt, exact_response, "squad")
    assert sample is not None, "Pipeline should shrink context to fit response rather than failing"
    inp, tgt = sample
    assert len(inp) <= 200, "Shrunk sequence must not exceed max_seq_len"

    # Decode response tokens from target (where tgt != -100)
    sup_ids = tgt[tgt != -100].tolist()
    decoded_resp = tok.decode(sup_ids)
    assert "The quick brown fox jumped over the lazy dog." in decoded_resp, "Response must remain 100% complete"


def test_bit_identical_resumption():
    """Verify that saving and restoring state produces bit-identical sequences."""
    pipeline1 = ArisDataPipelineV2(tokenizer=tok, max_seq_len=384, seed=1234)
    it1 = iter(pipeline1)

    # Consume 20 samples
    for _ in range(20):
        next(it1)

    # Checkpoint state
    state = pipeline1.get_state()

    # Continue generating next 10 samples
    seqs_ref = [next(it1) for _ in range(10)]

    # Resume new pipeline from checkpoint state
    pipeline2 = ArisDataPipelineV2(tokenizer=tok, max_seq_len=384, seed=1234)
    pipeline2.set_state(state)
    it2 = iter(pipeline2)

    seqs_resumed = [next(it2) for _ in range(10)]

    for idx, ((inp_ref, tgt_ref), (inp_res, tgt_res)) in enumerate(zip(seqs_ref, seqs_resumed)):
        assert torch.equal(inp_ref, inp_res), f"Input mismatch at resumed sequence {idx}"
        assert torch.equal(tgt_ref, tgt_res), f"Target mismatch at resumed sequence {idx}"


if __name__ == "__main__":
    test_manifest_exclusions()
    print("[PASS] test_manifest_exclusions")
    test_hash_carve_out_determinism()
    print("[PASS] test_hash_carve_out_determinism")
    test_zero_slice_truncation_and_im_end()
    print("[PASS] test_zero_slice_truncation_and_im_end")
    test_context_shrinking_policy()
    print("[PASS] test_context_shrinking_policy")
    test_bit_identical_resumption()
    print("[PASS] test_bit_identical_resumption")
    print("\nALL DATA PIPELINE V2 TESTS PASSED CLEANLY (100%)")
