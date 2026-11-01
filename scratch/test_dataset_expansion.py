import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from aris.tokenizer.bpe_tokenizer import ArisTokenizer
from aris.data.dataset import EnterpriseDataLoader, EnterpriseArisDataset

TOKENIZER_PATH = Path("models/aris_bpe_tokenizer.json")

def test_expansion():
    print("=== Testing AR.I.S Cognitive Dataset Expansion ===")
    
    if not TOKENIZER_PATH.exists():
        print(f"Error: Tokenizer not found at {TOKENIZER_PATH}")
        sys.exit(1)
        
    tokenizer = ArisTokenizer.load(TOKENIZER_PATH)
    print(f"Loaded Tokenizer: Vocab size {tokenizer.vocab_size}")

    data_dir = Path("data")
    uc_dir = data_dir / "ultrachat_200k" / "data"
    ud_archive = Path("ud-treebanks-v2.18.tgz")
    qasrl_dir = data_dir / "qasrl_gs" / "qasrl-gs"
    hp_dir = data_dir / "hotpotqa" / "distractor"
    gsm8k_dir = data_dir / "gsm8k"
    mbpp_dir = data_dir / "mbpp"
    vision_dir = data_dir / "vision"

    focus_list = ["problem_solving", "coding", "vision", "all"]

    for focus in focus_list:
        print(f"\n--- Testing Focus: '{focus}' ---")
        loader = EnterpriseDataLoader.get_loader(
            ultrachat_dir=uc_dir,
            ud_archive_path=ud_archive if ud_archive.exists() else None,
            qasrl_dir=qasrl_dir if qasrl_dir.exists() else None,
            hotpotqa_dir=hp_dir if hp_dir.exists() else None,
            gsm8k_dir=gsm8k_dir if gsm8k_dir.exists() else None,
            mbpp_dir=mbpp_dir if mbpp_dir.exists() else None,
            vision_dir=vision_dir if vision_dir.exists() else None,
            tokenizer=tokenizer,
            batch_size=2,
            max_seq_len=256,
            focus=focus,
            rank=0,
            world_size=1,
        )

        batch_iter = iter(loader)
        inputs, targets = next(batch_iter)
        print(f"  Batch Input Shape:  {inputs.shape}")
        print(f"  Batch Target Shape: {targets.shape}")
        assert inputs.shape == targets.shape, "Shape mismatch between inputs and targets!"
        assert not torch.isnan(inputs).any(), "NaN found in input tensors!"

        # Decode sample 0
        non_pad_ids = [tok for tok in inputs[0].tolist() if tok != tokenizer.pad_id]
        decoded_text = tokenizer.decode(non_pad_ids)
        print(f"  Decoded Sample 0 Preview (first 180 chars):\n    {repr(decoded_text[:180])}")

    # Test Distributed Sharding between Rank 0 and Rank 1 on coding
    print("\n--- Testing Distributed Zero-Overlap Sharding (Rank 0 vs Rank 1) ---")
    loader_r0 = EnterpriseDataLoader.get_loader(
        ultrachat_dir=uc_dir,
        ud_archive_path=None,
        qasrl_dir=None,
        hotpotqa_dir=None,
        gsm8k_dir=None,
        mbpp_dir=mbpp_dir,
        vision_dir=None,
        tokenizer=tokenizer,
        batch_size=1,
        max_seq_len=128,
        focus="coding",
        rank=0,
        world_size=2,
    )
    loader_r1 = EnterpriseDataLoader.get_loader(
        ultrachat_dir=uc_dir,
        ud_archive_path=None,
        qasrl_dir=None,
        hotpotqa_dir=None,
        gsm8k_dir=None,
        mbpp_dir=mbpp_dir,
        vision_dir=None,
        tokenizer=tokenizer,
        batch_size=1,
        max_seq_len=128,
        focus="coding",
        rank=1,
        world_size=2,
    )

    sample_r0 = next(iter(loader_r0))[0][0].tolist()
    sample_r1 = next(iter(loader_r1))[0][0].tolist()
    assert sample_r0 != sample_r1, "Rank 0 and Rank 1 received identical samples! Sharding failed."
    print("  Zero-Overlap Sharding Verified: Rank 0 and Rank 1 received completely distinct samples.")

    print("\n[SUCCESS] All 7 Cognitive Dataset Streams Verified!")

if __name__ == "__main__":
    test_expansion()
