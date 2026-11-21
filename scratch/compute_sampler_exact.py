"""
Exact empirical counter for post-truncation non-pad and supervised tokens.
Iterates each corpus iterator from EnterpriseArisDataset with max_seq_len=384
(the exact training setup), measuring:
- Post-truncation non-pad tokens per sequence (len(inp), strictly <= 384)
- Post-truncation supervised tokens per sequence ((tgt != -100).sum())
- Exact sequences drawn, non-pad tokens, supervised tokens, and epochs for:
  - Step 16,000 (256,000 sequences)
  - Step 21,250 (340,000 sequences)
"""

import sys
import json
import random
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aris.tokenizer.bpe_tokenizer import ArisTokenizer
from aris.data.dataset import EnterpriseArisDataset

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
TOKENIZER_PATH = MODELS_DIR / "aris_bpe_tokenizer.json"

def measure_corpus_tokens():
    tok = ArisTokenizer.load(TOKENIZER_PATH)
    
    # Initialize with the EXACT training max_seq_len=384
    ds = EnterpriseArisDataset(
        ultrachat_dir=DATA_DIR / "ultrachat_200k/data",
        ud_archive_path=Path("ud-treebanks-v2.18.tgz"),
        qasrl_dir=DATA_DIR / "qasrl_gs/qasrl-gs",
        hotpotqa_dir=DATA_DIR / "hotpotqa/distractor",
        gsm8k_dir=DATA_DIR / "gsm8k",
        mbpp_dir=DATA_DIR / "mbpp",
        vision_dir=DATA_DIR / "vision",
        squad_dir=DATA_DIR / "squad_v2",
        dolly_dir=DATA_DIR / "dolly",
        smoltalk_dir=DATA_DIR / "smoltalk",
        custom_chat_path=DATA_DIR / "custom_chat.json",
        tokenizer=tok,
        max_seq_len=384,
        split="train",
        focus="all",
    )
    
    iter_funcs = {
        "SQuAD 2.0": (ds._iter_squad, 2000, 130319),
        "Dolly-15k": (ds._iter_dolly, 2000, 15011),
        "SmolTalk": (ds._iter_smoltalk, 2000, 86320),
        "UltraChat 200k": (ds._iter_ultrachat, 2000, 718290),
        "UD Treebanks v2.18": (ds._iter_ud, 2000, 169904),
        "HotpotQA": (ds._iter_hotpotqa, 2000, 90447),
        "QA-SRL 2020": (ds._iter_qasrl, 1999, 1999),
        "GSM8K": (ds._iter_gsm8k, 2000, 7473),
        "MBPP": (ds._iter_mbpp, 884, 884),
        "Vision (detail_23k)": (ds._iter_vision, 2000, 23240),
        "Custom Chat": (ds._iter_custom_chat, 2, 2),
    }

    slots = {
        "SQuAD 2.0": 2,
        "Dolly-15k": 2,
        "SmolTalk": 2,
        "UltraChat 200k": 2,
        "UD Treebanks v2.18": 1,
        "HotpotQA": 1,
        "QA-SRL 2020": 1,
        "GSM8K": 2,
        "MBPP": 2,
        "Vision (detail_23k)": 2,
        "Custom Chat": 2,
    }
    cycle_total = sum(slots.values()) # 19
    
    random.seed(42)
    torch.manual_seed(42)
    
    measured_stats = {}
    print("="*80)
    print("MEASURING EMPIRICAL POST-TRUNCATION TOKENS (max_seq_len=384)")
    print("="*80)
    print(f"{'Corpus':20} | {'Sampled':7} | {'Mean Non-Pad (<=384)':21} | {'Mean Supervised':15} | {'Zero-Sup %':10}")
    print("-" * 80)
    
    for name, (fn, target_k, total_units) in iter_funcs.items():
        non_pad_lens = []
        sup_lens = []
        zero_sup_count = 0
        
        for i, (inp, tgt) in enumerate(fn()):
            # Post-truncation tensors:
            # inp is Tensor of shape (L,) where L <= 384
            # tgt is Tensor of shape (L,) where prompt has -100 and assistant has token IDs
            non_pad = len(inp)
            sup = (tgt != -100).sum().item()
            
            non_pad_lens.append(non_pad)
            sup_lens.append(sup)
            if sup == 0:
                zero_sup_count += 1
                
            if len(non_pad_lens) >= target_k:
                break
                
        mean_non_pad = float(np.mean(non_pad_lens))
        mean_sup = float(np.mean(sup_lens))
        pct_zero = (zero_sup_count / len(non_pad_lens)) * 100
        
        measured_stats[name] = {
            "mean_non_pad": mean_non_pad,
            "mean_sup": mean_sup,
            "units": total_units,
            "slots": slots[name],
        }
        print(f"{name:20} | {len(non_pad_lens):7} | {mean_non_pad:21.2f} | {mean_sup:15.2f} | {pct_zero:9.2f}%")

    # Generate tables for Step 16,000 and Step 21,250
    for step_num, total_seqs in [(16000, 256000), (21250, 340000)]:
        print("\n" + "="*95)
        print(f"SAMPLER TABLE FOR STEP {step_num:,} ({total_seqs:,} SEQUENCES DRAWN)")
        print("="*95)
        print(f"{'Corpus':20} | {'Slots':5} | {'Seq Drawn':10} | {'Non-Pad Tokens':15} | {'Supervised Toks':16} | {'Iterator Units':15} | {'Epochs':8}")
        print("-" * 95)
        
        cycles = total_seqs / cycle_total
        tot_non_pad = 0
        tot_sup = 0
        
        for name, data in measured_stats.items():
            s = data["slots"]
            seq_drawn = cycles * s
            non_pad_toks = seq_drawn * data["mean_non_pad"]
            sup_toks = seq_drawn * data["mean_sup"]
            units = data["units"]
            epochs = seq_drawn / units
            
            tot_non_pad += non_pad_toks
            tot_sup += sup_toks
            
            print(f"{name:20} | {s:2}/19 | {seq_drawn:10.0f} | {non_pad_toks:15,.0f} | {sup_toks:16,.0f} | {units:15,d} | {epochs:7.2f}x")
            
        print("-" * 95)
        print(f"{'TOTAL':20} | 19/19 | {total_seqs:10,d} | {tot_non_pad:15,.0f} | {tot_sup:16,.0f} |")

if __name__ == "__main__":
    measure_corpus_tokens()
