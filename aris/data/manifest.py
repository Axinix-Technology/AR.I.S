"""
AR.I.S Data Manifest V2 (Strict Non-Globbed Inventory)
Defines every single data file with an explicit role:
  - 'train': Authorized for model training.
  - 'val': In-training validation loss evaluation.
  - 'held-out': Final developmental benchmark milestone evaluation (never trained on).
  - 'excluded': Known contaminated, unaligned, or duplicate files (strictly forbidden).
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass(frozen=True)
class ManifestEntry:
    path: str
    corpus: str
    role: str # 'train', 'val', 'held-out', 'excluded'
    format: str # 'parquet', 'jsonl', 'json', 'tgz'
    description: str

DATA_MANIFEST: List[ManifestEntry] = [
    # 1. GSM8K (OpenAI Math Problem Solving)
    ManifestEntry(
        path="data/gsm8k/train.parquet",
        corpus="gsm8k",
        role="train",
        format="parquet",
        description="GSM8K official training set (7,473 problems)",
    ),
    ManifestEntry(
        path="data/gsm8k/test.parquet",
        corpus="gsm8k",
        role="held-out",
        format="parquet",
        description="GSM8K official test set (1,319 problems)",
    ),

    # 2. MBPP (Google Research Python Coding)
    ManifestEntry(
        path="data/mbpp/train.parquet",
        corpus="mbpp",
        role="train",
        format="parquet",
        description="MBPP training split (374 coding tasks)",
    ),
    ManifestEntry(
        path="data/mbpp/prompt.parquet",
        corpus="mbpp",
        role="train",
        format="parquet",
        description="MBPP few-shot prompt split (10 tasks)",
    ),
    ManifestEntry(
        path="data/mbpp/test.parquet",
        corpus="mbpp",
        role="held-out",
        format="parquet",
        description="MBPP official test split (500 coding tasks)",
    ),

    # 3. HotpotQA (Multi-Hop Complex Reasoning)
    ManifestEntry(
        path="data/hotpotqa/distractor/train-00000-of-00002.parquet",
        corpus="hotpotqa",
        role="train",
        format="parquet",
        description="HotpotQA distractor training shard 0 (45,223 rows)",
    ),
    ManifestEntry(
        path="data/hotpotqa/distractor/train-00001-of-00002.parquet",
        corpus="hotpotqa",
        role="train",
        format="parquet",
        description="HotpotQA distractor training shard 1 (45,224 rows)",
    ),
    ManifestEntry(
        path="data/hotpotqa/distractor/validation.parquet",
        corpus="hotpotqa",
        role="held-out",
        format="parquet",
        description="HotpotQA official distractor validation set (7,405 rows)",
    ),

    # 4. Stanford SQuAD 2.0 (Reading Comprehension & QA)
    ManifestEntry(
        path="data/squad_v2/train.parquet",
        corpus="squad",
        role="train",
        format="parquet",
        description="SQuAD 2.0 training reading comprehension (130,319 rows)",
    ),
    ManifestEntry(
        path="data/squad_v2/validation.parquet",
        corpus="squad",
        role="held-out",
        format="parquet",
        description="SQuAD 2.0 official validation reading comprehension (11,873 rows)",
    ),

    # 5. Databricks Dolly-15k (Direct Human Instruction)
    # Uses deterministic hash carve-out: sha256(prompt) % 100 == 0 -> held-out (1%), else train (99%)
    ManifestEntry(
        path="data/dolly/databricks-dolly-15k.jsonl",
        corpus="dolly",
        role="train",
        format="jsonl",
        description="Dolly-15k instructions (99% train split, 1% hash carve-out held-out)",
    ),

    # 6. SmolTalk (Everyday Conversations & Magpie Ultra QA)
    # Uses deterministic hash carve-out: sha256(prompt) % 100 == 0 -> held-out (1%), else train (99%)
    ManifestEntry(
        path="data/smoltalk/everyday_conversations.parquet",
        corpus="smoltalk_everyday",
        role="train",
        format="parquet",
        description="SmolTalk everyday dialogue (99% train split, 1% hash carve-out)",
    ),
    ManifestEntry(
        path="data/smoltalk/smol_magpie_ultra.parquet",
        corpus="smoltalk_magpie",
        role="train",
        format="parquet",
        description="SmolTalk Magpie Ultra QA (99% train split, 1% hash carve-out)",
    ),
    ManifestEntry(
        path="data/smoltalk/openhermes_100k.parquet",
        corpus="smoltalk_openhermes",
        role="train",
        format="parquet",
        description="SmolTalk OpenHermes 100k diverse high-density QA & reasoning (99% train split, 1% hash carve-out)",
    ),
    ManifestEntry(
        path="data/smoltalk/numina_cot_100k.parquet",
        corpus="smoltalk_numina",
        role="train",
        format="parquet",
        description="SmolTalk NuminaMath CoT 100k step-by-step mathematical reasoning (99% train split, 1% hash carve-out)",
    ),

    # 7. QA-SRL 2020 (Semantic Causal Reasoning) - NO TRAINING SPLIT
    ManifestEntry(
        path="data/qasrl_gs/qasrl-gs/dev.jsonl.gz",
        corpus="qasrl",
        role="held-out",
        format="jsonl",
        description="QA-SRL 2020 gold standard dev set (Evaluation only)",
    ),
    ManifestEntry(
        path="data/qasrl_gs/qasrl-gs/test.jsonl.gz",
        corpus="qasrl",
        role="held-out",
        format="jsonl",
        description="QA-SRL 2020 gold standard test set (Evaluation only)",
    ),

    # 8. UltraChat Clean Training Shards (6 files, 463,897 rows)
    ManifestEntry(
        path="data/ultrachat_200k/data/train_gen-00000-of-00003-a6c9fb894be3e50b.parquet",
        corpus="ultrachat",
        role="train",
        format="parquet",
        description="UltraChat clean training generation shard 0 (85,344 rows)",
    ),
    ManifestEntry(
        path="data/ultrachat_200k/data/train_gen-00001-of-00003-d6a0402e417f35ca.parquet",
        corpus="ultrachat",
        role="train",
        format="parquet",
        description="UltraChat clean training generation shard 1 (85,344 rows)",
    ),
    ManifestEntry(
        path="data/ultrachat_200k/data/train_gen-00002-of-00003-c0db75b92a2f48fd.parquet",
        corpus="ultrachat",
        role="train",
        format="parquet",
        description="UltraChat clean training generation shard 2 (85,344 rows)",
    ),
    ManifestEntry(
        path="data/ultrachat_200k/data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
        corpus="ultrachat",
        role="train",
        format="parquet",
        description="UltraChat clean training SFT shard 0 (69,289 rows)",
    ),
    ManifestEntry(
        path="data/ultrachat_200k/data/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet",
        corpus="ultrachat",
        role="train",
        format="parquet",
        description="UltraChat clean training SFT shard 1 (69,288 rows)",
    ),
    ManifestEntry(
        path="data/ultrachat_200k/data/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet",
        corpus="ultrachat",
        role="train",
        format="parquet",
        description="UltraChat clean training SFT shard 2 (69,288 rows)",
    ),

    # 9. UltraChat Test & Duplicate Splits (HELD-OUT & EXCLUDED)
    ManifestEntry(
        path="data/ultrachat_200k/data/test_gen-00000-of-00001-3d4cd8309148a71f.parquet",
        corpus="ultrachat",
        role="held-out",
        format="parquet",
        description="UltraChat official held-out test generation split (28,304 rows)",
    ),
    ManifestEntry(
        path="data/ultrachat_200k/data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
        corpus="ultrachat",
        role="held-out",
        format="parquet",
        description="UltraChat official held-out test SFT split (23,110 rows)",
    ),
    ManifestEntry(
        path="data/ultrachat_200k/data/train_sft.parquet",
        corpus="ultrachat",
        role="excluded",
        format="parquet",
        description="Byte-for-byte exact duplicate of test_sft (23,110 rows); strictly excluded",
    ),

    # 10. Universal Dependencies v2.18 (Syntax Treebanks)
    ManifestEntry(
        path="ud-treebanks-v2.18.tgz",
        corpus="ud",
        role="train",
        format="tgz",
        description="Universal Dependencies v2.18 archive (STRICTLY *-train.conllu members only)",
    ),

    # 11. Excluded Multimodal & Synthetic Data
    ManifestEntry(
        path="data/vision/detail_23k.json",
        corpus="vision",
        role="excluded",
        format="json",
        description="Multimodal text without image embeddings; causes coordinate hallucinations; excluded",
    ),
    ManifestEntry(
        path="data/custom_chat.json",
        corpus="custom_chat",
        role="excluded",
        format="json",
        description="3 hardcoded persona dialogues with spelling errors; caused 8,982x overfit loop; excluded",
    ),
]

def get_manifest_entries(role: Optional[str] = None, corpus: Optional[str] = None) -> List[ManifestEntry]:
    """Returns filtered entries from the strict manifest without filesystem globbing."""
    entries = DATA_MANIFEST
    if role:
        entries = [e for e in entries if e.role == role]
    if corpus:
        entries = [e for e in entries if e.corpus == corpus]
    return entries
