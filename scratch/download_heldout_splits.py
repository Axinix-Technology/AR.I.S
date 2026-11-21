"""
Download Held-Out Evaluation Splits (Fast, < 15 Seconds):
1. GSM8K test: openai/gsm8k main/test-00000-of-00001.parquet -> data/gsm8k/test.parquet (~490 KB)
2. SQuAD v2 validation: rajpurkar/squad_v2 squad_v2/validation-00000-of-00001.parquet -> data/squad_v2/validation.parquet (~4.3 MB)
3. MBPP test: google-research-datasets/mbpp full/test-00000-of-00001.parquet -> data/mbpp/test.parquet (~85 KB)
4. HotpotQA distractor validation: hotpotqa/hotpot_qa distractor/validation-00000-of-00001.parquet -> data/hotpotqa/distractor/validation.parquet (~5.4 MB)
"""

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")

from huggingface_hub import hf_hub_download

DATA_DIR = BASE_DIR / "data"

def download_all_heldout_splits():
    splits = [
        ("GSM8K test", DATA_DIR / "gsm8k/test.parquet", "openai/gsm8k", "main/test-00000-of-00001.parquet"),
        ("SQuAD v2 validation", DATA_DIR / "squad_v2/validation.parquet", "rajpurkar/squad_v2", "squad_v2/validation-00000-of-00001.parquet"),
        ("MBPP test", DATA_DIR / "mbpp/test.parquet", "google-research-datasets/mbpp", "full/test-00000-of-00001.parquet"),
        ("HotpotQA validation", DATA_DIR / "hotpotqa/distractor/validation.parquet", "hotpotqa/hotpot_qa", "distractor/validation-00000-of-00001.parquet"),
    ]

    for label, dest, repo_id, filename in splits:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            print(f"Downloading {label} ({repo_id})...")
            src = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", token=HF_TOKEN)
            shutil.copyfile(src, dest)
            print(f"  {label} saved: {dest} ({dest.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  {label} already present: {dest}")

if __name__ == "__main__":
    download_all_heldout_splits()
