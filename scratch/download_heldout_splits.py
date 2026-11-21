"""
Download Held-Out Evaluation Splits:
1. GSM8K test: openai/gsm8k main/test-00000-of-00001.parquet -> data/gsm8k/test.parquet
2. SQuAD v2 validation: rajpurkar/squad_v2 squad_v2/validation-00000-of-00001.parquet -> data/squad_v2/validation.parquet
3. HotpotQA distractor validation: hotpotqa/hotpot_qa distractor/validation-00000-of-00001.parquet -> data/hotpotqa/distractor/validation.parquet
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

def download_gsm8k_test():
    dest_dir = DATA_DIR / "gsm8k"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "test.parquet"
    if not dest.exists():
        print("Downloading GSM8K test split...")
        src = hf_hub_download(repo_id="openai/gsm8k", filename="main/test-00000-of-00001.parquet", repo_type="dataset", token=HF_TOKEN)
        shutil.copyfile(src, dest)
        print(f"GSM8K test saved: {dest} ({dest.stat().st_size / 1024:.2f} KB)")
    else:
        print(f"GSM8K test already exists: {dest}")

def download_squad_validation():
    dest_dir = DATA_DIR / "squad_v2"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "validation.parquet"
    if not dest.exists():
        print("Downloading SQuAD v2 validation split...")
        src = hf_hub_download(repo_id="rajpurkar/squad_v2", filename="squad_v2/validation-00000-of-00001.parquet", repo_type="dataset", token=HF_TOKEN)
        shutil.copyfile(src, dest)
        print(f"SQuAD v2 validation saved: {dest} ({dest.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"SQuAD v2 validation already exists: {dest}")

def download_hotpotqa_validation():
    dest_dir = DATA_DIR / "hotpotqa" / "distractor"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "validation.parquet"
    if not dest.exists():
        print("Downloading HotpotQA validation split...")
        src = hf_hub_download(repo_id="hotpotqa/hotpot_qa", filename="distractor/validation-00000-of-00001.parquet", repo_type="dataset", token=HF_TOKEN)
        shutil.copyfile(src, dest)
        print(f"HotpotQA validation saved: {dest} ({dest.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"HotpotQA validation already exists: {dest}")

if __name__ == "__main__":
    download_gsm8k_test()
    download_squad_validation()
    download_hotpotqa_validation()
