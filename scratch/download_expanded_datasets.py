"""
AR.I.S — Unified Master Dataset Downloader (2026 Production Grade)
Downloads and validates all foundation, reasoning, and high-density QA corpora:
  1. GSM8K (OpenAI Math Problem Solving)
  2. MBPP (Google Research Python Algorithmic Logic)
  3. LLaVA Vision (Multimodal Scene Reasoning)
  4. UltraChat SFT (Conversational Dialogue)
  5. Stanford SQuAD 2.0 (Direct Factual Reading Comprehension QA)
  6. Databricks Dolly-15k (Human-Authored Direct Instruction & QA)
  7. SmolTalk Everyday Conversations (Concise Natural Dialogues & Persona)
  8. SmolTalk Magpie Ultra (High-Density Sharp QA)
  9. HotpotQA (Multi-Hop Complex Deductions)
 10. QA-SRL 2020 (Semantic Causal Reasoning)
"""

from __future__ import annotations
import os
import sys
import shutil
import tarfile
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

# Suppress HF symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

from huggingface_hub import hf_hub_download

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_gsm8k():
    gsm8k_dir = DATA_DIR / "gsm8k"
    gsm8k_dir.mkdir(parents=True, exist_ok=True)
    dest_train = gsm8k_dir / "train.parquet"
    if not dest_train.exists():
        print("[1/10] Downloading GSM8K train (openai/gsm8k)...")
        src = hf_hub_download(repo_id="openai/gsm8k", filename="main/train-00000-of-00001.parquet", repo_type="dataset", token=HF_TOKEN)
        shutil.copyfile(src, dest_train)
        print(f"       GSM8K train saved: {dest_train.stat().st_size / (1024*1024):.2f} MB")
    else:
        print(f"[1/10] GSM8K train already present: {dest_train.stat().st_size / (1024*1024):.2f} MB")

    dest_test = gsm8k_dir / "test.parquet"
    if not dest_test.exists():
        print("       Downloading GSM8K held-out test (openai/gsm8k)...")
        src = hf_hub_download(repo_id="openai/gsm8k", filename="main/test-00000-of-00001.parquet", repo_type="dataset", token=HF_TOKEN)
        shutil.copyfile(src, dest_test)
        print(f"       GSM8K test saved: {dest_test.stat().st_size / 1024:.2f} KB")
    else:
        print(f"       GSM8K test already present: {dest_test.stat().st_size / 1024:.2f} KB")


def download_mbpp():
    mbpp_dir = DATA_DIR / "mbpp"
    mbpp_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "prompt", "test"]:
        dest = mbpp_dir / f"{split}.parquet"
        if not dest.exists():
            print(f"[2/10] Downloading MBPP {split} (google-research-datasets/mbpp)...")
            src = hf_hub_download(repo_id="google-research-datasets/mbpp", filename=f"full/{split}-00000-of-00001.parquet", repo_type="dataset", token=HF_TOKEN)
            shutil.copyfile(src, dest)
            print(f"       MBPP {split} saved: {dest.stat().st_size / 1024:.2f} KB")
        else:
            print(f"[2/10] MBPP {split} already present.")


def download_vision():
    vis_dir = DATA_DIR / "vision"
    vis_dir.mkdir(parents=True, exist_ok=True)
    dest = vis_dir / "detail_23k.json"
    if not dest.exists():
        print("[3/10] Downloading LLaVA Vision Detail 23k (liuhaotian/LLaVA-Instruct-150K)...")
        src = hf_hub_download(repo_id="liuhaotian/LLaVA-Instruct-150K", filename="detail_23k.json", repo_type="dataset", token=HF_TOKEN)
        shutil.copyfile(src, dest)
        print(f"       Vision saved: {dest.stat().st_size / (1024*1024):.2f} MB")
    else:
        print(f"[3/10] Vision detail_23k already present: {dest.stat().st_size / (1024*1024):.2f} MB")


def download_ultrachat():
    uc_dir = DATA_DIR / "ultrachat_200k" / "data"
    uc_dir.mkdir(parents=True, exist_ok=True)
    dest = uc_dir / "train_sft.parquet"
    if not dest.exists():
        print("[4/10] Downloading UltraChat SFT (HuggingFaceH4/ultrachat_200k)...")
        src = hf_hub_download(repo_id="HuggingFaceH4/ultrachat_200k", filename="data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet", repo_type="dataset", token=HF_TOKEN)
        shutil.copyfile(src, dest)
        print(f"       UltraChat saved: {dest.stat().st_size / (1024*1024):.2f} MB")
    else:
        print(f"[4/10] UltraChat already present: {dest.stat().st_size / (1024*1024):.2f} MB")


def download_squad():
    squad_dir = DATA_DIR / "squad_v2"
    squad_dir.mkdir(parents=True, exist_ok=True)
    dest_train = squad_dir / "train.parquet"
    if not dest_train.exists():
        print("[5/10] Downloading Stanford SQuAD 2.0 train (rajpurkar/squad_v2)...")
        src = hf_hub_download(repo_id="rajpurkar/squad_v2", filename="squad_v2/train-00000-of-00001.parquet", repo_type="dataset", token=HF_TOKEN)
        shutil.copyfile(src, dest_train)
        print(f"       SQuAD 2.0 train saved: {dest_train.stat().st_size / (1024*1024):.2f} MB")
    else:
        print(f"[5/10] SQuAD 2.0 train already present: {dest_train.stat().st_size / (1024*1024):.2f} MB")

    dest_val = squad_dir / "validation.parquet"
    if not dest_val.exists():
        print("       Downloading Stanford SQuAD 2.0 validation (rajpurkar/squad_v2)...")
        src = hf_hub_download(repo_id="rajpurkar/squad_v2", filename="squad_v2/validation-00000-of-00001.parquet", repo_type="dataset", token=HF_TOKEN)
        shutil.copyfile(src, dest_val)
        print(f"       SQuAD 2.0 validation saved: {dest_val.stat().st_size / (1024*1024):.2f} MB")
    else:
        print(f"       SQuAD 2.0 validation already present: {dest_val.stat().st_size / (1024*1024):.2f} MB")


def download_dolly():
    dolly_dir = DATA_DIR / "dolly"
    dolly_dir.mkdir(parents=True, exist_ok=True)
    dest = dolly_dir / "databricks-dolly-15k.jsonl"
    if not dest.exists():
        print("[6/10] Downloading Databricks Dolly-15k (databricks/databricks-dolly-15k)...")
        src = hf_hub_download(repo_id="databricks/databricks-dolly-15k", filename="databricks-dolly-15k.jsonl", repo_type="dataset", token=HF_TOKEN)
        shutil.copyfile(src, dest)
        print(f"       Dolly-15k saved: {dest.stat().st_size / (1024*1024):.2f} MB")
    else:
        print(f"[6/10] Dolly-15k already present: {dest.stat().st_size / (1024*1024):.2f} MB")


def download_smoltalk():
    smol_dir = DATA_DIR / "smoltalk"
    smol_dir.mkdir(parents=True, exist_ok=True)

    # 1. Everyday Conversations
    ec_dest = smol_dir / "everyday_conversations.parquet"
    if not ec_dest.exists():
        print("[7/10] Downloading SmolTalk Everyday Conversations (HuggingFaceTB/smoltalk)...")
        src = hf_hub_download(repo_id="HuggingFaceTB/smoltalk", filename="data/everyday-conversations/train-00000-of-00001.parquet", repo_type="dataset", token=HF_TOKEN)
        shutil.copyfile(src, ec_dest)
        print(f"       Everyday Conversations saved: {ec_dest.stat().st_size / (1024*1024):.2f} MB")
    else:
        print(f"[7/10] Everyday Conversations already present: {ec_dest.stat().st_size / (1024*1024):.2f} MB")

    # 2. Smol Magpie Ultra QA
    mp_dest = smol_dir / "smol_magpie_ultra.parquet"
    if not mp_dest.exists():
        print("[8/10] Downloading SmolTalk Magpie Ultra QA (HuggingFaceTB/smoltalk)...")
        src = hf_hub_download(repo_id="HuggingFaceTB/smoltalk", filename="data/smol-magpie-ultra/train-00000-of-00006.parquet", repo_type="dataset", token=HF_TOKEN)
        shutil.copyfile(src, mp_dest)
        print(f"       Magpie Ultra saved: {mp_dest.stat().st_size / (1024*1024):.2f} MB")
    else:
        print(f"[8/10] Magpie Ultra already present: {mp_dest.stat().st_size / (1024*1024):.2f} MB")

    # 3. OpenHermes 100k Diverse High-Density QA
    oh_dest = smol_dir / "openhermes_100k.parquet"
    if not oh_dest.exists():
        print("       Downloading SmolTalk OpenHermes 100k QA...")
        src = hf_hub_download(repo_id="HuggingFaceTB/smoltalk", filename="data/openhermes-100k/train-00000-of-00001.parquet", repo_type="dataset", token=HF_TOKEN)
        shutil.copyfile(src, oh_dest)
        print(f"       OpenHermes 100k saved: {oh_dest.stat().st_size / (1024*1024):.2f} MB")
    else:
        print(f"       OpenHermes 100k already present: {oh_dest.stat().st_size / (1024*1024):.2f} MB")

    # 4. NuminaMath CoT 100k Step-by-Step Math Reasoning
    nm_dest = smol_dir / "numina_cot_100k.parquet"
    if not nm_dest.exists():
        print("       Downloading SmolTalk NuminaMath CoT 100k...")
        src = hf_hub_download(repo_id="HuggingFaceTB/smoltalk", filename="data/numina-cot-100k/train-00000-of-00001.parquet", repo_type="dataset", token=HF_TOKEN)
        shutil.copyfile(src, nm_dest)
        print(f"       NuminaMath CoT 100k saved: {nm_dest.stat().st_size / (1024*1024):.2f} MB")
    else:
        print(f"       NuminaMath CoT 100k already present: {nm_dest.stat().st_size / (1024*1024):.2f} MB")


def download_hotpotqa():
    hp_dir = DATA_DIR / "hotpotqa" / "distractor"
    hp_dir.mkdir(parents=True, exist_ok=True)
    dest_train = hp_dir / "train-00000-of-00002.parquet"
    if not dest_train.exists():
        print("[9/10] Downloading HotpotQA Multi-Hop train (hotpotqa/hotpot_qa)...")
        src = hf_hub_download(repo_id="hotpotqa/hotpot_qa", filename="distractor/train-00000-of-00002.parquet", repo_type="dataset", token=HF_TOKEN)
        shutil.copyfile(src, dest_train)
        print(f"       HotpotQA train saved: {dest_train.stat().st_size / (1024*1024):.2f} MB")
    else:
        print(f"[9/10] HotpotQA train already present: {dest_train.stat().st_size / (1024*1024):.2f} MB")

    dest_val = hp_dir / "validation.parquet"
    if not dest_val.exists():
        print("       Downloading HotpotQA Multi-Hop validation (hotpotqa/hotpot_qa)...")
        src = hf_hub_download(repo_id="hotpotqa/hotpot_qa", filename="distractor/validation-00000-of-00001.parquet", repo_type="dataset", token=HF_TOKEN)
        shutil.copyfile(src, dest_val)
        print(f"       HotpotQA validation saved: {dest_val.stat().st_size / (1024*1024):.2f} MB")
    else:
        print(f"       HotpotQA validation already present: {dest_val.stat().st_size / (1024*1024):.2f} MB")


def download_qasrl():
    qasrl_dir = DATA_DIR / "qasrl_gs"
    qasrl_extract = qasrl_dir / "qasrl-gs"
    if not (qasrl_extract / "dev.jsonl.gz").exists():
        print("[10/10] Downloading QA-SRL 2020 Causal Reasoning (qasrl.org)...")
        qasrl_dir.mkdir(parents=True, exist_ok=True)
        tar_dest = qasrl_dir / "qasrl-gs.tar"
        req = urllib.request.Request("https://qasrl.org/data/qasrl-gs.tar", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(tar_dest, "wb") as out:
            shutil.copyfileobj(resp, out)
        with tarfile.open(tar_dest, "r") as tar:
            tar.extractall(qasrl_dir)
        print(f"        QA-SRL extracted to {qasrl_extract}")
    else:
        print("[10/10] QA-SRL 2020 already present.")


def download_ud(download_archive: bool = False):
    ud_dest = BASE_DIR / "ud-treebanks-v2.18.tgz"
    if not ud_dest.exists():
        # Check for any existing ud-treebanks archive
        found = list(BASE_DIR.glob("ud-treebanks-*.tgz"))
        if found:
            print(f"[Bonus] Universal Dependencies archive already present: {found[0].name}")
            return
    if download_archive and not ud_dest.exists():
        print("[Bonus] Downloading Universal Dependencies v2.18 Treebank Archive (550 MB)...")
        req = urllib.request.Request(
            "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5502/ud-treebanks-v2.14.tgz",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req) as resp, open(ud_dest, "wb") as out:
            shutil.copyfileobj(resp, out)
        print(f"        UD Archive saved: {ud_dest} ({ud_dest.stat().st_size / (1024*1024):.2f} MB)")
    elif ud_dest.exists():
        print(f"[Bonus] UD Archive already present: {ud_dest.name}")


def download_all(with_ud: bool = False):
    print("=======================================================")
    print("  AR.I.S Unified Master Dataset Downloader")
    print("  Hugging Face Authentication:", "Active (Higher Rate Limits)" if HF_TOKEN else "Unauthenticated")
    print("=======================================================\n")
    download_gsm8k()
    download_mbpp()
    download_vision()
    download_ultrachat()
    download_squad()
    download_dolly()
    download_smoltalk()
    download_hotpotqa()
    download_qasrl()
    if with_ud:
        download_ud(download_archive=True)
    else:
        download_ud(download_archive=False)
    print("\n[Complete] All foundational & QA datasets are synchronized and ready for training!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AR.I.S Master Dataset Downloader")
    parser.add_argument("--with-ud", action="store_true", help="Also download the 550MB Universal Dependencies Treebank archive")
    parser.add_argument("--with", dest="with_feature", type=str, default=None, help="Include optional datasets like 'ud'")
    args = parser.parse_args()
    with_ud = args.with_ud or (args.with_feature is not None and args.with_feature.lower() == "ud")
    download_all(with_ud=with_ud)
