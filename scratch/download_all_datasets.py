import os
import tarfile
import urllib.request
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_qasrl():
    qasrl_dir = DATA_DIR / "qasrl_gs"
    qasrl_extract = qasrl_dir / "qasrl-gs"
    if not (qasrl_extract / "dev.jsonl.gz").exists():
        print("[1/4] Downloading QA-SRL 2020 Causal Reasoning (622 KB)...")
        qasrl_dir.mkdir(parents=True, exist_ok=True)
        tar_dest = qasrl_dir / "qasrl-gs.tar"
        req = urllib.request.Request("https://qasrl.org/data/qasrl-gs.tar", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(tar_dest, "wb") as out:
            shutil.copyfileobj(resp, out)
        with tarfile.open(tar_dest, "r") as tar:
            tar.extractall(qasrl_dir)
        print(f"QA-SRL extracted to {qasrl_extract}")
    else:
        print("QA-SRL already exists.")

def download_hotpotqa():
    hp_dir = DATA_DIR / "hotpotqa" / "distractor"
    hp_dir.mkdir(parents=True, exist_ok=True)
    hp_dest = hp_dir / "train-00000-of-00002.parquet"
    if not hp_dest.exists():
        print("[2/4] Downloading HotpotQA Multi-Hop Reasoning...")
        src = hf_hub_download(repo_id="hotpotqa/hotpot_qa", filename="distractor/train-00000-of-00002.parquet", repo_type="dataset")
        shutil.copyfile(src, hp_dest)
        print(f"HotpotQA saved to {hp_dest}")
    else:
        print("HotpotQA already exists.")

def download_ultrachat():
    uc_dir = DATA_DIR / "ultrachat_200k" / "data"
    uc_dir.mkdir(parents=True, exist_ok=True)
    uc_dest = uc_dir / "train_sft.parquet"
    if not uc_dest.exists():
        print("[3/4] Downloading UltraChat Dialogue (77 MB)...")
        src = hf_hub_download(repo_id="HuggingFaceH4/ultrachat_200k", filename="data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet", repo_type="dataset")
        shutil.copyfile(src, uc_dest)
        print(f"UltraChat saved to {uc_dest}")
    else:
        print("UltraChat already exists.")

def download_ud(download_archive: bool = False):
    ud_dest = Path("ud-treebanks-v2.14.tgz")
    if download_archive and not ud_dest.exists():
        print("[4/4] Downloading Universal Dependencies Archive (550 MB)...")
        req = urllib.request.Request("https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5502/ud-treebanks-v2.14.tgz", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(ud_dest, "wb") as out:
            shutil.copyfileobj(resp, out)
        print(f"UD Archive saved to {ud_dest}")
    elif ud_dest.exists():
        print("UD Archive already exists.")

def download_expanded():
    # GSM8K
    gsm_dir = DATA_DIR / "gsm8k"
    gsm_dir.mkdir(parents=True, exist_ok=True)
    gsm_dest = gsm_dir / "train.parquet"
    if not gsm_dest.exists():
        src = hf_hub_download(repo_id="openai/gsm8k", filename="main/train-00000-of-00001.parquet", repo_type="dataset")
        shutil.copyfile(src, gsm_dest)

    # MBPP
    mbpp_dir = DATA_DIR / "mbpp"
    mbpp_dir.mkdir(parents=True, exist_ok=True)
    mbpp_dest = mbpp_dir / "train.parquet"
    if not mbpp_dest.exists():
        src = hf_hub_download(repo_id="google-research-datasets/mbpp", filename="full/train-00000-of-00001.parquet", repo_type="dataset")
        shutil.copyfile(src, mbpp_dest)

    # Vision
    vis_dir = DATA_DIR / "vision"
    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_dest = vis_dir / "detail_23k.json"
    if not vis_dest.exists():
        src = hf_hub_download(repo_id="liuhaotian/LLaVA-Instruct-150K", filename="detail_23k.json", repo_type="dataset")
        shutil.copyfile(src, vis_dest)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download all AR.I.S training corpora")
    parser.add_argument("--with-ud", action="store_true", help="Also download the large 550MB Universal Dependencies treebank archive")
    args = parser.parse_args()

    print("=== Downloading All Foundation Datasets for AR.I.S ===")
    download_qasrl()
    download_hotpotqa()
    download_ultrachat()
    download_expanded()
    if args.with_ud:
        download_ud(download_archive=True)
    print("\n[SUCCESS] All requested corpora are on disk and ready for training!")
