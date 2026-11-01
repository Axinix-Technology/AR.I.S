import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

DATA_DIR = Path("data")

def download_all():
    print("=== Downloading Expanded Datasets for AR.I.S ===")
    
    # 1. GSM8K (Problem Solving)
    gsm8k_dir = DATA_DIR / "gsm8k"
    gsm8k_dir.mkdir(parents=True, exist_ok=True)
    gsm8k_dest = gsm8k_dir / "train.parquet"
    if not gsm8k_dest.exists():
        print("[1/3] Downloading GSM8K (openai/gsm8k)...")
        src = hf_hub_download(repo_id="openai/gsm8k", filename="main/train-00000-of-00001.parquet", repo_type="dataset")
        shutil.copyfile(src, gsm8k_dest)
        print(f"GSM8K saved to {gsm8k_dest} ({gsm8k_dest.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"GSM8K already exists at {gsm8k_dest}")

    # 2. MBPP (Grade 1 Coding Knowledge)
    mbpp_dir = DATA_DIR / "mbpp"
    mbpp_dir.mkdir(parents=True, exist_ok=True)
    mbpp_dest = mbpp_dir / "train.parquet"
    if not mbpp_dest.exists():
        print("[2/3] Downloading MBPP (google-research-datasets/mbpp)...")
        src = hf_hub_download(repo_id="google-research-datasets/mbpp", filename="full/train-00000-of-00001.parquet", repo_type="dataset")
        shutil.copyfile(src, mbpp_dest)
        print(f"MBPP train saved to {mbpp_dest} ({mbpp_dest.stat().st_size / 1024:.2f} KB)")
    else:
        print(f"MBPP already exists at {mbpp_dest}")

    # Also download MBPP prompt and test for completeness
    for split in ["prompt", "test"]:
        split_dest = mbpp_dir / f"{split}.parquet"
        if not split_dest.exists():
            src = hf_hub_download(repo_id="google-research-datasets/mbpp", filename=f"full/{split}-00000-of-00001.parquet", repo_type="dataset")
            shutil.copyfile(src, split_dest)
            print(f"MBPP {split} saved to {split_dest}")

    # 3. Vision / Image Processing (liuhaotian/LLaVA-Instruct-150K detail_23k.json)
    vision_dir = DATA_DIR / "vision"
    vision_dir.mkdir(parents=True, exist_ok=True)
    vision_dest = vision_dir / "detail_23k.json"
    if not vision_dest.exists():
        print("[3/3] Downloading LLaVA-Instruct detail_23k (liuhaotian/LLaVA-Instruct-150K)...")
        src = hf_hub_download(repo_id="liuhaotian/LLaVA-Instruct-150K", filename="detail_23k.json", repo_type="dataset")
        shutil.copyfile(src, vision_dest)
        print(f"Vision dataset saved to {vision_dest} ({vision_dest.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"Vision dataset already exists at {vision_dest}")

    print("\nAll datasets downloaded successfully!")

if __name__ == "__main__":
    download_all()

