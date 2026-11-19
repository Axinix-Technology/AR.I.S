"""
AR.I.S — Central System Configuration
Purpose:
  Manages filesystem paths, compute device selection (GPU CUDA vs CPU),
  database credentials, and default hyperparameters for the neural model.
"""

from __future__ import annotations
import os
import torch
from pathlib import Path
from dotenv import load_dotenv

# Base Directory: root of the AR.I.S repository
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")

# Primary Directories
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
WORKSPACE_DIR = BASE_DIR / "workspace"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# Compute Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Paths
TOKENIZER_PATH = MODELS_DIR / "aris_bpe_tokenizer.json"
LM_CHECKPOINT_PATH = MODELS_DIR / "aris_lm_checkpoint.pt"

# Database Configuration (MySQL Relational Memory)
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "root@123"),
    "database": os.getenv("MYSQL_DATABASE", "aris_db"),
    "charset": "utf8mb4",
}

# Default Neural Architecture Hyperparameters (ArisLM Gen 2: ~24.6M Parameters)
MODEL_CONFIG = {
    "vocab_size": 4096,        # BPE Vocabulary size
    "max_seq_len": 384,        # Maximum context window (tokens)
    "embed_dim": 512,          # Dense representation dimension
    "num_heads": 8,            # Multi-head attention heads
    "num_layers": 8,           # Stacked Transformer Decoder blocks
    "ff_dim": 1408,            # SwiGLU hidden expansion
    "dropout": 0.0,            # Modern LLMs use 0.0 dropout
}
