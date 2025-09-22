import os
import random
import joblib
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# -------------------------------------------------------------------
# Load environment variables from .env (repo root)
# -------------------------------------------------------------------
# Find repo root (one level up from src/)
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

# -------------------------------------------------------------------
# Paths (default to repo-relative if not in .env)
# -------------------------------------------------------------------
def get_path(var_name: str, default: str):
    """Return path from env var, or default relative to repo root"""
    return Path(os.getenv(var_name, REPO_ROOT / default))

RAW_DIR       = get_path("RAW_DIR", "data/data_cleaned.csv")

#Directories
PROCESSED_DIR = get_path("PROCESSED_DIR", "data/processed")
MODELS_DIR    = get_path("MODELS_DIR", "models")
RESULTS_DIR   = get_path("RESULTS_DIR", "results")
LOG_DIR       = get_path("LOG_DIR", "logs")

# Create dirs if they don’t exist
for d in [PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------
SEED = int(os.getenv("SEED", 42))

def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch not always installed

# -------------------------------------------------------------------
# Model save/load helpers
# -------------------------------------------------------------------
def save_model(model, path: Path):
    """Save model to disk"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(path: Path):
    """Load model from disk"""
    return joblib.load(path)
