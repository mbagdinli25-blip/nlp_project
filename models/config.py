"""Shared configuration: paths and constants for all pipeline scripts.

Edit the CONSTANTS below for your environment. The defaults auto-switch
between Google Colab (Drive paths) and the local Mac.
"""
from pathlib import Path
import os
import sys

IN_COLAB = "google.colab" in sys.modules

# ---- Paths -----------------------------------------------------------------
# In Colab: mount Drive first (`from google.colab import drive; drive.mount('/content/drive')`)
# then place the deprem/ folder under MyDrive so these paths resolve.
if IN_COLAB:
    DATA_ROOT = Path("/content/drive/MyDrive/deprem")
    OUTPUT_ROOT = Path("/content/drive/MyDrive/deprem/outputs")
else:
    DATA_ROOT = Path("/Users/mehmetbagdinli/Desktop/deprem")
    OUTPUT_ROOT = Path(__file__).parent / "outputs"

TRAIN_CSV = DATA_ROOT / "annotation" / "final_300.csv"
CORPUS_CSV = DATA_ROOT / "corpus" / "corpus_final.csv"

MODEL_DIR = OUTPUT_ROOT / "berturk_model"
SCORED_CORPUS_CSV = OUTPUT_ROOT / "analytic_with_scores.csv"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---- Task constants --------------------------------------------------------
FRAMES = ["technical", "political", "development", "sustainability"]
TEXT_COL = "text"
STRATIFY_COL = "province"
RANDOM_STATE = 42

# BERTurk
HF_MODEL = "dbmdz/bert-base-turkish-cased"
MAX_LEN = 512
EPOCHS = 4
LR = 2e-5
BATCH_SIZE = 8
TEST_SIZE = 0.2

# Label normalization: 0–3 ordinal → 0–1 for regression stability
LABEL_MAX = 3.0


def get_device():
    """Return torch device: cuda > mps > cpu, with a printed warning on CPU."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    print("WARNING: no GPU detected, falling back to CPU. Training will be slow.",
          file=sys.stderr)
    return torch.device("cpu")
