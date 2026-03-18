import torch
from pathlib import Path

# Chemins principaux du projet
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"

# Paramètres audio
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Modèles Demucs supportés
SUPPORTED_MODELS = ["htdemucs", "htdemucs_ft", "mdx_extra", "mdx_extra_q"]
DEFAULT_MODEL = "htdemucs"
