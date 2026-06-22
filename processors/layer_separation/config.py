from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = BASE_DIR / "processors" / "layer_separation" / "checkpoints" / "sam2.1_hiera_tiny.pt"
OUTPUT_DIR = BASE_DIR / "processors" / "layer_separation" / "outputs"