import yaml
from pathlib import Path
import logging

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Конфигурационный файл не найден по пути: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_config = load_config()

OUTPUT_DIR = Path(_config["storage"]["output_dir"])
TEMP_DIR = Path(_config["storage"]["temp_dir"])
TEMP_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = Path(_config["storage"]["log_file"])
LOG_FILE.parent.mkdir(exist_ok=True)

DEPTH_MODEL_NAME = _config["models"]["depth"]["model_name"]
SAM_CHUNK_SIZE = _config["models"]["layer_separation"]["chunk_size"]


logging.basicConfig(
    level=logging.INFO,  # [DEBUG, INFO, WARNING, ERROR, CRITICAL]
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()  # также вывод в консоль
    ]
)

logging.getLogger("asyncio").setLevel(logging.CRITICAL)


BASE_DIR = Path(__file__).parent.parent
CHECKPOINT_DIR = BASE_DIR / "processors" / "layer_separation" / "checkpoints"

NANOTRACK_BACKBONE = CHECKPOINT_DIR / "nanotrack_backbone_sim.onnx"
NANOTRACK_HEAD = CHECKPOINT_DIR / "nanotrack_head_sim.onnx"

SAM2_CHECKPOINT = CHECKPOINT_DIR / "sam2.1_hiera_tiny.pt"

COTRACKER_CHECKPOINT = CHECKPOINT_DIR / "scaled_offline.pth"
