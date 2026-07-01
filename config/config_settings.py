import yaml
from pathlib import Path
import logging

CONFIG_PATH = Path(__file__).parent / "config.yaml"
BASE_DIR = Path(__file__).parent.parent


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


logging.basicConfig(
    level=logging.INFO,  # [DEBUG, INFO, WARNING, ERROR, CRITICAL]
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()  # также вывод в консоль
    ]
)

logging.getLogger("asyncio").setLevel(logging.CRITICAL)


_ls_config = _config["models"]["layer_separation"]
DEVICE = _ls_config.get("device", "cuda")
APP_SAMPLING_STEP = _ls_config.get("sampling_step", 10)

SAM2_CHECKPOINT = BASE_DIR / _ls_config["sam2"]["checkpoint"]
SAM2_MODEL_CFG = _ls_config["sam2"]["model_cfg"]
SAM2_CHUNK_SIZE = _ls_config["sam2"]["chunk_size"]

COTRACKER_CHECKPOINT = BASE_DIR / _ls_config["cotracker"]["checkpoint"]
COTRACKER_CHUNK_SIZE = _ls_config["cotracker"]["chunk_size"]
COTRACKER_GRID_STEP = _ls_config["cotracker"]["grid_step"]

NANOTRACK_BACKBONE = BASE_DIR / _ls_config["nanotrack"]["backbone"]
NANOTRACK_HEAD = BASE_DIR / _ls_config["nanotrack"]["head"]
