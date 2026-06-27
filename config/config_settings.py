import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Конфигурационный файл не найден по пути: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_config = load_config()

OUTPUT_DIR = Path(_config["storage"]["output_dir"])
TEMP_DIR = Path(_config["storage"]["temp_dir"])

DEPTH_MODEL_NAME = _config["models"]["depth"]["model_name"]
SAM_CHUNK_SIZE = _config["models"]["layer_separation"]["chunk_size"]

TEMP_DIR.mkdir(parents=True, exist_ok=True)
