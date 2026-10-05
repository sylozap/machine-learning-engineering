import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PREPARED_DIR = ROOT / "data" / "prepared"
AUGMENTED_DIR = ROOT / "data" / "augmented"
FEATURES_DIR = ROOT / "data" / "features"
MODELS_DIR = ROOT / "models"
METRICS_DIR = ROOT / "metrics"
PLOTS_DIR = ROOT / "plots"


def load_params():
    with open(ROOT / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
