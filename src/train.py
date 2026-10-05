"""Обучение классификатора на признаках CNN."""
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import FEATURES_DIR, METRICS_DIR, MODELS_DIR, load_params, save_json


def build_model(p):
    name, seed = p["model"], p["seed"]
    if name == "logreg":
        clf = LogisticRegression(C=p["logreg"]["C"], max_iter=p["logreg"]["max_iter"], random_state=seed)
    elif name == "svm":
        clf = SVC(C=p["svm"]["C"], kernel=p["svm"]["kernel"], random_state=seed)
    elif name == "random_forest":
        clf = RandomForestClassifier(n_estimators=p["random_forest"]["n_estimators"],
                                     max_depth=p["random_forest"]["max_depth"], n_jobs=-1, random_state=seed)
    else:
        raise ValueError(f"Unknown model: {name}")
    return make_pipeline(StandardScaler(), clf)


def load_train(use_augmented):
    parts = ["train", "train_aug"] if use_augmented else ["train"]
    data = [np.load(FEATURES_DIR / f"{part}.npz") for part in parts]
    return np.concatenate([d["X"] for d in data]), np.concatenate([d["y"] for d in data])


def main():
    p = load_params()["train"]
    X, y = load_train(p["use_augmented"])
    model = build_model(p)

    start = time.time()
    model.fit(X, y)
    train_time = time.time() - start

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model.joblib")
    save_json({"n_train": int(len(y)), "train_time_sec": round(train_time, 2)}, METRICS_DIR / "train.json")
    print(f"model={p['model']} augmented={p['use_augmented']} n_train={len(y)} time={train_time:.1f}s")


if __name__ == "__main__":
    main()
