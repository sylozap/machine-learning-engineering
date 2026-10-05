"""Стратифицированное разбиение размеченных изображений на train/val/test."""
import pandas as pd
from sklearn.model_selection import train_test_split

from common import PREPARED_DIR, RAW_DIR, load_params, save_json


def main():
    params = load_params()["data"]
    df = pd.read_csv(RAW_DIR / "Training_set.csv")
    df["path"] = "train/" + df["filename"]
    df = df[(RAW_DIR / df["path"]).map(lambda p: p.exists())].reset_index(drop=True)

    classes = sorted(df["label"].unique())
    df["target"] = df["label"].map({c: i for i, c in enumerate(classes)})

    holdout = params["val_size"] + params["test_size"]
    train, rest = train_test_split(df, test_size=holdout, stratify=df["target"], random_state=params["seed"])
    val, test = train_test_split(
        rest, test_size=params["test_size"] / holdout, stratify=rest["target"], random_state=params["seed"]
    )

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["path", "label", "target"]
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part[cols].to_csv(PREPARED_DIR / f"{name}.csv", index=False)
    save_json(classes, PREPARED_DIR / "classes.json")
    print(f"classes={len(classes)} train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
