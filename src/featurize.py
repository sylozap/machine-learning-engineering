"""Извлечение признаков изображений предобученной CNN (ImageNet) без дообучения."""
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm

from common import AUGMENTED_DIR, FEATURES_DIR, PREPARED_DIR, RAW_DIR, load_params

BACKBONES = {
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
}


class ImageDataset(Dataset):
    def __init__(self, df, root, transform):
        self.paths = [root / p for p in df["path"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


def build_backbone(name):
    builder, weights = BACKBONES[name]
    model = builder(weights=weights)
    # отрезаем классификатор ImageNet, оставляем вектор признаков после global pooling
    if name == "resnet50":
        model.fc = torch.nn.Identity()
    else:
        model.classifier = torch.nn.Identity()
    return model.eval(), weights.transforms()


@torch.no_grad()
def extract(model, transform, df, root, p):
    loader = DataLoader(ImageDataset(df, root, transform), batch_size=p["batch_size"],
                        num_workers=p["num_workers"], shuffle=False)
    return np.concatenate([model(x).numpy() for x in tqdm(loader, leave=False)]).astype(np.float32)


def main():
    p = load_params()["featurize"]
    model, transform = build_backbone(p["backbone"])
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": (pd.read_csv(PREPARED_DIR / "train.csv"), RAW_DIR),
        "val": (pd.read_csv(PREPARED_DIR / "val.csv"), RAW_DIR),
        "test": (pd.read_csv(PREPARED_DIR / "test.csv"), RAW_DIR),
        "train_aug": (pd.read_csv(AUGMENTED_DIR / "train_aug.csv"), AUGMENTED_DIR),
    }
    for name, (df, root) in splits.items():
        start = time.time()
        X = extract(model, transform, df, root, p)
        np.savez(FEATURES_DIR / f"{name}.npz", X=X, y=df["target"].to_numpy())
        print(f"{name}: {X.shape} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
