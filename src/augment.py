"""Офлайн-аугментация обучающей выборки: создаёт новый датасет из преобразованных копий."""
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2
from tqdm import tqdm

from common import AUGMENTED_DIR, PLOTS_DIR, PREPARED_DIR, RAW_DIR, load_params


def build_transform(p):
    return v2.Compose([
        v2.RandomResizedCrop(p["image_size"], scale=(p["crop_scale_min"], 1.0)),
        v2.RandomHorizontalFlip(p["hflip_p"]),
        v2.RandomRotation(p["rotation_deg"]),
        v2.ColorJitter(brightness=p["brightness"], contrast=p["contrast"],
                       saturation=p["saturation"], hue=p["hue"]),
        v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=p["blur_p"]),
    ])


def save_examples(train, transform, n_images=4, n_aug=4):
    fig, axes = plt.subplots(n_images, n_aug + 1, figsize=(2.2 * (n_aug + 1), 2.3 * n_images))
    for row, (_, item) in enumerate(train.sample(n_images, random_state=0).iterrows()):
        img = Image.open(RAW_DIR / item["path"]).convert("RGB")
        variants = [img] + [transform(img) for _ in range(n_aug)]
        for col, im in enumerate(variants):
            ax = axes[row, col]
            ax.imshow(im)
            ax.axis("off")
            if row == 0:
                ax.set_title("исходное" if col == 0 else f"аугм. {col}", fontsize=10)
        axes[row, 0].text(-10, 112, item["label"].title(), rotation=90, va="center", ha="right", fontsize=8)
    fig.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    fig.savefig(PLOTS_DIR / "augmentation_examples.png", dpi=120)
    plt.close(fig)


def main():
    p = load_params()["augment"]
    torch.manual_seed(p["seed"])
    transform = build_transform(p)
    train = pd.read_csv(PREPARED_DIR / "train.csv")

    if AUGMENTED_DIR.exists():
        shutil.rmtree(AUGMENTED_DIR)
    img_dir = AUGMENTED_DIR / "images"
    img_dir.mkdir(parents=True)

    rows = []
    for _, item in tqdm(train.iterrows(), total=len(train), desc="augment"):
        img = Image.open(RAW_DIR / item["path"]).convert("RGB")
        stem = item["path"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
        for k in range(p["copies_per_image"]):
            name = f"{stem}_aug{k}.jpg"
            transform(img).save(img_dir / name, quality=95)
            rows.append({"path": f"images/{name}", "label": item["label"], "target": item["target"]})

    pd.DataFrame(rows).to_csv(AUGMENTED_DIR / "train_aug.csv", index=False)
    save_examples(train, transform)
    print(f"augmented images: {len(rows)}")


if __name__ == "__main__":
    main()
