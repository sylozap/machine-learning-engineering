"""Загрузка датасета Butterfly Image Classification с Kaggle."""
import shutil

from kaggle.api.kaggle_api_extended import KaggleApi

from common import RAW_DIR, load_params


def main():
    params = load_params()["data"]
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    RAW_DIR.mkdir(parents=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(params["kaggle_dataset"], path=str(RAW_DIR), unzip=True, quiet=False)

    # Kaggle-овский test не размечен, поэтому он не используется
    shutil.rmtree(RAW_DIR / "test", ignore_errors=True)
    (RAW_DIR / "Testing_set.csv").unlink(missing_ok=True)
    print(f"Downloaded to {RAW_DIR}")


if __name__ == "__main__":
    main()
