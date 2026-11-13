"""Download the Chest X-Ray Pneumonia dataset from Kaggle into data/chest_xray.

Usage:  python download_dataset.py
(If this fails, download the zip manually - see README.md, Step 3.)
"""
import shutil
from pathlib import Path

import kagglehub

TARGET = Path("data/chest_xray")


def find_dataset_root(root: Path) -> Path:
    """Find the folder that directly contains 'train' and 'test'."""
    for p in [root, *root.rglob("*")]:
        if p.is_dir() and (p / "train").is_dir() and (p / "test").is_dir() and "__MACOSX" not in str(p):
            return p
    raise FileNotFoundError("Could not find train/test folders in the download.")


if __name__ == "__main__":
    print("Downloading dataset (about 1.2 GB)... please wait.")
    path = Path(kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia"))
    src = find_dataset_root(path)
    if TARGET.exists():
        shutil.rmtree(TARGET)
    shutil.copytree(src, TARGET)
    print(f"Done! Dataset is ready at: {TARGET.resolve()}")
