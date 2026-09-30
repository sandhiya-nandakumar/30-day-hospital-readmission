import os
import shutil
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
TARGET = DATA_DIR / "diabetic_data.csv"

if TARGET.exists():
    print(f"Dataset already exists: {TARGET}")
    raise SystemExit

print("Downloading the Kaggle dataset: brandao/diabetes")
print("If Kaggle asks for authentication, follow the KaggleHub instructions.")

try:
    import kagglehub
    downloaded = Path(kagglehub.dataset_download("brandao/diabetes"))
    print("Downloaded to:", downloaded)

    candidates = list(downloaded.rglob("diabetic_data.csv"))
    if not candidates:
        raise FileNotFoundError("diabetic_data.csv was not found in the downloaded Kaggle dataset.")

    shutil.copy2(candidates[0], TARGET)
    print("Copied dataset to:", TARGET)
except Exception as exc:
    print("\nAutomatic Kaggle download failed.")
    print("Download it manually from:")
    print("https://www.kaggle.com/datasets/brandao/diabetes")
    print("Then place diabetic_data.csv inside the data/ folder.")
    print("\nError:", exc)
