"""
Script to download Vietnamese ASR datasets: VIVOS and Common Voice
"""
import os
import requests
import tarfile
from datasets import load_dataset
import torchaudio
import pandas as pd
from tqdm import tqdm
import shutil

def download_vivos(output_dir):
    """Download VIVOS dataset from Zenodo"""
    print("Downloading VIVOS dataset from Zenodo...")
    vivos_dir = os.path.join(output_dir, "vivos")
    os.makedirs(vivos_dir, exist_ok=True)

    # Download from Zenodo
    url = "https://zenodo.org/record/7068130/files/vivos.tar.gz"
    tar_path = os.path.join(vivos_dir, "vivos.tar.gz")

    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(tar_path, 'wb') as f, tqdm(
        desc="Downloading VIVOS",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)

    print("Extracting...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(vivos_dir)

    print(f"VIVOS dataset extracted to {vivos_dir}")
    print("Note: VIVOS needs to be loaded from local files")

    return vivos_dir

def download_common_voice_vietnamese(output_dir):
    """Download Common Voice Vietnamese dataset"""
    print("Downloading Common Voice Vietnamese dataset...")
    cv_dir = os.path.join(output_dir, "common_voice_vi")

    # Load Common Voice Vietnamese from Hugging Face (version 16.0)
    try:
        cv_train = load_dataset("mozilla-foundation/common_voice_16_0", "vi", split="train")
        cv_test = load_dataset("mozilla-foundation/common_voice_16_0", "vi", split="test")
        cv_validation = load_dataset("mozilla-foundation/common_voice_16_0", "vi", split="validation")
    except:
        # Fallback to version 13.0
        print("Version 16.0 not found, trying version 13.0...")
        cv_train = load_dataset("mozilla-foundation/common_voice_13_0", "vi", split="train")
        cv_test = load_dataset("mozilla-foundation/common_voice_13_0", "vi", split="test")
        cv_validation = load_dataset("mozilla-foundation/common_voice_13_0", "vi", split="validation")

    # Filter only validated samples
    cv_train = cv_train.filter(lambda x: x["validation"] is True)
    cv_validation = cv_validation.filter(lambda x: x["validation"] is True)

    print(f"Common Voice Vietnamese:")
    print(f"Train samples (validated): {len(cv_train)}")
    print(f"Validation samples (validated): {len(cv_validation)}")
    print(f"Test samples: {len(cv_test)}")

    # Save to local directory
    from datasets import DatasetDict
    cv_dataset = DatasetDict({
        "train": cv_train,
        "validation": cv_validation,
        "test": cv_test
    })
    cv_dataset.save_to_disk(cv_dir)

    print(f"Common Voice Vietnamese saved to {cv_dir}")

    return cv_dataset

def download_fleurs_vietnamese(output_dir):
    """Download FLEURS Vietnamese dataset"""
    print("Downloading FLEURS Vietnamese dataset...")
    fleurs_dir = os.path.join(output_dir, "fleurs_vi")

    # Load FLEURS Vietnamese
    fleurs = load_dataset("google/fleurs", "vi_vn")

    print(f"FLEURS Vietnamese:")
    print(f"Train samples: {len(fleurs['train'])}")
    print(f"Validation samples: {len(fleurs['validation'])}")
    print(f"Test samples: {len(fleurs['test'])}")

    # Save to local directory
    fleurs.save_to_disk(fleurs_dir)

    print(f"FLEURS Vietnamese saved to {fleurs_dir}")

    return fleurs

if __name__ == "__main__":
    output_dir = "D:/Truong/data/raw"

    os.makedirs(output_dir, exist_ok=True)

    # Download datasets
    print("=" * 60)
    print("Downloading Vietnamese ASR Datasets")
    print("=" * 60)

    # Download VIVOS (skip if already exists)
    vivos_dir = os.path.join(output_dir, "vivos")
    if os.path.exists(vivos_dir) and os.listdir(vivos_dir):
        print(f"VIVOS already exists at {vivos_dir}, skipping download")
    else:
        vivos = download_vivos(output_dir)

    # Download Common Voice Vietnamese (optional - may fail due to access issues)
    try:
        cv_dataset = download_common_voice_vietnamese(output_dir)
    except Exception as e:
        print(f"Common Voice download failed: {e}")
        print("Will continue with VIVOS only for now")

    # Download FLEURS Vietnamese (optional - uncomment if needed)
    # fleurs = download_fleurs_vietnamese(output_dir)

    print("\n" + "=" * 60)
    print("Dataset download completed!")
    print("=" * 60)
