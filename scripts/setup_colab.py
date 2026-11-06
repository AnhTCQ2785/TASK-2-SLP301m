"""
Setup script for Google Colab
Run this first in Colab to set up the environment
"""
import os
import subprocess

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    subprocess.run(["pip", "install", "-q", "torch>=2.0.0", "torchaudio>=2.0.0"])
    subprocess.run(["pip", "install", "-q", "transformers>=4.30.0", "datasets>=2.14.0", "accelerate>=0.20.0"])
    subprocess.run(["pip", "install", "-q", "librosa", "soundfile", "pydub"])
    subprocess.run(["pip", "install", "-q", "jiwer", "sacrebleu", "torchmetrics"])
    subprocess.run(["pip", "install", "-q", "matplotlib", "seaborn", "tensorboard"])
    subprocess.run(["pip", "install", "-q", "tqdm", "numpy", "pandas", "scipy"])
    subprocess.run(["pip", "install", "-q", "sentencepiece", "sacremoses"])
    print("Packages installed successfully!")

def mount_drive():
    """Mount Google Drive for saving checkpoints"""
    from google.colab import drive
    print("Mounting Google Drive...")
    drive.mount('/content/drive')
    print("Google Drive mounted!")

def setup_directories():
    """Create necessary directories"""
    print("Setting up directories...")
    os.makedirs("/content/data/raw", exist_ok=True)
    os.makedirs("/content/data/processed", exist_ok=True)
    os.makedirs("/content/data/manifests", exist_ok=True)
    os.makedirs("/content/models", exist_ok=True)
    os.makedirs("/content/checkpoints", exist_ok=True)
    os.makedirs("/content/logs", exist_ok=True)
    print("Directories created!")

if __name__ == "__main__":
    print("=" * 60)
    print("Setting up Vietnamese ASR + MT Project on Colab")
    print("=" * 60)

    install_requirements()
    mount_drive()
    setup_directories()

    print("\n" + "=" * 60)
    print("Setup completed!")
    print("Next steps:")
    print("1. Run download_datasets.py to download datasets")
    print("2. Run process_dataset.py to process datasets")
    print("=" * 60)
