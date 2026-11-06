"""
Main pipeline script for Vietnamese ASR + MT Project
Runs the complete data processing pipeline
"""
import os
import sys
import argparse

def stage1_download():
    """Stage 1: Download datasets"""
    print("=" * 60)
    print("STAGE 1: Downloading Datasets")
    print("=" * 60)

    from download_datasets import download_vivos, download_common_voice_vietnamese

    raw_dir = "D:/Truong/data/raw"
    os.makedirs(raw_dir, exist_ok=True)

    # Download VIVOS
    vivos = download_vivos(raw_dir)

    # Download Common Voice
    cv = download_common_voice_vietnamese(raw_dir)

    print("\nStage 1 completed!")

def stage2_process():
    """Stage 2: Process datasets"""
    print("=" * 60)
    print("STAGE 2: Processing Datasets")
    print("=" * 60)

    from process_dataset import DatasetProcessor

    raw_dir = "D:/Truong/data/raw"
    processed_dir = "D:/Truong/data/processed"

    processor = DatasetProcessor(raw_dir, processed_dir, sample_rate=16000)

    # Process VIVOS
    vivos_processed = processor.process_vivos()

    # Process Common Voice
    cv_processed = processor.process_common_voice()

    print("\nStage 2 completed!")

def stage3_eda():
    """Stage 3: Run EDA"""
    print("=" * 60)
    print("STAGE 3: Running Exploratory Data Analysis")
    print("=" * 60)

    print("Please run the EDA notebook manually:")
    print("  jupyter notebook notebooks/eda.ipynb")
    print("or in Colab:")
    print("  Upload the notebook and run all cells")

def main():
    parser = argparse.ArgumentParser(description="Vietnamese ASR + MT Pipeline")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["download", "process", "eda", "all"],
                        help="Which stage to run")

    args = parser.parse_args()

    if args.stage in ["download", "all"]:
        stage1_download()

    if args.stage in ["process", "all"]:
        stage2_process()

    if args.stage in ["eda", "all"]:
        stage3_eda()

    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
