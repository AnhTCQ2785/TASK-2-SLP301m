"""
Script to process Vietnamese ASR datasets: cleaning, balancing, and creating manifests
"""
import os
import json
import pandas as pd
from datasets import load_from_disk
from datasets import Audio
import torch
import torchaudio
from collections import Counter
import numpy as np
from tqdm import tqdm

class DatasetProcessor:
    def __init__(self, raw_data_dir, processed_data_dir, sample_rate=16000):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.sample_rate = sample_rate
        os.makedirs(processed_data_dir, exist_ok=True)

    def load_vivos(self):
        """Load VIVOS dataset"""
        print("Loading VIVOS dataset...")
        vivos_path = os.path.join(self.raw_data_dir, "vivos")
        vivos = load_from_disk(vivos_path)
        return vivos

    def load_common_voice(self):
        """Load Common Voice Vietnamese dataset"""
        print("Loading Common Voice Vietnamese dataset...")
        cv_path = os.path.join(self.raw_data_dir, "common_voice_vi")
        cv = load_from_disk(cv_path)
        return cv

    def clean_text(self, text):
        """Clean and normalize text"""
        if text is None:
            return ""

        # Convert to string
        text = str(text)

        # Remove extra whitespace
        text = " ".join(text.split())

        # Remove special characters that are not Vietnamese letters or basic punctuation
        # Keep: Vietnamese letters, numbers, spaces, basic punctuation
        vietnamese_chars = "aáàảãạăắằẳẵặâấầẩẫậeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵđ"
        basic_punctuation = ".,!?;:'\"-"

        # Keep Vietnamese letters, numbers, spaces, and basic punctuation
        cleaned = ""
        for char in text:
            if char.lower() in vietnamese_chars or char.isdigit() or char.isspace() or char in basic_punctuation:
                cleaned += char
            elif char in "()[]{}":
                cleaned += " "  # Replace brackets with space
            else:
                cleaned += " "  # Replace other special chars with space

        # Clean up multiple spaces again
        cleaned = " ".join(cleaned.split())

        return cleaned.strip()

    def filter_by_duration(self, dataset, min_duration=0.5, max_duration=20.0):
        """Filter samples by audio duration"""
        print(f"Filtering by duration: {min_duration}s - {max_duration}s...")

        def duration_filter(example):
            audio = example["audio"]
            duration = audio["array"].shape[0] / audio["sampling_rate"]
            return min_duration <= duration <= max_duration

        filtered = dataset.filter(duration_filter, num_proc=4)
        print(f"Samples after duration filter: {len(filtered)}")
        return filtered

    def balance_dataset(self, dataset, strategy="downsample", target_samples=None):
        """
        Balance dataset by speaker or other criteria
        strategy: 'downsample', 'upsample', 'none'
        """
        print(f"Balancing dataset with strategy: {strategy}")

        if strategy == "none":
            return dataset

        # Get speaker distribution
        if "speaker_id" in dataset.column_names:
            speaker_col = "speaker_id"
        elif "client_id" in dataset.column_names:
            speaker_col = "client_id"
        else:
            print("No speaker column found, skipping balancing")
            return dataset

        speaker_counts = Counter(dataset[speaker_col])
        print(f"Speaker distribution before balancing:")
        for speaker, count in sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {speaker}: {count}")

        if strategy == "downsample":
            # Downsample to match the minimum class
            min_count = min(speaker_counts.values())
            print(f"Downsampling to {min_count} samples per speaker")

            balanced_samples = []
            for speaker in speaker_counts:
                speaker_data = dataset.filter(lambda x: x[speaker_col] == speaker)
                # Randomly sample min_count samples
                if len(speaker_data) > min_count:
                    indices = np.random.choice(len(speaker_data), min_count, replace=False)
                    for idx in indices:
                        balanced_samples.append(speaker_data[idx])
                else:
                    for item in speaker_data:
                        balanced_samples.append(item)

            # Convert back to dataset
            from datasets import Dataset
            balanced = Dataset.from_list(balanced_samples)

        elif strategy == "upsample":
            # Upsample to match the maximum class (limited)
            max_count = max(speaker_counts.values())
            target_count = min(max_count, target_samples if target_samples else max_count)
            print(f"Upsampling to {target_count} samples per speaker")

            balanced_samples = []
            for speaker in speaker_counts:
                speaker_data = dataset.filter(lambda x: x[speaker_col] == speaker)
                current_count = len(speaker_data)

                if current_count < target_count:
                    # Upsample by repetition
                    repeats = target_count // current_count
                    remainder = target_count % current_count

                    for _ in range(repeats):
                        for item in speaker_data:
                            balanced_samples.append(item)

                    # Add remainder samples
                    indices = np.random.choice(current_count, remainder, replace=False)
                    for idx in indices:
                        balanced_samples.append(speaker_data[idx])
                else:
                    for item in speaker_data:
                        balanced_samples.append(item)

            from datasets import Dataset
            balanced = Dataset.from_list(balanced_samples)

        print(f"Dataset size after balancing: {len(balanced)}")
        return balanced

    def create_manifest(self, dataset, output_path, split_name):
        """Create manifest file for dataset"""
        print(f"Creating manifest for {split_name}...")

        manifest = []
        for idx, item in enumerate(tqdm(dataset)):
            audio = item["audio"]

            # Get transcript
            if "sentence" in item:
                transcript = self.clean_text(item["sentence"])
            elif "transcription" in item:
                transcript = self.clean_text(item["transcription"])
            else:
                transcript = ""

            # Skip empty transcripts
            if not transcript:
                continue

            entry = {
                "audio_path": audio["path"],
                "duration": audio["array"].shape[0] / audio["sampling_rate"],
                "text": transcript,
                "speaker_id": item.get("speaker_id", item.get("client_id", "unknown")),
                "sample_rate": audio["sampling_rate"]
            }
            manifest.append(entry)

        # Save manifest
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        print(f"Manifest saved to {output_path}")
        print(f"Total entries: {len(manifest)}")

        return manifest

    def process_vivos(self):
        """Process VIVOS dataset"""
        print("\n" + "=" * 60)
        print("Processing VIVOS Dataset")
        print("=" * 60)

        vivos = self.load_vivos()

        # Process train split
        print("\nProcessing train split...")
        train = vivos["train"]

        # Filter by duration
        train = self.filter_by_duration(train, min_duration=0.5, max_duration=20.0)

        # Clean text
        def clean_text_batch(example):
            example["sentence"] = self.clean_text(example["sentence"])
            return example

        train = train.map(clean_text_batch, num_proc=4)

        # Filter empty text
        train = train.filter(lambda x: len(x["sentence"]) > 0)

        # Balance dataset (optional - can comment out)
        # train = self.balance_dataset(train, strategy="downsample")

        # Create manifest
        train_manifest = self.create_manifest(
            train,
            os.path.join(self.processed_data_dir, "manifests", "vivos_train.json"),
            "train"
        )

        # Process test split
        print("\nProcessing test split...")
        test = vivos["test"]
        test = self.filter_by_duration(test, min_duration=0.5, max_duration=20.0)
        test = test.map(clean_text_batch, num_proc=4)
        test = test.filter(lambda x: len(x["sentence"]) > 0)

        test_manifest = self.create_manifest(
            test,
            os.path.join(self.processed_data_dir, "manifests", "vivos_test.json"),
            "test"
        )

        # Save processed dataset
        from datasets import DatasetDict
        processed_vivos = DatasetDict({"train": train, "test": test})
        processed_vivos.save_to_disk(os.path.join(self.processed_data_dir, "vivos_processed"))

        print("\nVIVOS processing completed!")
        return processed_vivos

    def process_common_voice(self):
        """Process Common Voice Vietnamese dataset"""
        print("\n" + "=" * 60)
        print("Processing Common Voice Vietnamese Dataset")
        print("=" * 60)

        cv = self.load_common_voice()

        # Process train split
        print("\nProcessing train split...")
        train = cv["train"]

        # Filter by duration
        train = self.filter_by_duration(train, min_duration=0.5, max_duration=20.0)

        # Clean text
        def clean_text_batch(example):
            example["sentence"] = self.clean_text(example["sentence"])
            return example

        train = train.map(clean_text_batch, num_proc=4)

        # Filter empty text
        train = train.filter(lambda x: len(x["sentence"]) > 0)

        # Balance dataset (optional - can comment out)
        # train = self.balance_dataset(train, strategy="downsample")

        # Create manifest
        train_manifest = self.create_manifest(
            train,
            os.path.join(self.processed_data_dir, "manifests", "cv_train.json"),
            "train"
        )

        # Process validation split
        print("\nProcessing validation split...")
        validation = cv["validation"]
        validation = self.filter_by_duration(validation, min_duration=0.5, max_duration=20.0)
        validation = validation.map(clean_text_batch, num_proc=4)
        validation = validation.filter(lambda x: len(x["sentence"]) > 0)

        val_manifest = self.create_manifest(
            validation,
            os.path.join(self.processed_data_dir, "manifests", "cv_val.json"),
            "validation"
        )

        # Process test split
        print("\nProcessing test split...")
        test = cv["test"]
        test = self.filter_by_duration(test, min_duration=0.5, max_duration=20.0)
        test = test.map(clean_text_batch, num_proc=4)
        test = test.filter(lambda x: len(x["sentence"]) > 0)

        test_manifest = self.create_manifest(
            test,
            os.path.join(self.processed_data_dir, "manifests", "cv_test.json"),
            "test"
        )

        # Save processed dataset
        from datasets import DatasetDict
        processed_cv = DatasetDict({"train": train, "validation": validation, "test": test})
        processed_cv.save_to_disk(os.path.join(self.processed_data_dir, "cv_processed"))

        print("\nCommon Voice Vietnamese processing completed!")
        return processed_cv

    def get_statistics(self, dataset):
        """Get dataset statistics"""
        stats = {
            "total_samples": len(dataset),
            "speakers": len(set(dataset.get("speaker_id", dataset.get("client_id", [])))),
            "avg_duration": np.mean([item["audio"]["array"].shape[0] / item["audio"]["sampling_rate"] for item in dataset]),
            "total_duration": sum([item["audio"]["array"].shape[0] / item["audio"]["sampling_rate"] for item in dataset])
        }
        return stats

if __name__ == "__main__":
    raw_data_dir = "D:/Truong/data/raw"
    processed_data_dir = "D:/Truong/data/processed"

    processor = DatasetProcessor(raw_data_dir, processed_data_dir, sample_rate=16000)

    # Process VIVOS
    vivos_processed = processor.process_vivos()

    # Process Common Voice
    cv_processed = processor.process_common_voice()

    print("\n" + "=" * 60)
    print("Dataset processing completed!")
    print("=" * 60)
