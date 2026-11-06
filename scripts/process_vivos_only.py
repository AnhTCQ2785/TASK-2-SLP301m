"""
Script to process VIVOS dataset only (from Zenodo download)
"""
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob

def parse_vivos_metadata():
    """Parse VIVOS metadata from extracted files"""
    print("Parsing VIVOS metadata...")

    vivos_dir = "D:/Truong/data/raw/vivos/vivos"  # Note: nested vivos folder

    # VIVOS structure after extraction
    # vivos/vivos/
    #   train/
    #     waves/
    #       VIVOSSPK01/
    #         VIVOSSPK01_R001.wav
    #     transcripts.txt
    #   test/
    #     waves/
    #     transcripts.txt

    def load_split(split_name):
        split_dir = os.path.join(vivos_dir, split_name)
        waves_dir = os.path.join(split_dir, "waves")
        prompt_file = os.path.join(split_dir, "prompts.txt")

        if not os.path.exists(prompt_file):
            print(f"Warning: {prompt_file} not found")
            return []

        # Parse prompts.txt
        # Format: SPEAKER_ID_R001\tTEXT
        samples = []
        with open(prompt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)  # Split on first whitespace
                if len(parts) >= 2:
                    audio_id = parts[0]
                    transcript = parts[1]

                    # Extract speaker ID from audio_id
                    # Example: VIVOSSPK01_R001 -> VIVOSSPK01
                    speaker_id = audio_id.split("_")[0] if "_" in audio_id else audio_id

                    # Construct audio path
                    audio_filename = f"{audio_id}.wav"
                    audio_path = os.path.join(waves_dir, speaker_id, audio_filename)

                    if os.path.exists(audio_path):
                        samples.append({
                            "audio_path": audio_path,
                            "text": transcript,
                            "speaker_id": speaker_id,
                            "split": split_name
                        })

        print(f"{split_name}: {len(samples)} samples")
        return samples

    train_samples = load_split("train")
    test_samples = load_split("test")

    return train_samples, test_samples

def clean_text(text):
    """Clean and normalize Vietnamese text"""
    if text is None:
        return ""
    text = str(text)
    text = " ".join(text.split())
    vietnamese_chars = "aáàảãạăắằẳẵặâấầẩẫậeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵđ"
    basic_punctuation = ".,!?;:'\"-"
    cleaned = ""
    for char in text:
        if char.lower() in vietnamese_chars or char.isdigit() or char.isspace() or char in basic_punctuation:
            cleaned += char
        elif char in "()[]{}":
            cleaned += " "
        else:
            cleaned += " "
    return " ".join(cleaned.split()).strip()

def get_audio_duration(audio_path):
    """Get audio duration using librosa"""
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr
        return duration
    except:
        return 0

def process_vivos():
    """Process VIVOS dataset"""
    print("=" * 60)
    print("Processing VIVOS Dataset")
    print("=" * 60)

    # Load metadata
    train_samples, test_samples = parse_vivos_metadata()

    # Clean text
    print("Cleaning text...")
    for sample in tqdm(train_samples):
        sample["text"] = clean_text(sample["text"])
    for sample in tqdm(test_samples):
        sample["text"] = clean_text(sample["text"])

    # Filter empty text
    train_samples = [s for s in train_samples if s["text"]]
    test_samples = [s for s in test_samples if s["text"]]

    print(f"After filtering empty text: Train={len(train_samples)}, Test={len(test_samples)}")

    # Get durations (this might take a while)
    print("Getting audio durations (this may take a few minutes)...")
    try:
        import librosa
        for sample in tqdm(train_samples):
            sample["duration"] = get_audio_duration(sample["audio_path"])
        for sample in tqdm(test_samples):
            sample["duration"] = get_audio_duration(sample["audio_path"])

        # Filter by duration
        print("Filtering by duration (0.5s - 20s)...")
        train_samples = [s for s in train_samples if 0.5 <= s["duration"] <= 20.0]
        test_samples = [s for s in test_samples if 0.5 <= s["duration"] <= 20.0]

        print(f"After duration filter: Train={len(train_samples)}, Test={len(test_samples)}")
    except ImportError:
        print("librosa not installed, skipping duration filter")
        for sample in train_samples:
            sample["duration"] = 0
        for sample in test_samples:
            sample["duration"] = 0

    # Create manifests
    output_dir = "D:/Truong/data/manifests"
    os.makedirs(output_dir, exist_ok=True)

    print("Creating manifests...")
    with open(os.path.join(output_dir, "vivos_train.json"), "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "vivos_test.json"), "w", encoding="utf-8") as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)

    print(f"Manifests saved to {output_dir}")

    # Statistics
    print("\n" + "=" * 60)
    print("VIVOS Statistics")
    print("=" * 60)

    train_duration = sum(s["duration"] for s in train_samples) / 3600
    test_duration = sum(s["duration"] for s in test_samples) / 3600

    train_speakers = len(set(s["speaker_id"] for s in train_samples))
    test_speakers = len(set(s["speaker_id"] for s in test_samples))

    print(f"\nTrain:")
    print(f"  Samples: {len(train_samples)}")
    print(f"  Duration: {train_duration:.2f} hours")
    print(f"  Speakers: {train_speakers}")

    print(f"\nTest:")
    print(f"  Samples: {len(test_samples)}")
    print(f"  Duration: {test_duration:.2f} hours")
    print(f"  Speakers: {test_speakers}")

    print("\n" + "=" * 60)
    print("VIVOS processing completed!")
    print("=" * 60)

    return train_samples, test_samples

if __name__ == "__main__":
    process_vivos()
