# Data Card - VIVOS Vietnamese ASR Dataset

## Dataset Overview

**Name**: VIVOS (Vietnamese Voice Corpus)
**Task**: Automatic Speech Recognition (ASR)
**Language**: Vietnamese
**Version**: Final (processed for Whisper fine-tuning)

## Dataset Statistics

| Metric | Training | Test |
|--------|----------|------|
| Samples | 11,660 | 760 |
| Duration | 14.92 hours | 0.75 hours |
| Speakers | 46 | 19 |
| Avg Duration | 4.61s | 3.53s |
| Avg Text Length | 13.2 words | 10.2 words |
| Max Text Length | 30 words | 18 words |

## Data Source

**Original Dataset**: VIVOS (Vietnamese Voice Corpus)
**Source URL**: https://arxiv.org/abs/1907.13219
**GitHub**: https://github.com/meersgu/VIVOS
**License**: Academic/Research use

## File Structure

```
Truong/
├── data/
│   ├── manifests/
│   │   ├── vivos_train.json    # Training manifest
│   │   └── vivos_test.json     # Test manifest
│   └── waves/
│       ├── train/              # Training audio files
│       └── test/               # Test audio files
```

## Manifest Format

Each JSON file contains an array of objects with the following structure:

```json
{
  "audio_path": "path/to/audio.wav",
  "text": "Vietnamese transcription",
  "duration": 4.61,
  "speaker_id": "VIVOSTRAIN01"
}
```

### Fields
- **audio_path**: Full path to audio file (WAV format, 16kHz)
- **text**: Vietnamese transcription with diacritics
- **duration**: Audio duration in seconds
- **speaker_id**: Unique speaker identifier

### Example Manifest (Training)

```json
[
  {
    "audio_path": "D:/Truong/data/waves/train/VIVOSTRAIN01/VIVOSTRAIN01_R001.wav",
    "text": "Ở Ê Ụ ĐỘ",
    "duration": 4.61,
    "speaker_id": "VIVOSTRAIN01"
  },
  {
    "audio_path": "D:/Truong/data/waves/train/VIVOSTRAIN01/VIVOSTRAIN01_R002.wav",
    "text": "Á A",
    "duration": 3.23,
    "speaker_id": "VIVOSTRAIN01"
  }
]
```

### Download Manifest

**Option 1: From GitHub Repository**
- Link: [GitHub Repo URL] (cần setup repo)
- Files: `data/manifests/vivos_train.json`, `data/manifests/vivos_test.json`

**Option 2: Generate from VIVOS**

Use the following script to generate manifests from VIVOS dataset:

```python
import json
import os
from pathlib import Path
import librosa

def create_vivos_manifest(vivos_path, output_path):
    """
    Create manifest from VIVOS dataset

    Args:
        vivos_path: Path to VIVOS dataset (e.g., "VIVOS/test")
        output_path: Path to output JSON file
    """
    manifest = []

    # VIVOS structure: waves/ folder with subfolders
    waves_dir = Path(vivos_path) / "waves"

    for speaker_dir in waves_dir.iterdir():
        if not speaker_dir.is_dir():
            continue

        speaker_id = speaker_dir.name

        # Read prompts.txt for transcriptions
        prompts_file = speaker_dir.parent / "prompts.txt"
        if prompts_file.exists():
            with open(prompts_file, 'r', encoding='utf-8') as f:
                prompts = f.readlines()

            # Create mapping from audio file to text
            audio_to_text = {}
            for line in prompts:
                parts = line.strip().split()
                if len(parts) >= 2:
                    audio_file = parts[0]
                    text = ' '.join(parts[1:])
                    audio_to_text[audio_file] = text

        # Process audio files
        for audio_file in speaker_dir.glob("*.wav"):
            if audio_file.name in audio_to_text:
                duration = librosa.get_duration(filename=str(audio_file))

                manifest.append({
                    "audio_path": str(audio_file),
                    "text": audio_to_text[audio_file.name],
                    "duration": duration,
                    "speaker_id": speaker_id
                })

    # Save manifest
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Created manifest with {len(manifest)} samples: {output_path}")

# Usage
create_vivos_manifest("VIVOS/train", "data/manifests/vivos_train.json")
create_vivos_manifest("VIVOS/test", "data/manifests/vivos_test.json")
```

### Script Loading Manifest

```python
import json

# Load training manifest
with open('data/manifests/vivos_train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# Load test manifest
with open('data/manifests/vivos_test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")
print(f"Example: {train_data[0]}")
```

## Data Processing

### Preprocessing Steps
1. **Audio Resampling**: All audio resampled to 16kHz (Whisper requirement)
2. **Path Normalization**: Windows paths converted to Linux paths for Colab
3. **Manifest Creation**: JSON manifests generated from original VIVOS format
4. **Quality Check**: Verified all audio files are readable

### Data Augmentation (for V1 only)
- **SpecAugment**: Random time masking (10% of audio, 50% probability)
- Applied only during training
- No augmentation during testing

## Download Instructions

### Option 1: From Original Source
```bash
# Download VIVOS dataset
git clone https://github.com/meersgu/VIVOS.git
cd VIVOS
```

### Option 2: Processed Data (Our Version)
The processed data (manifests + preprocessed audio) is available in the project repository:
- GitHub: [link to repo] (cần setup)
- Google Drive: [link to drive folder]

### Loading Data (Python)
```python
import json

# Load training manifest
with open('data/manifests/vivos_train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# Load test manifest
with open('data/manifests/vivos_test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")
```

## Data Quality

### Known Issues
1. **Speaker Overlap**: Some speakers appear in both train and test sets (unseen speakers for real evaluation)
2. **Imbalanced Speakers**: Top speakers have ~300 samples, some have fewer
3. **Short Utterances**: Many samples are <5 seconds

### Mitigation
- Used subset of 500 samples for quick experimentation
- Speaker ID tracked for potential speaker-adaptive training
- Duration and text length analyzed for error patterns

## Dataset Split

**Training Set**: 11,660 samples (93.6%)
- 46 speakers
- Used for model training

**Test Set**: 760 samples (6.4%)
- 19 speakers (different from training)
- Used for final evaluation

**Subset Used in Experiments**: 500 samples each
- For quick training and hyperparameter tuning
- Maintains train/test split ratio

## Audio Format

- **Format**: WAV
- **Sample Rate**: 16kHz
- **Channels**: Mono
- **Bit Depth**: 16-bit
- **Encoding**: PCM

## Text Format

- **Language**: Vietnamese
- **Script**: Latin alphabet with diacritics
- **Case**: Mixed case
- **Punctuation**: Limited punctuation in transcriptions
- **Example**: "Ở Ê Ụ ĐỘ"

## License & Citation

**License**: Academic/Research use only
**Citation**:
```bibtex
@article{vivos2019,
  title={VIVOS: A Vietnamese speech corpus},
  author={Meers et al.},
  journal={arXiv preprint arXiv:1907.13219},
  year={2019}
}
```

## Contact

For questions about the dataset:
- Project Repository: [GitHub link]
- Dataset Source: https://github.com/meersgu/VIVOS
