# Vietnamese ASR with Whisper Fine-tuning

Task 2: Automatic Speech Recognition for Vietnamese using OpenAI Whisper model fine-tuning on VIVOS dataset.

## 📋 Project Overview

This project implements and evaluates Vietnamese ASR using OpenAI Whisper model with fine-tuning on the VIVOS dataset. We propose and compare three improvement methods: SpecAugment, Cosine Learning Rate Schedule, and Larger Model.

## 🎯 Results Summary

| Version | WER | Improvement vs Baseline | Decision |
|---------|-----|-------------------------|----------|
| Baseline | 0.6439 | - | - |
| V1 (SpecAugment) | 0.5727 | +11.0% | Keep |
| V2 (Cosine LR) | 0.5461 | +15.2% | **Best** |
| V3 (Whisper Base) | 0.6759 | -5.0% | Discard |

**Best Method**: V2 (Cosine Learning Rate Schedule)

## 📁 Project Structure

```
Truong/
├── data/
│   ├── manifests/
│   │   ├── vivos_train.json    # Training manifest
│   │   └── vivos_test.json     # Test manifest
│   └── waves/                  # Audio files
├── scripts/
│   ├── train_baseline.py       # Baseline training script
│   ├── train_v1.py            # V1 (SpecAugment) training
│   ├── train_v2.py            # V2 (Cosine LR) training
│   ├── train_v3.py            # V3 (Whisper Base) training
│   └── compare_versions.py    # Version comparison
├── results/
│   ├── training_histories.json
│   ├── v1_history.json
│   ├── v2_history.json
│   ├── v3_history.json
│   └── baseline_results.json
├── models/
│   ├── whisper_vivos_seed42/
│   ├── whisper_vivos_v1_specaugment/
│   ├── whisper_vivos_v2_cosine/
│   └── whisper_vivos_v3_base/
├── checkpoints/               # Model checkpoints
├── COLAB_CELLS.md            # Colab notebook cells
├── TASK2_REPORT.md           # Detailed report
├── TASK2_CONCLUSION.md       # Conclusion
├── DATA_CARD.md              # Data documentation
├── EXPERIMENT_LOG.md         # Experiment log
├── SLIDE_CONTENT.md          # Presentation slides
├── CONTRIBUTION.md           # Team contribution
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers
- Google Colab (recommended) or GPU

### Installation

```bash
pip install torch torchaudio transformers datasets accelerate
pip install librosa soundfile jiwer sacrebleu
pip install matplotlib seaborn tensorboard tqdm
```

### Dataset Preparation

1. Download VIVOS dataset:
```bash
# From original source
wget https://github.com/meersgu/VIVOS/archive/refs/heads/master.zip
unzip master.zip
```

2. Create manifests (see `DATA_CARD.md` for details)

### Training

#### Option 1: Using Colab (Recommended)

1. Open `COLAB_CELLS.md`
2. Copy each cell to Google Colab
3. Run cells sequentially

#### Option 2: Local Training

```bash
# Baseline training
python scripts/train_baseline.py

# V1 (SpecAugment)
python scripts/train_v1.py

# V2 (Cosine LR)
python scripts/train_v2.py

# V3 (Whisper Base)
python scripts/train_v3.py

# Compare results
python scripts/compare_versions.py
```

## 📊 Dataset

**VIVOS (Vietnamese Voice Corpus)**
- Training: 11,660 samples (14.92 hours)
- Test: 760 samples (0.75 hours)
- Speakers: 46 training, 19 test
- Language: Vietnamese with diacritics

See `DATA_CARD.md` for detailed dataset information.

## 🔬 Experiments

### Baseline Configuration
- Model: Whisper Small
- Subset: 500 samples
- Epochs: 2
- Batch size: 1
- Learning rate: 1e-5
- Seed: 42

### Proposed Methods

#### V1: SpecAugment
- Random time masking (10% of audio)
- Applied to 50% of training samples
- Improvement: +11.0%

#### V2: Cosine Learning Rate
- Initial LR: 5e-5
- Warmup: 10% of steps
- Cosine annealing schedule
- Improvement: +15.2%

#### V3: Larger Model
- Model: Whisper Base
- Same configuration as baseline
- Result: -5.0% (discarded)

See `EXPERIMENT_LOG.md` for detailed experiment logs.

## 📈 Results

### Zero-shot vs Fine-tuned

| Metric | Zero-shot | Fine-tuned | Improvement |
|--------|-----------|------------|-------------|
| WER | 1.0102 | 0.5784 | +42.7% |
| CER | 1.4318 | 0.3475 | +75.7% |

### Error Analysis
- Substitutions: 53.96%
- Insertions: 6.04%
- Deletions: 0.00%

## 📝 Documentation

- **TASK2_REPORT.md**: Detailed technical report
- **TASK2_CONCLUSION.md**: Executive summary
- **DATA_CARD.md**: Dataset documentation
- **EXPERIMENT_LOG.md**: Experiment log
- **SLIDE_CONTENT.md**: Presentation slides
- **CONTRIBUTION.md**: Team contribution

## 🏷️ Version Tags

- `v1.0`: Baseline implementation
- `v1.1`: Added V1 (SpecAugment)
- `v1.2`: Added V2 (Cosine LR)
- `v1.3`: Added V3 (Whisper Base)
- `v1.4`: Final comparison and documentation

## 👥 Team

See `CONTRIBUTION.md` for team member contributions.

## 📄 License

This project is for academic/research purposes. The VIVOS dataset has its own license - please refer to the original source.

## 🙏 Acknowledgments

- OpenAI for Whisper model
- VIVOS dataset creators
- Hugging Face for Transformers library

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 🔗 Links

- [VIVOS Dataset](https://github.com/meersgu/VIVOS)
- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
