# Slide Content - Task 2 Presentation

## Slide 1: Title
**Vietnamese ASR with Whisper Fine-tuning**
- Task 2: Automatic Speech Recognition
- Dataset: VIVOS
- Model: OpenAI Whisper
- Team: [Team Name]

## Slide 2: Problem Statement
**Why Vietnamese ASR?**
- Vietnamese has complex diacritics and tones
- Limited ASR resources for Vietnamese
- Whisper pretrained on multilingual data
- Need to fine-tune for Vietnamese

**Goal**: Improve Vietnamese ASR performance through fine-tuning and proposed methods

## Slide 3: Dataset Analysis (EDA)
**VIVOS Dataset Overview**
- Training: 11,660 samples (14.92 hours)
- Test: 760 samples (0.75 hours)
- Speakers: 46 training, 19 test
- Avg duration: 4.61s (train), 3.53s (test)
- Avg text length: 13.2 words (train), 10.2 words (test)

**Key Findings**:
- Training set 15x larger than test
- Unseen speakers in test set
- Balanced speaker distribution in training

## Slide 4: Pipeline Dữ liệu
**Data Processing Pipeline**
1. Load VIVOS dataset
2. Create JSON manifests (audio_path, text, duration, speaker_id)
3. Normalize paths (Windows → Linux)
4. Resample audio to 16kHz
5. Split train/test (93.6% / 6.4%)
6. Subset selection (500 samples for experiments)

**Output**: JSON manifests + WAV audio files

## Slide 5: Baseline Implementation
**Model Configuration**
- Base model: Whisper Small
- Subset: 500 samples
- Epochs: 2
- Batch size: 1
- Learning rate: 1e-5
- Optimizer: AdamW

**Training Results**:
- Epoch 1: Loss = 1.1498, WER = 0.6581
- Epoch 2: Loss = 0.2504, WER = 0.6439
- Final WER: 0.6439

## Slide 6: Baseline & Gap Analysis
**Zero-shot vs Fine-tuned**
| Metric | Zero-shot | Fine-tuned | Improvement |
|--------|-----------|------------|-------------|
| WER | 1.0102 | 0.5784 | +42.7% |
| CER | 1.4318 | 0.3475 | +75.7% |

**Gap Analysis**:
- Fine-tuning significantly improves performance
- Main improvement from adapting to Vietnamese
- Substitutions: 53.96%, Insertions: 6.04%, Deletions: 0%
- Longer texts have lower WER (more context helps)

## Slide 7: Proposed Method V1 - SpecAugment
**Hypothesis**: Time masking improves generalization

**Implementation**:
- Random time masking (10% of audio)
- Applied to 50% of training samples
- No augmentation during testing

**Results**:
- Epoch 1: WER = 1.9578
- Epoch 2: WER = 0.5727
- Improvement: +11.0% vs baseline

**Conclusion**: Keep - Effective after initial training

## Slide 8: Proposed Method V2 - Cosine LR
**Hypothesis**: Cosine annealing improves convergence

**Implementation**:
- Initial LR: 5e-5
- Warmup: 10% of steps
- Cosine annealing schedule

**Results**:
- Epoch 1: WER = 0.7590
- Epoch 2: WER = 0.5461
- Improvement: +15.2% vs baseline

**Conclusion**: Keep - Best performing method

## Slide 9: Proposed Method V3 - Larger Model
**Hypothesis**: Larger model capacity improves performance

**Implementation**:
- Model: Whisper Base (vs Whisper Small)
- Same training configuration

**Results**:
- Epoch 1: WER = 0.8720
- Epoch 2: WER = 0.6759
- Improvement: -5.0% vs baseline

**Conclusion**: Discard - Underperforms due to limited data

## Slide 10: Version Comparison
| Version | WER | Improvement | Decision |
|---------|-----|-------------|----------|
| Baseline | 0.6439 | - | - |
| V1 (SpecAugment) | 0.5727 | +11.0% | Keep |
| V2 (Cosine LR) | 0.5461 | +15.2% | **Best** |
| V3 (Whisper Base) | 0.6759 | -5.0% | Discard |

**Winner**: V2 (Cosine Learning Rate)

## Slide 11: Feasibility & Limitations
**Feasibility**: ✅ Yes
- Minimal computational overhead
- Easy to implement
- Consistent improvement (15.2%)
- Same training time as baseline

**Limitations**:
- Small dataset subset (500/11,660)
- Limited epochs (2)
- Single seed (no statistical significance)
- Batch size constraint (1)
- Vietnamese diacritic errors remain

## Slide 12: Conclusion & Future Work
**Conclusion**:
- Best method: V2 (Cosine LR) with WER = 0.5461
- Total improvement: 15.2% over baseline, 45.9% over zero-shot
- Feasible and effective approach

**Future Work**:
- Train on full dataset (11,660 samples)
- Increase epochs to 5-10
- Test multiple seeds (42, 123, 456)
- Combine V1 + V2 for synergy
- Improve batch size with better hardware
- Vietnamese-specific preprocessing

## Slide 13: Q&A
**Questions?**
