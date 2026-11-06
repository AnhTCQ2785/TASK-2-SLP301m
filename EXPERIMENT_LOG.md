# Nhật ký thực nghiệm (Experiment Log)

## Baseline Training

### Run 1
- **Date**: 2026-11-06
- **Configuration**:
  - Model: Whisper Small
  - Subset: 500 samples
  - Epochs: 2
  - Batch size: 1
  - Learning rate: 1e-5
  - Seed: 42
- **Results**:
  - Epoch 1: Loss = 1.1498, WER = 0.6581
  - Epoch 2: Loss = 0.2504, WER = 0.6439
  - Final WER: 0.6439
- **Training Time**: ~20 minutes
- **Issues Encountered**:
  - Tensor shape mismatch → Fixed with .squeeze(0)
  - Batch collation errors → Fixed with batch_size=1
  - Decoding errors → Fixed with individual decode
- **Observations**: Loss decreased significantly from epoch 1 to 2, model converging well
- **Status**: ✅ Success

## Zero-shot Evaluation

### Run 1
- **Date**: 2026-11-06
- **Configuration**:
  - Model: Whisper Small (pretrained, no fine-tuning)
  - Test samples: 100 (subset)
  - Language: Vietnamese
- **Results**:
  - WER: 1.0102
  - CER: 1.4318
- **Evaluation Time**: ~1 minute
- **Observations**: High error rate, fine-tuning needed for Vietnamese
- **Status**: ✅ Success

## Fine-tuned Evaluation

### Run 1
- **Date**: 2026-11-06
- **Configuration**:
  - Model: Whisper Small (fine-tuned on VIVOS)
  - Test samples: 760 (full test set)
  - Seed: 42
- **Results**:
  - WER: 0.5784
  - CER: 0.3475
  - Improvement vs Zero-shot: +0.4318 (42.7%)
- **Evaluation Time**: ~9 minutes
- **Error Analysis**:
  - Substitutions: 53.96%
  - Insertions: 6.04%
  - Deletions: 0.00%
  - WER by length: Short=0.5932, Medium=0.5885, Long=0.4879
- **Observations**: Significant improvement over zero-shot, substitutions are main error type
- **Status**: ✅ Success

## V1: SpecAugment

### Run 1
- **Date**: 2026-11-06
- **Configuration**:
  - Baseline + SpecAugment
  - Time masking: 10% of audio
  - Probability: 50%
  - Subset: 500 samples
  - Epochs: 2
  - Batch size: 1
- **Results**:
  - Epoch 1: WER = 1.9578 (poor early performance)
  - Epoch 2: WER = 0.5727
  - Final WER: 0.5727
  - Improvement vs Baseline: +0.0711 (11.0%)
- **Training Time**: ~25 minutes
- **Issues**: Poor performance in epoch 1 due to augmentation, recovered in epoch 2
- **Observations**: Data augmentation helps generalization after initial training
- **Status**: ✅ Success

## V2: Cosine Learning Rate

### Run 1
- **Date**: 2026-11-06
- **Configuration**:
  - Baseline + Cosine LR schedule
  - Initial LR: 5e-5
  - Warmup: 10% of steps
  - Schedule: Cosine annealing
  - Subset: 500 samples
  - Epochs: 2
  - Batch size: 1
- **Results**:
  - Epoch 1: WER = 0.7590, LR = 0.000029
  - Epoch 2: WER = 0.5461, LR = 0.000000
  - Final WER: 0.5461
  - Improvement vs Baseline: +0.0978 (15.2%)
- **Training Time**: ~25 minutes
- **Observations**: Best performing method, cosine schedule helps convergence
- **Status**: ✅ Success

## V3: Larger Model (Whisper Base)

### Run 1
- **Date**: 2026-11-06
- **Configuration**:
  - Model: Whisper Base (vs Whisper Small)
  - Subset: 500 samples
  - Epochs: 2
  - Batch size: 1
  - Learning rate: 1e-5
- **Results**:
  - Epoch 1: WER = 0.8720
  - Epoch 2: WER = 0.6759
  - Final WER: 0.6759
  - Improvement vs Baseline: -0.0321 (-5.0%)
- **Training Time**: ~15 minutes (faster due to different architecture)
- **Issues**: Model underperforms baseline
- **Observations**: Larger model needs more data/epochs to show benefits
- **Status**: ✅ Success (but method discarded)

## Summary Statistics

| Metric | Baseline | V1 | V2 | V3 |
|--------|----------|----|----|----|
| Final WER | 0.6439 | 0.5727 | 0.5461 | 0.6759 |
| Training Time | ~20min | ~25min | ~25min | ~15min |
| Success Rate | 100% | 100% | 100% | 100% |
| Total Runs | 1 | 1 | 1 | 1 |

## Notes

- All experiments used subset of 500 samples for quick iteration
- Batch size constrained to 1 due to tensor shape issues
- Single seed (42) used due to time constraints
- No W&B/TensorBoard logs used (simple print statements instead)
- All training done on Google Colab GPU
