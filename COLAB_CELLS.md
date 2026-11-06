# CELL 1: Setup
import os
import shutil
from google.colab import drive

if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')
else:
    print("Drive already mounted")

print("=" * 60)
print("STEP 1: COPY DATA")
print("=" * 60)

if os.path.exists('/content/drive/MyDrive'):
    print("Drive mounted")
    !ls -la "/content/drive/MyDrive"
else:
    raise Exception("Drive not mounted")

zip_path = "/content/drive/MyDrive/TASK 2 SLP301m/Truong/Truong.zip"

if not os.path.exists(zip_path):
    print("Truong.zip not found")
    print(f"Path: {zip_path}")
    raise Exception("Truong.zip not found")

print(f"Found: {zip_path}")

print("Copying...")
!cp "{zip_path}" /content/Truong.zip

print("Unzipping...")
!unzip -q -o /content/Truong.zip -d /content/

if os.path.exists('/content/Truong'):
    print("Unzip success")
    %cd /content/Truong
else:
    raise Exception("Unzip failed")

print("Directory structure:")
!ls -la

print("Installing dependencies...")
!pip install -q torch torchaudio transformers datasets accelerate
!pip install -q librosa soundfile jiwer sacrebleu
!pip install -q matplotlib seaborn tensorboard tqdm

print("Setup complete")

# CELL 2: EDA
import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

print("=" * 60)
print("STEP 2: EDA")
print("=" * 60)

with open('data/manifests/vivos_train.json', 'r', encoding='utf-8') as f:
    train = json.load(f)
with open('data/manifests/vivos_test.json', 'r', encoding='utf-8') as f:
    test = json.load(f)

print(f"Train samples: {len(train)}")
print(f"Test samples: {len(test)}")

train_durations = [s['duration'] for s in train]
test_durations = [s['duration'] for s in test]
train_hours = sum(train_durations) / 3600
test_hours = sum(test_durations) / 3600

print(f"Train: {train_hours:.2f} hours (mean: {np.mean(train_durations):.2f}s)")
print(f"Test: {test_hours:.2f} hours (mean: {np.mean(test_durations):.2f}s)")

train_speakers = Counter([s['speaker_id'] for s in train])
test_speakers = Counter([s['speaker_id'] for s in test])
print(f"Train speakers: {len(train_speakers)}, avg {len(train)/len(train_speakers):.1f} samples/speaker")
print(f"Test speakers: {len(test_speakers)}, avg {len(test)/len(test_speakers):.1f} samples/speaker")

train_text_len = [len(s['text'].split()) for s in train]
test_text_len = [len(s['text'].split()) for s in test]
print(f"Train text length: mean {np.mean(train_text_len):.1f} words, max {max(train_text_len)}")
print(f"Test text length: mean {np.mean(test_text_len):.1f} words, max {max(test_text_len)}")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].hist(train_durations, bins=50, alpha=0.7, label='Train')
axes[0, 0].hist(test_durations, bins=50, alpha=0.7, label='Test')
axes[0, 0].set_xlabel('Duration (s)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Duration Distribution')
axes[0, 0].legend()

axes[0, 1].hist(train_text_len, bins=50, alpha=0.7, label='Train')
axes[0, 1].hist(test_text_len, bins=50, alpha=0.7, label='Test')
axes[0, 1].set_xlabel('Word Count')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Text Length Distribution')
axes[0, 1].legend()

top_train_speakers = train_speakers.most_common(20)
axes[1, 0].bar([s[0] for s in top_train_speakers], [s[1] for s in top_train_speakers])
axes[1, 0].set_xlabel('Speaker ID')
axes[1, 0].set_ylabel('Sample Count')
axes[1, 0].set_title('Top 20 Train Speakers')
axes[1, 0].tick_params(axis='x', rotation=45)

axes[1, 1].text(0.1, 0.9, f"Train sample:\n{train[0]['text']}",
                transform=axes[1, 1].transAxes, fontsize=9, verticalalignment='top')
axes[1, 1].text(0.1, 0.5, f"Test sample:\n{test[0]['text']}",
                transform=axes[1, 1].transAxes, fontsize=9, verticalalignment='top')
axes[1, 1].axis('off')
axes[1, 1].set_title('Sample Texts')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=100)
print("EDA plots saved: eda_plots.png")

print("=" * 60)
print("DATA STATISTICS")
print("=" * 60)
print(f"{'Metric':<20} {'Train':<15} {'Test':<15}")
print("-" * 50)
print(f"{'Samples':<20} {len(train):<15} {len(test):<15}")
print(f"{'Hours':<20} {train_hours:.2f}{'':<12} {test_hours:.2f}{'':<12}")
print(f"{'Speakers':<20} {len(train_speakers):<15} {len(test_speakers):<15}")
print(f"{'Avg duration (s)':<20} {np.mean(train_durations):.2f}{'':<12} {np.mean(test_durations):.2f}{'':<12}")
print(f"{'Avg word count':<20} {np.mean(train_text_len):.1f}{'':<12} {np.mean(test_text_len):.1f}{'':<12}")

print("EDA complete")

# CELL 3: Create Training Script
print("=" * 60)
print("STEP 3: CREATE TRAINING SCRIPT")
print("=" * 60)

training_script = '''
import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm
import jiwer
import librosa

USE_SUBSET = True
SUBSET_SIZE = 500
NUM_EPOCHS = 2 if USE_SUBSET else 5
BATCH_SIZE = 1
SEEDS = [42]

print("=" * 60)
print("TRAINING CONFIG")
print("=" * 60)
print(f"USE_SUBSET: {USE_SUBSET}")
print(f"SUBSET_SIZE: {SUBSET_SIZE if USE_SUBSET else 'Full dataset'}")
print(f"NUM_EPOCHS: {NUM_EPOCHS}")
print(f"BATCH_SIZE: {BATCH_SIZE}")
print(f"SEEDS: {SEEDS}")
print("=" * 60)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class VivosDataset(Dataset):
    def __init__(self, manifest_path, processor, max_samples=None):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.processor = processor

        for item in self.data:
            item['audio_path'] = item['audio_path'].replace('\\\\', '/')
            if 'D:/Truong/' in item['audio_path']:
                item['audio_path'] = item['audio_path'].replace('D:/Truong/', '/content/Truong/')
            if 'D:\\\\Truong\\\\' in item['audio_path']:
                item['audio_path'] = item['audio_path'].replace('D:\\\\Truong\\\\', '/content/Truong/')

        if max_samples:
            self.data = self.data[:max_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio, sr = librosa.load(item['audio_path'], sr=16000)
        inputs = self.processor(audio=audio, sampling_rate=16000, return_tensors="pt")
        labels = self.processor.tokenizer(item['text'], return_tensors="pt").input_ids.squeeze(0)
        inputs['labels'] = labels
        return inputs

def compute_wer(preds, labels, processor):
    pred_texts = []
    label_texts = []
    for pred, label in zip(preds, labels):
        pred_text = processor.decode(pred, skip_special_tokens=True)
        label_text = processor.decode(label, skip_special_tokens=True)
        pred_texts.append(pred_text)
        label_texts.append(label_text)
    return jiwer.wer(label_texts, pred_texts)

def train(seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_name = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_name, language="vi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

    train_dataset = VivosDataset("data/manifests/vivos_train.json", processor,
                                 max_samples=SUBSET_SIZE if USE_SUBSET else None)
    test_dataset = VivosDataset("data/manifests/vivos_test.json", processor,
                                max_samples=SUBSET_SIZE if USE_SUBSET else None)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    history = {'seed': seed, 'train_loss': [], 'test_wer': []}

    model.train()
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        total_loss = 0

        for batch in tqdm(train_loader):
            input_features = batch['input_features'].squeeze(0).to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Average loss: {avg_loss:.4f}")
        history['train_loss'].append(avg_loss)

        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'seed': seed,
        }, f"checkpoints/whisper_seed{seed}_epoch_{epoch+1}.pt")

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_features = batch['input_features'].squeeze(0).to(device)
                labels = batch['labels'].to(device)
                generated_ids = model.generate(input_features=input_features)
                all_preds.extend(generated_ids.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        wer = compute_wer(all_preds, all_labels, processor)
        print(f"Test WER: {wer:.4f}")
        history['test_wer'].append(wer)
        model.train()

    os.makedirs("models", exist_ok=True)
    model.save_pretrained(f"models/whisper_vivos_seed{seed}")
    processor.save_pretrained(f"models/whisper_vivos_seed{seed}")

    print(f"Training completed for seed {seed}")
    return history

def main():
    all_histories = []

    for seed in SEEDS:
        print(f"{'='*60}")
        print(f"TRAINING WITH SEED {seed}")
        print(f"{'='*60}")
        try:
            history = train(seed)
            all_histories.append(history)
        except Exception as e:
            print(f"Training failed for seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue

    os.makedirs("results", exist_ok=True)
    with open('results/training_histories.json', 'w', encoding='utf-8') as f:
        json.dump(all_histories, f, ensure_ascii=False, indent=2)
    print("Training histories saved")

    if all_histories:
        final_wers = [h['test_wer'][-1] for h in all_histories]
        mean_wer = np.mean(final_wers)
        std_wer = np.std(final_wers)

        print(f"{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"WER: {mean_wer:.4f} ± {std_wer:.4f}")
        print(f"Individual WERs: {final_wers}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()
'''

os.makedirs('scripts', exist_ok=True)
with open('scripts/train_baseline.py', 'w', encoding='utf-8') as f:
    f.write(training_script)

print("Training script created: scripts/train_baseline.py")

# CELL 4: Run Baseline Training
print("=" * 60)
print("STEP 4: BASELINE TRAINING")
print("=" * 60)

print("Starting baseline training...")
print("Estimated time: 30-60 minutes (500 samples, batch_size=1)")

!python scripts/train_baseline.py

print("Baseline training completed")

# CELL 5: Baseline Evaluation
import json
import torch
import matplotlib.pyplot as plt
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import jiwer
from tqdm import tqdm

print("=" * 60)
print("BASELINE EVALUATION")
print("=" * 60)

with open('results/training_histories.json', 'r', encoding='utf-8') as f:
    histories = json.load(f)

print(f"Loaded {len(histories)} training histories")

if len(histories) > 0:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for h in histories:
        plt.plot(range(1, len(h['train_loss']) + 1), h['train_loss'], label=f"Seed {h['seed']}")
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for h in histories:
        plt.plot(range(1, len(h['test_wer']) + 1), h['test_wer'], label=f"Seed {h['seed']}")
    plt.xlabel('Epoch')
    plt.ylabel('Test WER')
    plt.title('Test WER per Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('learning_curve.png')
    print("Learning curve saved: learning_curve.png")
else:
    print("No training histories to plot")

# CELL 6: Zero-shot vs Fine-tuned
import librosa

print("=" * 60)
print("ZERO-SHOT vs FINE-TUNED")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('data/manifests/vivos_test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

for item in test_data:
    item['audio_path'] = item['audio_path'].replace('\\', '/')
    if 'D:/Truong/' in item['audio_path']:
        item['audio_path'] = item['audio_path'].replace('D:/Truong/', '/content/Truong/')

print("Evaluating Zero-shot...")
model_zero = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
processor_zero = WhisperProcessor.from_pretrained("openai/whisper-small", language="vi", task="transcribe")

all_preds_zero = []
all_labels_zero = []

model_zero.eval()
with torch.no_grad():
    for item in tqdm(test_data[:100], desc="Zero-shot"):
        audio, sr = librosa.load(item['audio_path'], sr=16000)
        inputs = processor_zero(audio=audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        generated_ids = model_zero.generate(input_features=input_features)
        pred_text = processor_zero.batch_decode(generated_ids, skip_special_tokens=True)[0]
        all_preds_zero.append(pred_text)
        all_labels_zero.append(item['text'])

wer_zero = jiwer.wer(all_labels_zero, all_preds_zero)
cer_zero = jiwer.cer(all_labels_zero, all_preds_zero)

print(f"Zero-shot Results (100 samples):")
print(f"WER: {wer_zero:.4f}")
print(f"CER: {cer_zero:.4f}")

print("Evaluating Fine-tuned...")

if not histories or len(histories) == 0:
    print("ERROR: No training histories found. Training must have failed.")
    print("Please fix the training error and re-run CELL 4 (Baseline Training)")
    raise Exception("No training data available")

best_seed = histories[0]['seed']
model_path = f"models/whisper_vivos_seed{best_seed}"

if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    print("Please complete training first")
    raise Exception("Model not found")

processor = WhisperProcessor.from_pretrained(model_path, language="vi", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)

all_preds_finetuned = []
all_labels_finetuned = []

model.eval()
with torch.no_grad():
    for item in tqdm(test_data, desc="Fine-tuned"):
        audio, sr = librosa.load(item['audio_path'], sr=16000)
        inputs = processor(audio=audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        generated_ids = model.generate(input_features=input_features)
        pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        all_preds_finetuned.append(pred_text)
        all_labels_finetuned.append(item['text'])

wer_finetuned = jiwer.wer(all_labels_finetuned, all_preds_finetuned)
cer_finetuned = jiwer.cer(all_labels_finetuned, all_preds_finetuned)

print(f"Fine-tuned Results:")
print(f"WER: {wer_finetuned:.4f}")
print(f"CER: {cer_finetuned:.4f}")

print("=" * 60)
print("COMPARISON TABLE")
print("=" * 60)
print(f"{'Metric':<15} {'Zero-shot':<15} {'Fine-tuned':<15} {'Improvement':<15}")
print("-" * 60)
print(f"{'WER':<15} {wer_zero:.4f}{'':<11} {wer_finetuned:.4f}{'':<11} {(wer_zero - wer_finetuned):+.4f}")
print(f"{'CER':<15} {cer_zero:.4f}{'':<11} {cer_finetuned:.4f}{'':<11} {(cer_zero - cer_finetuned):+.4f}")

# CELL 7: Error Analysis
print("=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

def analyze_errors(ground_truths, predictions):
    substitutions = 0
    insertions = 0
    deletions = 0
    total_words = 0

    for gt, pred in zip(ground_truths, predictions):
        gt_words = gt.split()
        pred_words = pred.split()

        for i, word in enumerate(pred_words):
            if i < len(gt_words):
                if word != gt_words[i]:
                    substitutions += 1
            else:
                insertions += 1

        deletions = max(0, len(gt_words) - len(pred_words))
        total_words += len(gt_words)

    return {
        'substitutions': substitutions / total_words * 100 if total_words > 0 else 0,
        'insertions': insertions / total_words * 100 if total_words > 0 else 0,
        'deletions': deletions / total_words * 100 if total_words > 0 else 0
    }

error_stats = analyze_errors(all_labels_finetuned, all_preds_finetuned)
print(f"Error types:")
print(f"Substitutions: {error_stats['substitutions']:.2f}%")
print(f"Insertions: {error_stats['insertions']:.2f}%")
print(f"Deletions: {error_stats['deletions']:.2f}%")

short_texts = [(gt, pred) for gt, pred in zip(all_labels_finetuned, all_preds_finetuned) if len(gt.split()) < 5]
medium_texts = [(gt, pred) for gt, pred in zip(all_labels_finetuned, all_preds_finetuned) if 5 <= len(gt.split()) < 15]
long_texts = [(gt, pred) for gt, pred in zip(all_labels_finetuned, all_preds_finetuned) if len(gt.split()) >= 15]

wer_short = jiwer.wer([gt for gt, _ in short_texts], [pred for _, pred in short_texts]) if short_texts else 0
wer_medium = jiwer.wer([gt for gt, _ in medium_texts], [pred for _, pred in medium_texts]) if medium_texts else 0
wer_long = jiwer.wer([gt for gt, _ in long_texts], [pred for _, pred in long_texts]) if long_texts else 0

print(f"WER by length:")
print(f"Short (<5 words): {wer_short:.4f} ({len(short_texts)} samples)")
print(f"Medium (5-15 words): {wer_medium:.4f} ({len(medium_texts)} samples)")
print(f"Long (>=15 words): {wer_long:.4f} ({len(long_texts)} samples)")

print("10 sample errors:")
error_count = 0
for i, (gt, pred) in enumerate(zip(all_labels_finetuned, all_preds_finetuned)):
    if gt != pred and error_count < 10:
        print(f"\n{error_count+1}. True: {gt}")
        print(f"   Pred: {pred}")
        error_count += 1

results = {
    'zero_shot': {'wer': wer_zero, 'cer': cer_zero},
    'fine_tuned': {'wer': wer_finetuned, 'cer': cer_finetuned},
    'error_analysis': error_stats,
    'wer_by_length': {'short': wer_short, 'medium': wer_medium, 'long': wer_long},
    'training_histories': histories
}

with open('results/baseline_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Results saved: results/baseline_results.json")

# CELL 8: Create V1 Script (SpecAugment)
print("=" * 60)
print("STEP 5: PROPOSED METHODS")
print("=" * 60)
print("V1: SpecAugment")

v1_script = '''
import os, json, torch, random, numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm
import jiwer, librosa

USE_SUBSET = True
SUBSET_SIZE = 500
NUM_EPOCHS = 2 if USE_SUBSET else 5
BATCH_SIZE = 1

class SpecAugmentDataset(Dataset):
    def __init__(self, manifest_path, processor, use_specaugment=True):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.processor = processor
        self.use_specaugment = use_specaugment

        for item in self.data:
            item['audio_path'] = item['audio_path'].replace('\\\\', '/')
            if 'D:/Truong/' in item['audio_path']:
                item['audio_path'] = item['audio_path'].replace('D:/Truong/', '/content/Truong/')
            if 'D:\\\\Truong\\\\' in item['audio_path']:
                item['audio_path'] = item['audio_path'].replace('D:\\\\Truong\\\\', '/content/Truong/')

        if USE_SUBSET:
            self.data = self.data[:SUBSET_SIZE]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio, sr = librosa.load(item['audio_path'], sr=16000)

        if self.use_specaugment:
            audio = self._specaugment(audio)

        inputs = self.processor(audio=audio, sampling_rate=16000, return_tensors="pt")
        labels = self.processor.tokenizer(item['text'], return_tensors="pt").input_ids.squeeze(0)
        inputs['labels'] = labels
        return inputs

    def _specaugment(self, audio):
        if random.random() > 0.5:
            audio = torch.from_numpy(audio).float()
            mask_len = int(0.1 * len(audio))
            mask_start = random.randint(0, len(audio) - mask_len)
            audio[mask_start:mask_start+mask_len] = 0
            audio = audio.numpy()
        return audio

def compute_wer(preds, labels, processor):
    pred_texts = []
    label_texts = []
    for pred, label in zip(preds, labels):
        pred_text = processor.decode(pred, skip_special_tokens=True)
        label_text = processor.decode(label, skip_special_tokens=True)
        pred_texts.append(pred_text)
        label_texts.append(label_text)
    return jiwer.wer(label_texts, pred_texts)

def train_v1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="vi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)

    train_dataset = SpecAugmentDataset("data/manifests/vivos_train.json", processor, use_specaugment=True)
    test_dataset = SpecAugmentDataset("data/manifests/vivos_test.json", processor, use_specaugment=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    history = {'train_loss': [], 'test_wer': []}

    model.train()
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        total_loss = 0

        for batch in tqdm(train_loader):
            input_features = batch['input_features'].squeeze(0).to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                input_features = batch['input_features'].squeeze(0).to(device)
                labels = batch['labels'].to(device)
                generated_ids = model.generate(input_features=input_features)
                all_preds.extend(generated_ids.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        wer = compute_wer(all_preds, all_labels, processor)
        history['test_wer'].append(wer)
        print(f"Test WER: {wer:.4f}")
        model.train()

    model.save_pretrained("models/whisper_vivos_v1_specaugment")
    processor.save_pretrained("models/whisper_vivos_v1_specaugment")

    with open('results/v1_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    return history

if __name__ == "__main__":
    history = train_v1()
'''

with open('scripts/train_v1.py', 'w', encoding='utf-8') as f:
    f.write(v1_script)

print("V1 script created")

# CELL 9: Run V1 Training
print("Training V1...")
!python scripts/train_v1.py
print("V1 completed")

# CELL 10: Create V2 Script (Cosine LR)
print("V2: Cosine LR")

v2_script = '''
import os, json, torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration, get_cosine_schedule_with_warmup
from tqdm import tqdm
import jiwer, librosa

USE_SUBSET = True
SUBSET_SIZE = 500
NUM_EPOCHS = 2 if USE_SUBSET else 5
BATCH_SIZE = 1

class VivosDataset(Dataset):
    def __init__(self, manifest_path, processor):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.processor = processor

        for item in self.data:
            item['audio_path'] = item['audio_path'].replace('\\\\', '/')
            if 'D:/Truong/' in item['audio_path']:
                item['audio_path'] = item['audio_path'].replace('D:/Truong/', '/content/Truong/')
            if 'D:\\\\Truong\\\\' in item['audio_path']:
                item['audio_path'] = item['audio_path'].replace('D:\\\\Truong\\\\', '/content/Truong/')

        if USE_SUBSET:
            self.data = self.data[:SUBSET_SIZE]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio, sr = librosa.load(item['audio_path'], sr=16000)
        inputs = self.processor(audio=audio, sampling_rate=16000, return_tensors="pt")
        labels = self.processor.tokenizer(item['text'], return_tensors="pt").input_ids.squeeze(0)
        inputs['labels'] = labels
        return inputs

def compute_wer(preds, labels, processor):
    pred_texts = []
    label_texts = []
    for pred, label in zip(preds, labels):
        pred_text = processor.decode(pred, skip_special_tokens=True)
        label_text = processor.decode(label, skip_special_tokens=True)
        pred_texts.append(pred_text)
        label_texts.append(label_text)
    return jiwer.wer(label_texts, pred_texts)

def train_v2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="vi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)

    train_dataset = VivosDataset("data/manifests/vivos_train.json", processor)
    test_dataset = VivosDataset("data/manifests/vivos_test.json", processor)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = NUM_EPOCHS
    num_training_steps = len(train_loader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps
    )

    history = {'train_loss': [], 'test_wer': [], 'learning_rates': []}

    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0

        for batch in tqdm(train_loader):
            input_features = batch['input_features'].squeeze(0).to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        history['train_loss'].append(avg_loss)
        history['learning_rates'].append(current_lr)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                input_features = batch['input_features'].squeeze(0).to(device)
                labels = batch['labels'].to(device)
                generated_ids = model.generate(input_features=input_features)
                all_preds.extend(generated_ids.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        wer = compute_wer(all_preds, all_labels, processor)
        history['test_wer'].append(wer)
        print(f"Test WER: {wer:.4f}, LR: {current_lr:.6f}")
        model.train()

    model.save_pretrained("models/whisper_vivos_v2_cosine")
    processor.save_pretrained("models/whisper_vivos_v2_cosine")

    with open('results/v2_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    return history

if __name__ == "__main__":
    history = train_v2()
'''

with open('scripts/train_v2.py', 'w', encoding='utf-8') as f:
    f.write(v2_script)

print("V2 script created")

# CELL 11: Run V2 Training
print("Training V2...")
!python scripts/train_v2.py
print("V2 completed")

# CELL 12: Create V3 Script (Larger Model)
print("V3: Larger Model (Whisper Base)")

v3_script = '''
import os, json, torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm
import jiwer, librosa

USE_SUBSET = True
SUBSET_SIZE = 500
NUM_EPOCHS = 2 if USE_SUBSET else 5
BATCH_SIZE = 1

class VivosDataset(Dataset):
    def __init__(self, manifest_path, processor):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.processor = processor

        for item in self.data:
            item['audio_path'] = item['audio_path'].replace('\\\\', '/')
            if 'D:/Truong/' in item['audio_path']:
                item['audio_path'] = item['audio_path'].replace('D:/Truong/', '/content/Truong/')
            if 'D:\\\\Truong\\\\' in item['audio_path']:
                item['audio_path'] = item['audio_path'].replace('D:\\\\Truong\\\\', '/content/Truong/')

        if USE_SUBSET:
            self.data = self.data[:SUBSET_SIZE]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio, sr = librosa.load(item['audio_path'], sr=16000)
        inputs = self.processor(audio=audio, sampling_rate=16000, return_tensors="pt")
        labels = self.processor.tokenizer(item['text'], return_tensors="pt").input_ids.squeeze(0)
        inputs['labels'] = labels
        return inputs

def compute_wer(preds, labels, processor):
    pred_texts = []
    label_texts = []
    for pred, label in zip(preds, labels):
        pred_text = processor.decode(pred, skip_special_tokens=True)
        label_text = processor.decode(label, skip_special_tokens=True)
        pred_texts.append(pred_text)
        label_texts.append(label_text)
    return jiwer.wer(label_texts, pred_texts)

def train_v3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "openai/whisper-base"
    processor = WhisperProcessor.from_pretrained(model_name, language="vi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

    train_dataset = VivosDataset("data/manifests/vivos_train.json", processor)
    test_dataset = VivosDataset("data/manifests/vivos_test.json", processor)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    history = {'train_loss': [], 'test_wer': []}

    model.train()
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        total_loss = 0

        for batch in tqdm(train_loader):
            input_features = batch['input_features'].squeeze(0).to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                input_features = batch['input_features'].squeeze(0).to(device)
                labels = batch['labels'].to(device)
                generated_ids = model.generate(input_features=input_features)
                all_preds.extend(generated_ids.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        wer = compute_wer(all_preds, all_labels, processor)
        history['test_wer'].append(wer)
        print(f"Test WER: {wer:.4f}")
        model.train()

    model.save_pretrained("models/whisper_vivos_v3_base")
    processor.save_pretrained("models/whisper_vivos_v3_base")

    with open('results/v3_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    return history

if __name__ == "__main__":
    history = train_v3()
'''

with open('scripts/train_v3.py', 'w', encoding='utf-8') as f:
    f.write(v3_script)

print("V3 script created")

# CELL 13: Run V3 Training
print("Training V3...")
!python scripts/train_v3.py
print("V3 completed")

# CELL 14: Create Comparison Script
print("=" * 60)
print("VERSION COMPARISON")
print("=" * 60)

comparison_script = '''
import json, os

baseline = json.load(open('results/training_histories.json', 'r'))
v1 = json.load(open('results/v1_history.json', 'r')) if os.path.exists('results/v1_history.json') else None
v2 = json.load(open('results/v2_history.json', 'r')) if os.path.exists('results/v2_history.json') else None
v3 = json.load(open('results/v3_history.json', 'r')) if os.path.exists('results/v3_history.json') else None

print("=" * 80)
print("VERSION COMPARISON TABLE")
print("=" * 80)
print("| Version | Hypothesis | Change | Final WER | Improvement | Conclusion |")
print("|---------|-----------|--------|-----------|-------------|-------------|")

baseline_wer = baseline[0]['test_wer'][-1]
print(f"| Baseline | - | - | {baseline_wer:.4f} | - | - |")

if v1:
    v1_wer = v1['test_wer'][-1]
    improvement = baseline_wer - v1_wer
    print(f"| V1 | SpecAugment | Time masking | {v1_wer:.4f} | {improvement:+.4f} | {'Keep' if improvement > 0 else 'Discard'} |")

if v2:
    v2_wer = v2['test_wer'][-1]
    improvement = baseline_wer - v2_wer
    print(f"| V2 | Cosine LR | Cosine schedule | {v2_wer:.4f} | {improvement:+.4f} | {'Keep' if improvement > 0 else 'Discard'} |")

if v3:
    v3_wer = v3['test_wer'][-1]
    improvement = baseline_wer - v3_wer
    print(f"| V3 | Larger model | Whisper Base | {v3_wer:.4f} | {improvement:+.4f} | {'Keep' if improvement > 0 else 'Discard'} |")

print("=" * 80)

# Determine best version
best_version = "Baseline"
best_wer = baseline_wer
if v1 and v1_wer < best_wer:
    best_version = "V1 (SpecAugment)"
    best_wer = v1_wer
if v2 and v2_wer < best_wer:
    best_version = "V2 (Cosine LR)"
    best_wer = v2_wer
if v3 and v3_wer < best_wer:
    best_version = "V3 (Whisper Base)"
    best_wer = v3_wer

# Calculate total improvement
total_improvement = baseline_wer - best_wer
total_improvement_pct = (total_improvement / baseline_wer) * 100

print("CONCLUSION:")
print(f"1. Best version: {best_version} (WER: {best_wer:.4f})")
print(f"2. Total improvement: {total_improvement:.4f} ({total_improvement_pct:.1f}%) over baseline")
print(f"3. Feasibility: Yes - Minimal computational overhead, easy to implement")
print(f"4. Limitations: Small dataset (500 samples), limited epochs (2), single seed (42), batch_size=1")
print(f"5. Future work: Train on full dataset, increase epochs, test multiple seeds")
'''

with open('scripts/compare_versions.py', 'w', encoding='utf-8') as f:
    f.write(comparison_script)

print("Comparison script created")

# CELL 15: Run Comparison
print("Running comparison...")
!python scripts/compare_versions.py
print("Comparison completed")
