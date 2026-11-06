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
            item['audio_path'] = item['audio_path'].replace('\\', '/')
            if 'D:/Truong/' in item['audio_path']:
                item['audio_path'] = item['audio_path'].replace('D:/Truong/', '/content/Truong/')
            if 'D:\\Truong\\' in item['audio_path']:
                item['audio_path'] = item['audio_path'].replace('D:\\Truong\\', '/content/Truong/')

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
