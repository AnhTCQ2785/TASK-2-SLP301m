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
            item['audio_path'] = item['audio_path'].replace('\\', '/')
            if 'D:/Truong/' in item['audio_path']:
                item['audio_path'] = item['audio_path'].replace('D:/Truong/', '/content/Truong/')
            if 'D:\\Truong\\' in item['audio_path']:
                item['audio_path'] = item['audio_path'].replace('D:\\Truong\\', '/content/Truong/')

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
