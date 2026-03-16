import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter


class TextDataset(Dataset):
    def __init__(self, csv_file, max_seq_length=50, vocab_size=10000):
        self.df = pd.read_csv(csv_file)
        self.max_seq_length = max_seq_length

        all_words = []
        for text in self.df['clean_text']:
            if isinstance(text, str):
                all_words.extend(text.split())

        word_counts = Counter(all_words)
        common_words = word_counts.most_common(vocab_size)

        self.word_to_idx = \
            {
                '<PAD>': 0,
                '<UNK>': 1
            }

        for idx, (word, _) in enumerate(common_words, start=2):
            self.word_to_idx[word] = idx

        with open('vocab_rnn.json', 'w', encoding='utf-8') as f:
            json.dump(self.word_to_idx, f, ensure_ascii=False, indent=4)

        self.vocab_size = len(self.word_to_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df['clean_text'].iloc[idx])
        label = int(self.df['emotion'].iloc[idx])

        words = text.split()
        seq = [self.word_to_idx.get(w, 1) for w in words]

        if len(seq) > self.max_seq_length:
            seq = seq[:self.max_seq_length]
        else:
            seq = seq + [0] * (self.max_seq_length - len(seq))

        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class EmotionRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, output_dim=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.rnn(embedded)
        out = self.fc(hidden[-1])
        return out


def train_model(dataset_path, target_acc, save_path):
    try:
        dataset = TextDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EmotionRNN(dataset.vocab_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epoch = 1
        current_acc = 0.0

        while current_acc < target_acc:
            model.train()
            correct = 0
            total = 0

            for texts, labels in dataloader:
                texts, labels = texts.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(texts)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            current_acc = correct / total
            print(f"CURRENT_LEARN_PROGRESS: {current_acc:.4f} EPOCH: {epoch}", flush=True)
            sys.stdout.flush()
            epoch += 1

        model.eval()
        model.to('cpu')

        dummy_input = torch.zeros(1, dataset.max_seq_length, dtype=torch.long)

        try:
            torch.onnx.export(
                model,
                dummy_input,
                save_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamo=False
            )
        except TypeError:
            torch.onnx.export(
                model,
                dummy_input,
                save_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )

        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'emotion_text_dataset.csv'
    target_acc = float(sys.argv[2]) if len(sys.argv) > 2 else 0.70
    model_out_path = sys.argv[3] if len(sys.argv) > 3 else 'model_rnn.onnx'

    train_model(csv_path, target_acc, model_out_path)