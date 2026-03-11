import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


class EmotionDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file).dropna()
        self.features = self.data.iloc[:, 1:].values.astype(np.float32)
        self.labels = self.data.iloc[:, 0].values.astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])
        y = torch.tensor(self.labels[idx])
        return x, y


class EmotionNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Вхідний шар + Нормалізація + Dropout (захист від шуму)
        self.fc1 = nn.Linear(30, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        # Прихований шар
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)

        # Вихідний шар
        self.fc3 = nn.Linear(64, 7)

    def forward(self, x):
        # Якщо даних мало (наприклад, 1 картинка під час інференсу), BatchNorm може видати помилку,
        # тому застосовуємо його обережно
        if x.size(0) > 1:
            x = self.drop1(self.relu1(self.bn1(self.fc1(x))))
            x = self.drop2(self.relu2(self.bn2(self.fc2(x))))
        else:
            x = self.drop1(self.relu1(self.fc1(x)))
            x = self.drop2(self.relu2(self.fc2(x)))

        x = self.fc3(x)
        return x


def train_model(dataset_path, target_acc, save_path):
    try:
        print(f"\n[!!!] SCRIPT RUNNING FROM: {os.path.abspath(__file__)} [!!!]\n", flush=True)
        sys.stdout.flush()

        dataset = EmotionDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        model = EmotionNet()
        criterion = nn.CrossEntropyLoss()
        # Зменшили learning rate, щоб мережа вчилася обережніше
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        smoothed_acc = 0.0
        epoch = 1

        while smoothed_acc < target_acc:
            model.train()  # Обов'язково вмикаємо режим тренування для Dropout/BatchNorm
            correct_train = 0
            total_train = 0

            for inputs, labels in dataloader:
                # Пропускаємо неповні батчі, щоб BatchNorm не ламався
                if inputs.size(0) <= 1:
                    continue

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            if total_train == 0:
                continue

            current_acc = correct_train / total_train

            # Згладжуємо точність для плавного прогрес-бару
            if epoch == 1:
                smoothed_acc = current_acc
            else:
                smoothed_acc = 0.9 * smoothed_acc + 0.1 * current_acc

            print(f"CURRENT_LEARN_PROGRESS: {smoothed_acc:.4f} CURRENT_EPOCH: {epoch}", flush=True)
            sys.stdout.flush()
            epoch += 1

        print("\n[!!!] EXPORTING VIA LEGACY ONNX [!!!]", flush=True)
        sys.stdout.flush()

        model.eval()  # Обов'язково вимикаємо Dropout перед експортом!
        model.to('cpu')

        dummy_input = torch.randn(1, 30, dtype=torch.float32)

        try:
            torch.onnx.export(
                model,
                dummy_input,
                save_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=False,
                input_names=['input_coords'],
                output_names=['emotion_probs'],
                dynamo=False
            )
        except TypeError:
            torch.onnx.export(
                model,
                dummy_input,
                save_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=False,
                input_names=['input_coords'],
                output_names=['emotion_probs']
            )

        print("[!!!] EXPORT TRUE COMPLETED [!!!]\n", flush=True)
        sys.stdout.flush()
        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(1)
    train_model(sys.argv[1], float(sys.argv[2]), sys.argv[3])