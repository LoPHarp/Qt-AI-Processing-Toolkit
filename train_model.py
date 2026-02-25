import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader

class FaceDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file).dropna()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_str = row['Image']
        img_array = np.array(img_str.split(), dtype=np.float32)
        img = img_array.reshape(1, 96, 96) / 255.0
        pts = row.drop('Image').values.astype(np.float32) / 96.0
        return torch.tensor(img), torch.tensor(pts)

class FaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64 * 22 * 22, 128)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(128, 30)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = self.flatten(x)

        x = self.relu3(self.fc1(x))
        x = self.fc2(x)

        return x

def train_model(dataset_path, target_acc, save_path):
    try:
        dataset = FaceDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        model = FaceNet()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        current_acc = 0.0
        epoch = 1

        while current_acc < target_acc:
            total_loss = 0.0
            for imgs, pts in dataloader:
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, pts)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            current_acc = max(0.0, min(1.0, 1.0 - (avg_loss * 15)))

            print(f"CURRENT_LEARN_PROGRESS: {current_acc:.4f} CURRENT_EPOCH: {epoch}", flush=True)
            sys.stdout.flush()
            epoch += 1

        dummy_input = torch.randn(1, 1, 96, 96)
        torch.onnx.export(model, dummy_input, save_path, input_names=['input'], output_names=['output'])
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 4: sys.exit(1)
    train_model(sys.argv[1], float(sys.argv[2]), sys.argv[3])