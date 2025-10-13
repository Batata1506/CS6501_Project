import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

# optional: for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class DEAMDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx]["features"], dtype=torch.float32)  # (T, F)
        y = torch.tensor(
            [self.data[idx]["valence"], self.data[idx]["arousal"]],
            dtype=torch.float32,
        )
        return x, y

class EmotionLSTM(nn.Module):
    def __init__(self, input_dim=260, hidden_dim=128, num_layers=2, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)   # h_n = [num_layers, batch, hidden_dim]
        out = self.fc(h_n[-1])       # take final hidden layer
        return out
