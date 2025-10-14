# train_model.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
import random

# ================================
# 1️⃣ Load preprocessed data
# ================================
with open("D:/ECE_Masters/CS6501/truncated_data_10s.pkl", "rb") as f:
    truncated_data = pickle.load(f)

print(f"✅ Loaded {len(truncated_data)} samples from truncated_data.pkl")
print("Example input shape:", truncated_data[0]["features"].shape)


# ================================
# 2️⃣ Define PyTorch Dataset
# ================================
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


# ================================
# 3️⃣ Define the LSTM Model
# ================================
class EmotionLSTM(nn.Module):
    def __init__(self, input_dim=260, hidden_dim=128, num_layers=2, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)   # (num_layers, batch, hidden_dim)
        out = self.fc(h_n[-1])       # last hidden layer output
        return out


# ================================
# 4️⃣ Train/Test Split
# ================================
dataset = DEAMDataset(truncated_data)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)


# ================================
# 5️⃣ Model, Loss, Optimizer
# ================================
model = EmotionLSTM(input_dim=260, hidden_dim=128)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# ================================
# 6️⃣ Training Loop
# ================================
for epoch in range(15):  # adjust epochs as needed
    model.train()
    total_loss = 0

    for X, y in train_loader:
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Train Loss: {total_loss / len(train_loader):.4f}")


# ================================
# 7️⃣ Evaluation
# ================================
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for X, y in test_loader:
        preds = model(X)
        y_true.append(y.numpy())
        y_pred.append(preds.numpy())

y_true = np.vstack(y_true)
y_pred = np.vstack(y_pred)

rmse_valence = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
rmse_arousal = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))

print(f"\n✅ Results:")
print(f"RMSE Valence: {rmse_valence:.3f}")
print(f"RMSE Arousal: {rmse_arousal:.3f}")

# optional: save trained model
torch.save(model.state_dict(), "emotion_lstm_model.pt")
print("💾 Model saved as emotion_lstm_model.pt")
