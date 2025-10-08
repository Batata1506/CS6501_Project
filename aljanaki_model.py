import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr

# ============================================================
# 1. Load data
# ============================================================
audio_features = pd.read_csv("audio_features.csv")
static_annotations = pd.read_csv(
    "DEAM/DEAM_Annotations/annotations/annotations_averaged_per_song/song_level/static_annotations_averaged_songs_1_2000.csv",
    index_col="song_id"
)
# Remove spaces from column names
static_annotations.columns = [c.strip() for c in static_annotations.columns]

# Extract song_id from file names
audio_features["song_id"] = audio_features["Id"].str.extract(r"(\d+)").astype(int)

# Merge features with labels
merged = audio_features.merge(
    static_annotations[["valence_mean", "arousal_mean"]],
    left_on="song_id",
    right_index=True
).dropna()

# Separate features and labels
X = merged.drop(columns=["Id", "song_id", "valence_mean", "arousal_mean"])
y = merged[["valence_mean", "arousal_mean"]]

# ============================================================
# 2. Train/test split BEFORE scaling
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.2, random_state=42
)

# ============================================================
# 3. Scale data using training set only
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on training set only
X_test_scaled = scaler.transform(X_test)        # transform test set using training parameters

# ============================================================
# 4. PyTorch Dataset and DataLoader
# ============================================================
class DEAMStaticDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create datasets and dataloaders
train_dataset = DEAMStaticDataset(X_train_scaled, y_train)
test_dataset = DEAMStaticDataset(X_test_scaled, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# ============================================================
# 5. MLP model
# ============================================================
class EmotionMLP(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 2)  # valence and arousal
        )
    def forward(self, x):
        return self.net(x)

# Initialize model
model = EmotionMLP(input_size=X_train.shape[1])

# ============================================================
# 6. Training
# ============================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
criterion = nn.MSELoss()              # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    model.train()                     # set model to training mode
    total_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()          # reset gradients
        pred = model(Xb)               # forward pass
        loss = criterion(pred, yb)     # compute MSE loss
        loss.backward()                # backpropagation
        optimizer.step()               # update weights
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/50 - Loss: {total_loss/len(train_loader):.4f}")

# ============================================================
# 7. Evaluation
# ============================================================
model.eval()                          # set model to evaluation mode
with torch.no_grad():                 # disable gradient computation
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    preds = model(X_test_t).cpu().numpy()

# Compute Pearson correlation for valence and arousal
val_corr = pearsonr(preds[:, 0], y_test[:, 0])[0]
aro_corr = pearsonr(preds[:, 1], y_test[:, 1])[0]
print(f"Correlation Valence: {val_corr:.3f}, Arousal: {aro_corr:.3f}")