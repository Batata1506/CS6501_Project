import pickle, torch, torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from pathlib import Path
import re
import matplotlib.cm as cm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------
# Load processed datasets
# ----------------------------------------------------------------------
DATA_DIR = Path("D:/ECE_Masters/CS6501/Project/processed_data")
pkl_files = sorted(DATA_DIR.glob("truncated_data_*s.pkl"))
if not pkl_files:
    raise FileNotFoundError("No truncated_data_*.pkl files found")

print(f"Found {len(pkl_files)} processed datasets:")
for f in pkl_files:
    print("  -", f.name)

# ----------------------------------------------------------------------
# LSTM Model
# ----------------------------------------------------------------------
class EmotionLSTM(nn.Module):
    def __init__(self, input_size, hidden=128, layers=4, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 2)  # valence + arousal
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ----------------------------------------------------------------------
# Train Loop
# ----------------------------------------------------------------------
def train_lstm(X_train, y_train, X_val, y_val, input_size, epochs=40, lr=5e-4):
    model = EmotionLSTM(input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.float32))
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = np.mean([
                loss_fn(model(xb.to(device)), yb.to(device)).item()
                for xb, yb in val_dl
            ])
        scheduler.step(val_loss)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - val_loss: {val_loss:.4f}")
    return model

# ----------------------------------------------------------------------
# Run for all clip lengths
# ----------------------------------------------------------------------
results = []
predictions = {}

for file in pkl_files:
    match = re.search(r"(\d+)s", file.name)
    clip_len = int(match.group(1)) if match else None
    with open(file, "rb") as f:
        data = pickle.load(f)
    print(f"\n{clip_len}s clips → {len(data)} samples")

    X, y = [], []
    for d in data:
        X.append(d["features"])
        y.append([d["valence"], d["arousal"]])
    X, y = np.array(X), np.array(y)

    # Normalize features
    scaler = StandardScaler().fit(X.reshape(-1, X.shape[-1]))
    X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # Normalize labels
    y_min, y_max = np.min(y, axis=0), np.max(y, axis=0)
    y_norm = (y - y_min) / (y_max - y_min)

    # Split
    n_train = int(0.8 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y_norm[:n_train], y_norm[n_train:]
    input_size = X.shape[2]

    model = train_lstm(X_train, y_train, X_val, y_val, input_size)

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().numpy()

    # Denormalize
    y_pred = y_pred * (y_max - y_min) + y_min
    y_true = y_val * (y_max - y_min) + y_min

    rmse_v = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    rmse_a = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
    r_v, _ = pearsonr(y_true[:, 0], y_pred[:, 0])
    r_a, _ = pearsonr(y_true[:, 1], y_pred[:, 1])

    results.append({
        "clip_len": clip_len,
        "rmse_val": rmse_v,
        "rmse_aro": rmse_a,
        "r_val": r_v,
        "r_aro": r_a
    })

    predictions[clip_len] = {"valence": y_pred[:, 0], "arousal": y_pred[:, 1]}

    print(f"{clip_len}s → RMSE V:{rmse_v:.3f}, A:{rmse_a:.3f} | r V:{r_v:.3f}, A:{r_a:.3f}")

# ----------------------------------------------------------------------
# Plot RMSE and Correlation
# ----------------------------------------------------------------------
results = sorted(results, key=lambda x: x["clip_len"])
durations = [r["clip_len"] for r in results]
rmse_v = [r["rmse_val"] for r in results]
rmse_a = [r["rmse_aro"] for r in results]
r_v = [r["r_val"] for r in results]
r_a = [r["r_aro"] for r in results]

plt.figure(figsize=(8, 5))
plt.plot(durations, rmse_v, "o-", label="Valence RMSE")
plt.plot(durations, rmse_a, "s-", label="Arousal RMSE")
plt.xlabel("Audio Clip Length (s)")
plt.ylabel("RMSE (lower = better)")
plt.title("LSTM Performance vs Clip Length (DEAM)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(durations, r_v, "o-", label="Valence r")
plt.plot(durations, r_a, "s-", label="Arousal r")
plt.xlabel("Audio Clip Length (s)")
plt.ylabel("Correlation (higher = better)")
plt.title("LSTM Correlation vs Clip Length (DEAM)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# Emotion Space Visualization
# ----------------------------------------------------------------------
def normalize(val):
    return (val - 5) / 4  # Map DEAM 1–9 to -1–1

def plot_emotion_space(predictions_dict):
    plt.figure(figsize=(6, 6))
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    plt.title("Predicted Emotion Space (LSTM – DEAM)")

    # Emotion zones
    plt.text(0.6, 0.7, 'Happy', fontsize=12, ha='center')
    plt.text(-0.6, 0.7, 'Relaxed', fontsize=12, ha='center')
    plt.text(-0.6, -0.7, 'Sad', fontsize=12, ha='center')
    plt.text(0.6, -0.7, 'Angry', fontsize=12, ha='center')

    colors = cm.viridis(np.linspace(0, 1, len(predictions_dict)))

    for (clip_len, vals), color in zip(predictions_dict.items(), colors):
        val_norm = normalize(vals["valence"])
        aro_norm = normalize(vals["arousal"])
        plt.scatter(val_norm, aro_norm, alpha=0.5, s=10, color=color, label=f"{clip_len}s")

    plt.legend(title="Clip Length")
    plt.tight_layout()
    plt.show()

plot_emotion_space(predictions)

