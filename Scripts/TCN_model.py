import pickle, torch, torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from pathlib import Path
import re
import time
import psutil
import os

device = torch.device("cpu")

# ----------------------------------------------------------------------
# Utility: count model parameters
# ----------------------------------------------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ----------------------------------------------------------------------
# Utility: CPU stats
# ----------------------------------------------------------------------
def get_cpu_stats():
    process = psutil.Process(os.getpid())
    cpu_percent = psutil.cpu_percent(interval=None)
    mem_mb = process.memory_info().rss / (1024 ** 2)
    times = process.cpu_times()
    return {
        "cpu_percent": cpu_percent,
        "memory_mb": mem_mb,
        "user_time": times.user,
        "system_time": times.system,
    }

# ----------------------------------------------------------------------
# Load processed datasets (most-active segments)
# ----------------------------------------------------------------------
DATA_DIR = Path("D:/ECE_Masters/CS6501/Project/processed_data")
pkl_files = sorted(DATA_DIR.glob("active_segment_data_*s.pkl"))
if not pkl_files:
    raise FileNotFoundError("No active_segment_data_*.pkl files found in processed_data")

print(f"Found {len(pkl_files)} processed datasets:")
for f in pkl_files:
    print("  -", f.name)

# ----------------------------------------------------------------------
# TCN building blocks
# ----------------------------------------------------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride,
                 dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs,
                               kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs,
                               kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) \
            if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.3):
        super().__init__()

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                              dilation=dilation_size,
                              padding=padding,
                              dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

        final_channels = num_channels[-1]
        self.fc1 = nn.Linear(final_channels, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 2)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: [batch, T, F] -> [batch, F, T]
        x = x.permute(0, 2, 1)
        y = self.network(x)           # [batch, C, T]
        y_last = y[:, :, -1]          # [batch, C]
        z = self.fc1(y_last)
        z = self.act(z)
        z = self.dropout(z)
        return self.fc2(z)            # [batch, 2]


# ----------------------------------------------------------------------
# Train loop + analytics
# ----------------------------------------------------------------------
def train_tcn(X_train, y_train, X_val, y_val,
              input_size, epochs=40, lr=5e-4):
    model = TCN(input_size=input_size,
                num_channels=[64, 64, 128, 128],
                kernel_size=3,
                dropout=0.3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.float32))

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)

    param_count = count_params(model)
    start_time = time.perf_counter()
    last_stats = None

    EARLY_STOP = False
    best_loss = float("inf")
    patience = 8
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = np.mean([
                loss_fn(model(xb.to(device)), yb.to(device)).item()
                for xb, yb in val_dl
            ])

        scheduler.step(val_loss)
        last_stats = get_cpu_stats()

        if EARLY_STOP:
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - val_loss: {val_loss:.4f}")
            print(
                f"      CPU: {last_stats['cpu_percent']:.1f}% | "
                f"RAM: {last_stats['memory_mb']:.1f} MB | "
                f"user: {last_stats['user_time']:.1f}s | "
                f"sys: {last_stats['system_time']:.1f}s"
            )

    total_time = time.perf_counter() - start_time
    time_per_epoch = total_time / (epoch + 1)

    stats = {
        "params": param_count,
        "total_time_sec": total_time,
        "time_per_epoch_sec": time_per_epoch,
        "cpu_percent": last_stats["cpu_percent"],
        "memory_mb": last_stats["memory_mb"],
        "user_time": last_stats["user_time"],
        "system_time": last_stats["system_time"],
    }

    return model, stats


# ----------------------------------------------------------------------
# Run for all clip lengths
# ----------------------------------------------------------------------
results = []

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

    # 80/20 split with fixed seed
    rng = np.random.default_rng(seed=42)
    idx = np.arange(len(X))
    rng.shuffle(idx)

    n_train = int(0.8 * len(X))
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_norm[train_idx], y_norm[val_idx]

    input_size = X.shape[2]

    model, stats = train_tcn(X_train, y_train, X_val, y_val, input_size)

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().numpy()

    # Denormalize
    y_pred = y_pred * (y_max - y_min) + y_min
    y_true = y_norm[val_idx] * (y_max - y_min) + y_min

    rmse_v = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    rmse_a = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
    r_v, _ = pearsonr(y_true[:, 0], y_pred[:, 0])
    r_a, _ = pearsonr(y_true[:, 1], y_pred[:, 1])

    results.append({
        "model": "TCN",
        "clip_len": clip_len,
        "rmse_val": rmse_v,
        "rmse_aro": rmse_a,
        "r_val": r_v,
        "r_aro": r_a,
        "params": stats["params"],
        "time_per_epoch": stats["time_per_epoch_sec"],
        "total_time": stats["total_time_sec"],
        "cpu_percent": stats["cpu_percent"],
        "memory_mb": stats["memory_mb"],
        "user_time": stats["user_time"],
        "system_time": stats["system_time"],
    })

    print(f"{clip_len}s → RMSE V:{rmse_v:.3f}, A:{rmse_a:.3f} | r V:{r_v:.3f}, A:{r_a:.3f}")
    print(
        f"   Params: {stats['params']:,} | "
        f"Time/epoch: {stats['time_per_epoch_sec']:.2f}s | "
        f"RAM: {stats['memory_mb']:.1f} MB | "
        f"CPU: {stats['cpu_percent']:.1f}%"
    )

# ----------------------------------------------------------------------
# Summary table (for report)
# ----------------------------------------------------------------------
print("\n================ SUMMARY ================")
for r in sorted(results, key=lambda x: x["clip_len"]):
    print(
        f"{r['model']} | {r['clip_len']}s | "
        f"RMSE(V/A): {r['rmse_val']:.3f}/{r['rmse_aro']:.3f} | "
        f"r(V/A): {r['r_val']:.3f}/{r['r_aro']:.3f} | "
        f"Params: {r['params']/1e6:.2f}M | "
        f"Time/epoch: {r['time_per_epoch']:.2f}s | "
        f"RAM: {r['memory_mb']:.1f} MB | "
        f"CPU: {r['cpu_percent']:.1f}%"
    )

# ----------------------------------------------------------------------
# Plot RMSE and Correlation vs clip length
# ----------------------------------------------------------------------
results_sorted = sorted(results, key=lambda x: x["clip_len"])
durations = [r["clip_len"] for r in results_sorted]
rmse_v = [r["rmse_val"] for r in results_sorted]
rmse_a = [r["rmse_aro"] for r in results_sorted]
r_v = [r["r_val"] for r in results_sorted]
r_a = [r["r_aro"] for r in results_sorted]

plt.figure(figsize=(8, 5))
plt.plot(durations, rmse_v, "o-", label="Valence RMSE")
plt.plot(durations, rmse_a, "s-", label="Arousal RMSE")
plt.xlabel("Audio Clip Length (s)")
plt.ylabel("RMSE (lower = better)")
plt.title("TCN Performance vs Clip Length (DEAM – most active segments)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(durations, r_v, "o-", label="Valence r")
plt.plot(durations, r_a, "s-", label="Arousal r")
plt.xlabel("Audio Clip Length (s)")
plt.ylabel("Correlation (higher = better)")
plt.title("TCN Correlation vs Clip Length (DEAM – most active segments)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
