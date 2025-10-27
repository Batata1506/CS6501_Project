import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from pathlib import Path
import re
import matplotlib.cm as cm

# ----------------------------------------------------------------------
# Load processed datasets
# ----------------------------------------------------------------------
DATA_DIR = Path("D:/ECE_Masters/CS6501/Project/processed_data")
pkl_files = sorted(DATA_DIR.glob("truncated_data_*s.pkl"))
if not pkl_files:
    raise FileNotFoundError(f"No truncated_data_*.pkl files found in {DATA_DIR}")

print(f"Found {len(pkl_files)} processed datasets:")
for f in pkl_files:
    print("  -", f.name)

results = []
predictions = {}

# ----------------------------------------------------------------------
# Evaluate SVR model for each clip length
# ----------------------------------------------------------------------
for file in pkl_files:
    match = re.search(r"(\d+)s", file.name)
    clip_len = int(match.group(1)) if match else None

    with open(file, "rb") as f:
        data = pickle.load(f)

    print(f"\nProcessing {file.name} ({len(data)} samples)")

    X, y_val, y_aro = [], [], []
    for d in data:
        X.append(d["features"].flatten())
        y_val.append(d["valence"])
        y_aro.append(d["arousal"])

    X = np.array(X)
    y_val = np.array(y_val)
    y_aro = np.array(y_aro)

    X_train, X_test, yv_train, yv_test = train_test_split(
        X, y_val, test_size=0.2, random_state=42
    )
    _, _, ya_train, ya_test = train_test_split(
        X, y_aro, test_size=0.2, random_state=42
    )

    svr_val = SVR(kernel="rbf", C=10, epsilon=0.1)
    svr_aro = SVR(kernel="rbf", C=10, epsilon=0.1)

    svr_val.fit(X_train, yv_train)
    svr_aro.fit(X_train, ya_train)

    yv_pred = svr_val.predict(X_test)
    ya_pred = svr_aro.predict(X_test)

    rmse_val = np.sqrt(mean_squared_error(yv_test, yv_pred))
    rmse_aro = np.sqrt(mean_squared_error(ya_test, ya_pred))
    r_val, _ = pearsonr(yv_test, yv_pred)
    r_aro, _ = pearsonr(ya_test, ya_pred)

    results.append({
        "clip_len": clip_len,
        "rmse_val": rmse_val,
        "rmse_aro": rmse_aro,
        "r_val": r_val,
        "r_aro": r_aro
    })

    predictions[clip_len] = {
        "valence": yv_pred,
        "arousal": ya_pred
    }

    print(f"{clip_len}s → RMSE V:{rmse_val:.3f}, A:{rmse_aro:.3f} | r V:{r_val:.3f}, A:{r_aro:.3f}")

# ----------------------------------------------------------------------
# Metrics vs Clip Length
# ----------------------------------------------------------------------
results = sorted(results, key=lambda x: x["clip_len"])
durations = [r["clip_len"] for r in results]
rmse_val = [r["rmse_val"] for r in results]
rmse_aro = [r["rmse_aro"] for r in results]
r_val = [r["r_val"] for r in results]
r_aro = [r["r_aro"] for r in results]

# Plot RMSE
plt.figure(figsize=(8, 5))
plt.plot(durations, rmse_val, "o-", label="Valence RMSE")
plt.plot(durations, rmse_aro, "s-", label="Arousal RMSE")
plt.xlabel("Audio Clip Length (s)")
plt.ylabel("RMSE (lower = better)")
plt.title("SVR Performance vs Clip Length (DEAM)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot correlation
plt.figure(figsize=(8, 5))
plt.plot(durations, r_val, "o-", label="Valence r")
plt.plot(durations, r_aro, "s-", label="Arousal r")
plt.xlabel("Audio Clip Length (s)")
plt.ylabel("Correlation (higher = better)")
plt.title("SVR Correlation vs Clip Length (DEAM)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# Emotion Space Visualization
# ----------------------------------------------------------------------
def normalize(val):
    return (val - 5) / 4  # map 1–9 → -1–1

def plot_emotion_space(predictions_dict):
    plt.figure(figsize=(6, 6))
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    plt.title("Predicted Emotion Space (SVR – DEAM)")

    # Emotion zones
    plt.text(0.6, 0.7, 'Happy', fontsize=12, ha='center')
    plt.text(-0.6, 0.7, 'Relaxed', fontsize=12, ha='center')
    plt.text(-0.6, -0.7, 'Sad', fontsize=12, ha='center')
    plt.text(0.6, -0.7, 'Angry', fontsize=12, ha='center')

    colors = cm.viridis(np.linspace(0, 1, len(predictions_dict)))

    # Plot each clip length
    for (clip_len, vals), color in zip(predictions_dict.items(), colors):
        val_norm = normalize(vals["valence"])
        aro_norm = normalize(vals["arousal"])
        plt.scatter(val_norm, aro_norm, alpha=0.5, s=10, color=color, label=f"{clip_len}s")

    plt.legend(title="Clip Length")
    plt.tight_layout()
    plt.show()

plot_emotion_space(predictions)
