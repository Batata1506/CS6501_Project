# baseline_models_all.py
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from pathlib import Path
import re

# ================================
# 1️⃣ Paths
# ================================
DATA_DIR = Path("D:/ECE_Masters/CS6501/Project/processed_data")
pkl_files = sorted(DATA_DIR.glob("truncated_data_*s.pkl"))

if not pkl_files:
    raise FileNotFoundError(f"No truncated_data_*.pkl files found in {DATA_DIR}")

print(f"✅ Found {len(pkl_files)} processed datasets:")
for f in pkl_files:
    print("  -", f.name)

# ================================
# 2️⃣ Metric Helpers
# ================================
def ccc(y_true, y_pred):
    y_true_mean, y_pred_mean = np.mean(y_true), np.mean(y_pred)
    cov = np.mean((y_true - y_true_mean)*(y_pred - y_pred_mean))
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    return (2 * cov) / (var_true + var_pred + (y_true_mean - y_pred_mean)**2 + 1e-8)

# ================================
# 3️⃣ Storage for Results
# ================================
results = []

# ================================
# 4️⃣ Loop through all truncations
# ================================
for file in pkl_files:
    match = re.search(r"(\d+)s", file.name)
    clip_len = int(match.group(1)) if match else None

    with open(file, "rb") as f:
        truncated_data = pickle.load(f)

    print(f"\n🎵 Processing {file.name} ({len(truncated_data)} samples)")

    # Prepare X and y
    X = []
    y_val, y_aro = [], []

    for d in truncated_data:
        feat = d["features"].flatten()
        X.append(feat)
        y_val.append(d["valence"])
        y_aro.append(d["arousal"])

    X = np.array(X)
    y_val = np.array(y_val)
    y_aro = np.array(y_aro)

    # Split
    X_train, X_test, yv_train, yv_test = train_test_split(X, y_val, test_size=0.2, random_state=42)
    _, _, ya_train, ya_test = train_test_split(X, y_aro, test_size=0.2, random_state=42)

    # Train SVR
    svr_val = SVR(kernel='rbf', C=10, epsilon=0.1)
    svr_aro = SVR(kernel='rbf', C=10, epsilon=0.1)
    svr_val.fit(X_train, yv_train)
    svr_aro.fit(X_train, ya_train)

    # Predictions
    yv_pred = svr_val.predict(X_test)
    ya_pred = svr_aro.predict(X_test)

    # Metrics
    rmse_val = np.sqrt(mean_squared_error(yv_test, yv_pred))
    rmse_aro = np.sqrt(mean_squared_error(ya_test, ya_pred))
    r_val, _ = pearsonr(yv_test, yv_pred)
    r_aro, _ = pearsonr(ya_test, ya_pred)
    ccc_val = ccc(yv_test, yv_pred)
    ccc_aro = ccc(ya_test, ya_pred)

    results.append({
        "clip_len": clip_len,
        "rmse_val": rmse_val,
        "rmse_aro": rmse_aro,
        "r_val": r_val,
        "r_aro": r_aro,
        "ccc_val": ccc_val,
        "ccc_aro": ccc_aro
    })

    print(f"✅ {clip_len}s → RMSE V:{rmse_val:.3f}, A:{rmse_aro:.3f} | r V:{r_val:.3f}, A:{r_aro:.3f}")

# ================================
# 5️⃣ Convert to Numpy for Plotting
# ================================
results = sorted(results, key=lambda x: x["clip_len"])
durations = [r["clip_len"] for r in results]

rmse_val = [r["rmse_val"] for r in results]
rmse_aro = [r["rmse_aro"] for r in results]
r_val = [r["r_val"] for r in results]
r_aro = [r["r_aro"] for r in results]
ccc_val = [r["ccc_val"] for r in results]
ccc_aro = [r["ccc_aro"] for r in results]

# ================================
# 6️⃣ Plot RMSE
# ================================
plt.figure(figsize=(8, 5))
plt.plot(durations, rmse_val, 'o-', label='Valence RMSE')
plt.plot(durations, rmse_aro, 's-', label='Arousal RMSE')
plt.xlabel("Audio Clip Length (s)")
plt.ylabel("RMSE (lower = better)")
plt.title("SVR Performance vs Clip Length (DEAM)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ================================
# 7️⃣ Plot Correlation & CCC
# ================================
plt.figure(figsize=(8, 5))
plt.plot(durations, r_val, 'o-', label='Valence r')
plt.plot(durations, r_aro, 's-', label='Arousal r')
plt.plot(durations, ccc_val, 'o--', label='Valence CCC')
plt.plot(durations, ccc_aro, 's--', label='Arousal CCC')
plt.xlabel("Audio Clip Length (s)")
plt.ylabel("Score (higher = better)")
plt.title("SVR Correlation and CCC vs Clip Length")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
