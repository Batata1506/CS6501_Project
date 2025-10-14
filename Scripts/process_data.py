# process_data.py
import pandas as pd
from pathlib import Path
import numpy as np
import pickle

# ================================
# 1️⃣ Paths + Load Annotations
# ================================
DEAM_ROOT = Path("D:/ECE_Masters/CS6501/Project/DEAM")
ANNOT_PATH = DEAM_ROOT / "annotations" / "annotations averaged per song" / "song_level"

# Load annotation file
anno_df = pd.read_csv(ANNOT_PATH / "static_annotations_averaged_songs_1_2000.csv")
anno_df.columns = anno_df.columns.str.strip()  # remove leading/trailing spaces

print(f"✅ Loaded {len(anno_df)} songs")
print("Annotation columns:", anno_df.columns.tolist())
print(anno_df.head())


# ================================
# 2️⃣ Truncate Function
# ================================
def truncate_features(feats_df, seconds=10, sr=2):
    """
    Shorten a song's feature dataframe to the first <seconds>.
    Each frame = 1/sr seconds (DEAM uses 0.5s/frame, sr=2).
    """
    frames_to_keep = int(seconds * sr)
    if 'frameTime' in feats_df.columns:
        X = feats_df.drop(columns=['frameTime']).to_numpy(dtype=np.float32)
    else:
        X = feats_df.to_numpy(dtype=np.float32)
    return X[:frames_to_keep]


# ================================
# 3️⃣ Build Truncated Dataset(s)
# ================================
truncate_lengths = [10, 15, 20, 25, 30, 35, 40, 45]  # seconds to test
sr = 2  # 2 Hz = 0.5 s per frame

for t_len in truncate_lengths:
    truncated_data = []

    for _, row in anno_df.iterrows():
        sid = int(row["song_id"])
        feat_path = DEAM_ROOT / "features" / f"{sid}.csv"
        if not feat_path.exists():
            continue

        try:
            feats = pd.read_csv(feat_path, sep=';')
        except Exception as e:
            print(f"⚠️ Could not read {feat_path.name}: {e}")
            continue

        X_short = truncate_features(feats, seconds=t_len, sr=sr)

        truncated_data.append({
            "song_id": sid,
            "features": X_short,
            "valence": row["valence_mean"],
            "arousal": row["arousal_mean"]
        })

    # ================================
    # 4️⃣ Save Processed Dataset
    # ================================
    output_file = f"truncated_data_{t_len}s.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(truncated_data, f)

    print(f"\n✅ Saved {len(truncated_data)} songs for {t_len}s clips → {output_file}")
    print("Example shape:", truncated_data[0]['features'].shape)
