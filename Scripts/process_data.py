# process_data.py
import pandas as pd
from pathlib import Path
import numpy as np

# ================================
# 1️⃣ Paths + Load Annotations
# ================================
DEAM_ROOT = Path("C:/Users/tahal/Documents/ECE_Masters/CS6501/Project/DEAM")
ANNOT_PATH = DEAM_ROOT / "annotations" / "annotations averaged per song" / "song_level"

# Load annotation file
anno_df = pd.read_csv(ANNOT_PATH / "static_annotations_averaged_songs_1_2000.csv")
anno_df.columns = anno_df.columns.str.strip()  # fix leading spaces

print(f"Loaded {len(anno_df)} songs")
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
    X = feats_df.drop(columns=['frameTime']).to_numpy(dtype=np.float32)
    return X[:frames_to_keep]


# ================================
# 3️⃣ Build Dataset
# ================================
truncate_len = 10  # seconds (you can change this to 5, 15, 45 later)
sr = 2             # 2 Hz = 0.5s per frame

truncated_data = []

for _, row in anno_df.iterrows():
    sid = int(row["song_id"])
    feat_path = DEAM_ROOT / "features" / f"{sid}.csv"
    if not feat_path.exists():
        continue

    # Load and truncate
    feats = pd.read_csv(feat_path, sep=';')
    X_short = truncate_features(feats, seconds=truncate_len, sr=sr)

    truncated_data.append({
        "song_id": sid,
        "features": X_short,
        "valence": row["valence_mean"],
        "arousal": row["arousal_mean"]
    })

print(f"✅ Loaded {len(truncated_data)} truncated songs ({truncate_len}s each)")
print("Example shape:", truncated_data[0]['features'].shape)
