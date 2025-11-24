# process_data.py
import pandas as pd
from pathlib import Path
import numpy as np
import pickle

# ============================================
# 1. Paths + Load Annotations
# ============================================
DEAM_ROOT = Path("D:/ECE_Masters/CS6501/Project/DEAM")
ANNOT_PATH = DEAM_ROOT / "annotations" / "annotations averaged per song" / "song_level"

# Processed data output dir (relative to where you run the script)
OUTPUT_DIR = Path("D:/ECE_Masters/CS6501/Project/processed_data")
OUTPUT_DIR.mkdir(exist_ok=True)

anno_df = pd.read_csv(ANNOT_PATH / "static_annotations_averaged_songs_1_2000.csv")
anno_df.columns = anno_df.columns.str.strip()

print(f"Loaded {len(anno_df)} songs")
print("Annotation columns:", anno_df.columns.tolist())
print(anno_df.head())


# ============================================
# 2. Helpers for feature matrices and windows
# ============================================

def get_feature_matrix(feats_df: pd.DataFrame) -> np.ndarray:
    """Convert feature DataFrame to numeric matrix [T, F], dropping frameTime if present."""
    if 'frameTime' in feats_df.columns:
        feats_df = feats_df.drop(columns=['frameTime'])
    return feats_df.to_numpy(dtype=np.float32)


def find_most_active_window(X: np.ndarray, window_frames: int) -> int:
    """
    Find the start index of the window with the highest activity.
    Activity per frame = L2 norm of feature vector.
    """
    n_frames = X.shape[0]
    if n_frames <= window_frames:
        return 0

    frame_activity = np.linalg.norm(X, axis=1)  # [T]
    kernel = np.ones(window_frames, dtype=np.float32)
    window_scores = np.convolve(frame_activity, kernel, mode='valid')  # [T - window_frames + 1]

    return int(np.argmax(window_scores))


def extract_most_active_segment(feats_df: pd.DataFrame, seconds: int, sr: int = 2):
    """
    Extract the most frequency-active segment of given duration (in seconds).

    Window selection is based on:
        - MFCC features
        - Spectral features
        - RMS / loudness

    The segment returned contains the full feature vector for each frame.

    Returns:
        segment: np.ndarray [window_frames, F]
        start_frame: int
        start_time_sec: float
    """
    # 1) Use only frequency-related features for activity scoring
    freq_cols = [
        c for c in feats_df.columns
        if c.startswith("pcm_fftMag")          # MFCC + spectral FFT-based features
        or "mfcc" in c.lower()                 # MFCCs
        or "spectral" in c.lower()             # spectral stats
        or "rms" in c.lower()                  # RMS energy
        or "loudness" in c.lower()             # loudness
    ]

    if not freq_cols:
        # Fallback: use all numeric features (dropping frameTime)
        X_activity = get_feature_matrix(feats_df)
    else:
        X_activity = feats_df[freq_cols].to_numpy(dtype=np.float32)

    # 2) Full feature matrix for the actual model input
    X_full = get_feature_matrix(feats_df)

    window_frames = int(seconds * sr)

    if X_activity.shape[0] == 0:
        return None, None, None

    start_frame = find_most_active_window(X_activity, window_frames)
    end_frame = start_frame + window_frames

    # Extract full-feature segment aligned with the most active window
    segment = X_full[start_frame:end_frame]
    segment = segment[:window_frames]  # clamp if track is shorter

    start_time_sec = start_frame / sr
    return segment, start_frame, start_time_sec


# ============================================
# 3. Build datasets for various segment lengths
# ============================================
segment_lengths = [10, 20, 30, 40]
sr = 2  # 0.5s per frame (DEAM)

for t_len in segment_lengths:
    truncated_data = []
    print(f"\nProcessing {t_len} second segments (most active)")

    for _, row in anno_df.iterrows():
        sid = int(row["song_id"])
        feat_path = DEAM_ROOT / "features" / f"{sid}.csv"

        if not feat_path.exists():
            continue

        try:
            feats = pd.read_csv(feat_path, sep=';')
        except Exception as e:
            print(f"Could not read {feat_path.name}: {e}")
            continue

        seg_feats, start_frame, start_time = extract_most_active_segment(
            feats, seconds=t_len, sr=sr
        )

        if seg_feats is None:
            continue

        truncated_data.append({
            "song_id": sid,
            "features": seg_feats,
            "valence": row["valence_mean"],
            "arousal": row["arousal_mean"],
            "start_frame": start_frame,
            "start_time_sec": start_time
        })

    # Save once per segment length, after all songs
    output_file = OUTPUT_DIR / f"active_segment_data_{t_len}s.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(truncated_data, f)

    if len(truncated_data) > 0:
        print(f"Saved {len(truncated_data)} songs for {t_len}s segments -> {output_file}")
        print("Example feature shape:", truncated_data[0]["features"].shape)
        print("Example start time (s):", truncated_data[0]["start_time_sec"])
    else:
        print(f"No data saved for {t_len}s")
