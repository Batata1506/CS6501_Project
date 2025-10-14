import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import os

# Make pandas tables display nicely
pd.set_option('display.max_colwidth', 120)
pd.set_option('display.precision', 3)

rows15to45 = [f"sample_{i}ms" for i in range(15000, 45000, 500)]

DEAM_DATA_DYNAMIC_AROUSAL = pd.read_csv(
    "DEAM/DEAM_Annotations/annotations/annotations_averaged_per_song/dynamic/arousal.csv",
    index_col="song_id"
)
DEAM_DATA_DYNAMIC_AROUSAL = DEAM_DATA_DYNAMIC_AROUSAL[rows15to45]
#print(DEAM_DATA_DYNAMIC_AROUSAL.head())

DEAM_DATA_DYNAMIC_VALENCE = pd.read_csv(
    "DEAM/DEAM_Annotations/annotations/annotations_averaged_per_song/dynamic/valence.csv",
    index_col="song_id"
)
DEAM_DATA_DYNAMIC_VALENCE = DEAM_DATA_DYNAMIC_VALENCE[rows15to45]
#print(DEAM_DATA_DYNAMIC_VALENCE.head())

mean_dynamic_arousal = DEAM_DATA_DYNAMIC_AROUSAL.mean(axis=1)
mean_dynamic_valence = DEAM_DATA_DYNAMIC_VALENCE.mean(axis=1)

# Scatter plot
"""plt.figure(figsize=(8,6))
plt.scatter(mean_dynamic_valence, mean_dynamic_arousal, alpha=0.7, color='blue')
plt.xlabel("Valence")
plt.ylabel("Arousal")
plt.title("Scatter plot of songs on the Valence-Arousal plane (15s–45s)")
plt.grid(True)
plt.show()"""

DEAM_DATA_STATIC_AVERAGE_ANNOTATIONS = pd.read_csv(
    "DEAM/DEAM_Annotations/annotations/annotations_averaged_per_song/song_level/static_annotations_averaged_songs_1_2000.csv",
    index_col="song_id"
)
#print(DEAM_DATA_STATIC_AVERAGE_ANNOTATIONS.head())

valence = DEAM_DATA_STATIC_AVERAGE_ANNOTATIONS[' valence_mean']
arousal = DEAM_DATA_STATIC_AVERAGE_ANNOTATIONS[' arousal_mean']

# Scatter plot
"""plt.figure(figsize=(8,6))
plt.scatter(valence, arousal, alpha=0.7, color='green')
plt.xlabel("Valence")
plt.ylabel("Arousal")
plt.title("Scatter plot of songs on the Valence-Arousal plane (static annotations)")
plt.grid(True)
plt.show()"""

# Path to the folder containing audio files
audio_folder = "DEAM/DEAM_audio/MEMD_audio"

# List to store extracted data
data = []

# Function to extract audio features from a file
import librosa

def to_1d_array(x):
    """Convert a scalar or array to a 1D numpy array."""
    return np.atleast_1d(x).ravel()

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # --- MFCCs ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = to_1d_array(mfccs.mean(axis=1))
    mfccs_std = to_1d_array(mfccs.std(axis=1))

    # --- Chroma ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = to_1d_array(chroma.mean(axis=1))
    chroma_std = to_1d_array(chroma.std(axis=1))

    # --- Spectral features ---
    spec_centroid = to_1d_array(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    spec_bandwidth = to_1d_array(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
    spec_rolloff = to_1d_array(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())

    # Spectral contrast (7 bands)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = to_1d_array(spec_contrast.mean(axis=1))

    # --- Other features ---
    zcr = to_1d_array(librosa.feature.zero_crossing_rate(y).mean())
    rms = to_1d_array(librosa.feature.rms(y=y).mean())

    # Tempo (beats per minute)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = to_1d_array(tempo)

    # --- Concatenate all features into a single 1D array ---
    features = np.concatenate([
        mfccs_mean, mfccs_std,
        chroma_mean, chroma_std,
        spec_centroid, spec_bandwidth, spec_rolloff,
        spec_contrast_mean,
        zcr, rms, tempo
    ])

    return features

# Loop through all audio files in the folder
for filename in os.listdir(audio_folder):
    if filename.endswith(".wav") or filename.endswith(".mp3"):
        file_path = os.path.join(audio_folder, filename)
        features = extract_features(file_path)
        # Append to the list: Id + features
        data.append([filename] + features.tolist())

# Create a pandas DataFrame
num_features = len(data[0]) - 1
feature_names = []

# MFCCs
feature_names += [f'mfcc_mean_{i+1}' for i in range(13)]
feature_names += [f'mfcc_std_{i+1}' for i in range(13)]

# Chroma
feature_names += [f'chroma_mean_{i+1}' for i in range(12)]
feature_names += [f'chroma_std_{i+1}' for i in range(12)]

# Spectral features
feature_names += ['spec_centroid','spec_bandwidth','spec_rolloff']

# Spectral contrast (7 bands)
feature_names += [f'spec_contrast_{i+1}' for i in range(7)]

# ZCR, RMS, tempo
feature_names += ['zcr','rms','tempo']

df = pd.DataFrame(data, columns=['Id'] + feature_names)

# Save the DataFrame to a CSV file
df.to_csv("audio_features.csv", index=False)

print("Done !")