import librosa
import numpy as np
from pathlib import Path
import torch


def process_audio(file_path, sr=22050, duration=30, hop_length=512, n_mels=128):
    # 1. Load audio
    y, sr = librosa.load(file_path, sr=sr, duration=duration)

    # 2. Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 3. Pad or truncate to fixed length
    fixed_frames = int(np.ceil((sr * duration) / hop_length))
    if mel_db.shape[1] < fixed_frames:
        pad_width = fixed_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :fixed_frames]

    # 4. Normalize
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

    # 5. Convert to PyTorch tensor
    # Only add channel dimension (CNN expects (channels, height, width))
    mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)  # shape: (1, n_mels, frames)

    return mel_tensor


