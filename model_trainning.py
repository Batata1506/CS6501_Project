from emotion_CNN import EmotionCNN
import process_audio
import torch
import torch.optim as optim
import torch.nn as nn 
from pathlib import Path

# Model setup
model = EmotionCNN(num_classes=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Audio directory
audio_dir = Path(r"c:\Users\tahal\Documents\ECE_Masters\CS6501\Project\MEMD_audio")
exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

for epoch in range(10):
    running_loss = 0.0
    for p in audio_dir.rglob('*'):
        if p.suffix.lower() in exts:
            try:
                # Preprocess audio
                mel_db = process_audio.process_audio(str(p))  # expect (1, mel_bins, time_frames)

                # TODO: replace with actual label from your dataset!
                label = torch.tensor(0)  # single integer, not [0]

                # Forward pass
                optimizer.zero_grad()
                outputs = model(mel_db)
                loss = criterion(outputs, label.unsqueeze(0))  # ensure batch dimension
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            except Exception as e:
                print(f"Error processing {p.name}: {e}")

    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")
