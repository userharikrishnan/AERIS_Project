import os
import time
import json
import wave
import pyaudio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------
# Hyperparameters & Constants
# ---------------------------------------------------------
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 1.5  # Hotword usually takes ~1 second
N_FFT = 400
HOP_LENGTH = 160

MODELS_DIR = "checkpoints"
os.makedirs(MODELS_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "custom_wakeword.pt")
DATA_DIR = "data/wakeword_samples"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------
# Pytorch Neural Network Model
# ---------------------------------------------------------
class WakeWordNet(nn.Module):
    def __init__(self, input_size):
        super(WakeWordNet, self).__init__()
        # Extremely lightweight 1D Conv / MLP
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Binary classification: 1 = Hotword, 0 = Background
        )

    def forward(self, x):
        return self.net(x)

def audio_to_features(audio_data: np.ndarray) -> torch.Tensor:
    """Converts 1D audio waveform into a flattened Spectrogram feature tensor using PyTorch."""
    # Convert numpy array to torch float tensor
    waveform = torch.FloatTensor(audio_data).unsqueeze(0)  # Shape [1, L]
    
    # Compute STFT (Spectrogram)
    # Using return_complex=True as recommended in newer PyTorch
    stft = torch.stft(
        waveform, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH, 
        return_complex=True
    )
    # Get magnitude
    mag = torch.abs(stft)
    
    # Flatten it into a 1D feature vector for our simple MLP
    return mag.flatten()


# ---------------------------------------------------------
# Data Collection
# ---------------------------------------------------------
def record_sample(p, stream, duration=RECORD_SECONDS) -> np.ndarray:
    frames = []
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        
    # Convert bytes to numpy int16 array
    return np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0

def collect_data(num_samples: int = 15):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("\n" + "="*50)
    print(" 🎤 AERIS CUSTOM WAKE-WORD CALIBRATION")
    print("="*50)
    print("We need to record your specific voice saying 'Hey AERIS'.")
    print("This runs 100% offline and takes 2 minutes.\n")

    input("Press ENTER when you are ready to start...")

    X, y = [], []

    # 1. Positive Samples (The Hotword)
    print("\n--- PHASE 1: Positive Samples ---")
    print("When you see [RECORDING], say 'Hey AERIS' clearly.")
    for i in range(num_samples):
        print(f"\nSample {i+1}/{num_samples} -> Get ready...")
        time.sleep(1.0)
        print("🔴 [RECORDING] Say: 'Hey AERIS' ...", end="", flush=True)
        audio = record_sample(p, stream)
        print(" Done!")
        features = audio_to_features(audio)
        X.append(features)
        y.append(1.0) # Label 1 = Hotword

    # 2. Negative Samples (Silence)
    print("\n--- PHASE 2: Background Noise (Silence) ---")
    print("Just remain quiet. Do not speak.")
    for i in range(num_samples // 2):
        print(f"Recording silence {i+1}...", end="", flush=True)
        audio = record_sample(p, stream)
        print(" Done.")
        features = audio_to_features(audio)
        X.append(features)
        y.append(0.0) # Label 0

    # 3. Negative Samples (Other Words)
    print("\n--- PHASE 3: Other Words ---")
    print("Say random command words when prompted (e.g. 'Hey Google', 'Turn off', 'Hello').")
    for i in range(num_samples // 2):
        print(f"\nSample {i+1} -> Get ready...")
        time.sleep(1.0)
        print("🔴 [RECORDING] Say something RANDOM ...", end="", flush=True)
        audio = record_sample(p, stream)
        print(" Done!")
        features = audio_to_features(audio)
        X.append(features)
        y.append(0.0) # Label 0

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Stack tensors
    X_tensor = torch.stack(X)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return X_tensor, y_tensor


# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
def train(X, y):
    print("\n" + "="*50)
    print(" 🧠 TRAINING NEURAL NETWORK...")
    print("="*50)
    
    # DataLoader
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    input_size = X.shape[1]
    model = WakeWordNet(input_size=input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 30
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")

    print("\n✅ Training Complete!")
    
    # Save the model and input size metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
    }, CHECKPOINT_PATH)
    print(f"Model saved securely to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    X, y = collect_data(num_samples=15)
    train(X, y)
