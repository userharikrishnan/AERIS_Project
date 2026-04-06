import os
import torch
import torch.nn as nn
import numpy as np

N_FFT = 400
HOP_LENGTH = 160
RATE = 16000
RECORD_SECONDS = 1.5
CHANNELS = 1

class WakeWordNet(nn.Module):
    def __init__(self, input_size):
        super(WakeWordNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class WakeWordDetector:
    def __init__(self, checkpoint_path="checkpoints/custom_wakeword.pt"):
        self.device = torch.device("cpu")
        self.model = None
        
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                input_size = checkpoint['input_size']
                self.model = WakeWordNet(input_size=input_size)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                print(f"[WakeWordDetector] Loaded custom AERIS acoustic model ({input_size} features)")
            except Exception as e:
                print(f"[WakeWordDetector] Failed to load local model: {e}")

    def is_active(self):
        return self.model is not None

    def audio_to_features(self, audio_data_np: np.ndarray) -> torch.Tensor:
        waveform = torch.FloatTensor(audio_data_np).unsqueeze(0).to(self.device)
        stft = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, return_complex=True)
        mag = torch.abs(stft)
        return mag.flatten().unsqueeze(0)  # Shape: [1, Features]

    def predict(self, raw_audio_bytes: bytes, target_length_sec=1.5) -> bool:
        """
        Takes raw 16-bit PCM audio bytes, normalizes, extracts features, and runs inference.
        Returns True if the hotword is detected.
        """
        if not self.model: return False

        # Convert bytes to floats
        audio_np = np.frombuffer(raw_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Trim or pad to exactly 'target_length_sec'
        target_len = int(RATE * target_length_sec)
        
        if len(audio_np) > target_len:
            audio_np = audio_np[:target_len]
        elif len(audio_np) < target_len:
            audio_np = np.pad(audio_np, (0, target_len - len(audio_np)), 'constant')

        features = self.audio_to_features(audio_np)
        
        # Ensure exact feature size match with the trained model
        input_size = self.model.net[0].in_features
        if features.shape[1] > input_size:
            features = features[:, :input_size]
        elif features.shape[1] < input_size:
            pad = nn.ConstantPad1d((0, input_size - features.shape[1]), 0)
            features = pad(features)
        
        with torch.no_grad():
            output = self.model(features)
            confidence = output.item()
            
        return confidence > 0.55  # Lowered from 0.8 — MLP trained on 15 samples can't reliably exceed 0.8
