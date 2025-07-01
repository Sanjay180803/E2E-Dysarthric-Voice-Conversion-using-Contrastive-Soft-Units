"""
Enhanced preprocessing with automatic severity classification
Novel contribution: Automated dysarthria severity assessment
"""

import torch
import torch.nn as nn
import librosa
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path

class DysarthriaSeverityClassifier(nn.Module):
    """
    Novel automatic severity classifier based on speech characteristics
    
    Key Innovation: Multi-modal severity assessment using acoustic features
    """
    
    def __init__(self, input_dim: int = 80):
        super().__init__()
        
        # Acoustic feature extractor
        self.acoustic_encoder = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Prosodic feature extractor
        self.prosody_encoder = nn.Sequential(
            nn.Linear(10, 64),  # 10 prosodic features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Fusion and classification
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 4)  # healthy, mild, moderate, severe
        )
        
    def extract_prosodic_features(self, audio: np.ndarray, sr: int = 16000) -> torch.Tensor:
        """Extract prosodic features for severity assessment"""
        # F0 statistics
        f0, _, _ = librosa.pyin(audio, fmin=50, fmax=400, sr=sr)
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) > 0:
            f0_mean = np.mean(f0_clean)
            f0_std = np.std(f0_clean)
            f0_range = np.max(f0_clean) - np.min(f0_clean)
            voiced_ratio = len(f0_clean) / len(f0)
        else:
            f0_mean = f0_std = f0_range = voiced_ratio = 0
        
        # Energy and rhythm features
        energy = librosa.feature.rms(y=audio)[0]
        energy_std = np.std(energy)
        energy_mean = np.mean(energy)
        
        # Speaking rate estimation
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        features = torch.tensor([
            f0_mean, f0_std, f0_range, voiced_ratio,
            energy_mean, energy_std, tempo,
            spectral_centroid, spectral_rolloff, zero_crossing_rate
        ], dtype=torch.float32)
        
        return features
    
    def forward(self, mel_spec: torch.Tensor, prosodic_features: torch.Tensor) -> torch.Tensor:
        """Forward pass for severity classification"""
        # Acoustic encoding
        acoustic_features = self.acoustic_encoder(mel_spec.unsqueeze(1))
        
        # Prosodic encoding
        prosody_features = self.prosody_encoder(prosodic_features)
        
        # Fusion and classification
        combined = torch.cat([acoustic_features, prosody_features], dim=-1)
        severity_logits = self.classifier(combined)
        
        return severity_logits

class EnhancedDataPreprocessor:
    """
    Enhanced preprocessor with automatic severity assessment
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.severity_classifier = DysarthriaSeverityClassifier()
        
        if model_path and Path(model_path).exists():
            self.severity_classifier.load_state_dict(torch.load(model_path))
            self.severity_classifier.eval()
            
        self.severity_names = ['healthy', 'mild', 'moderate', 'severe']
    
    def assess_severity(self, audio_path: str) -> Tuple[str, float]:
        """
        Assess dysarthria severity automatically
        
        Returns:
            (predicted_severity, confidence_score)
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
        mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
        
        # Extract prosodic features
        prosodic_features = self.severity_classifier.extract_prosodic_features(audio, sr)
        prosodic_features = prosodic_features.unsqueeze(0)
        
        # Predict severity
        with torch.no_grad():
            severity_logits = self.severity_classifier(mel_spec, prosodic_features)
            severity_probs = torch.softmax(severity_logits, dim=-1)
            
            predicted_idx = torch.argmax(severity_probs, dim=-1).item()
            confidence = torch.max(severity_probs, dim=-1)[0].item()
            
            predicted_severity = self.severity_names[predicted_idx]
            
        return predicted_severity, confidence