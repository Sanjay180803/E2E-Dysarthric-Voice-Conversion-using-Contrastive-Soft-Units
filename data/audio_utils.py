"""
Audio processing utilities for dysarthric voice conversion
"""

import librosa
import torch
import torchaudio
import numpy as np
import soundfile as sf
from typing import Tuple, Optional, Union
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class AudioProcessor:
    """Audio processing utilities"""
    
    def __init__(self, 
                 sampling_rate: int = 16000,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 win_length: int = 1024,
                 n_mel_channels: int = 80,
                 mel_fmin: float = 0.0,
                 mel_fmax: float = 8000.0,
                 center: bool = False,
                 normalize: bool = True):
        
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.center = center
        self.normalize = normalize
        
        # Initialize mel filterbank
        self.mel_basis = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax
        )
        self.mel_basis = torch.from_numpy(self.mel_basis).float()
        
    def load_audio(self, 
                   audio_path: Union[str, Path], 
                   normalize: bool = None) -> torch.Tensor:
        """Load audio file and convert to torch tensor"""
        normalize = normalize if normalize is not None else self.normalize
        
        # Load with soundfile for better compatibility
        audio, sr = sf.read(str(audio_path))
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
            
        # Resample if necessary
        if sr != self.sampling_rate:
            audio = librosa.resample(
                audio, 
                orig_sr=sr, 
                target_sr=self.sampling_rate
            )
        
        # Convert to tensor
        audio = torch.from_numpy(audio).float()
        
        # Normalize
        if normalize:
            audio = audio / max(abs(audio.max()), abs(audio.min()))
            
        return audio
    
    def save_audio(self, 
                   audio: torch.Tensor, 
                   save_path: Union[str, Path],
                   normalize: bool = True):
        """Save audio tensor to file"""
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
            
        if normalize:
            audio = audio / max(abs(audio.max()), abs(audio.min()))
            
        sf.write(str(save_path), audio, self.sampling_rate)
    
    def get_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to mel spectrogram"""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        # STFT
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=self.center,
            window=torch.hann_window(self.win_length),
            return_complex=True
        )
        
        # Magnitude spectrogram
        mag_spec = torch.abs(spec)
        
        # Apply mel filterbank
        mel_spec = torch.matmul(
            self.mel_basis.to(mag_spec.device), 
            mag_spec
        )
        
        # Log scale
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        
        return mel_spec.squeeze(0)  # Remove batch dimension
    
    def mel_to_linear(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to linear spectrogram (approximate)"""
        # This is an approximation - in practice, you might want a more 
        # sophisticated approach or use the original linear spectrogram
        mel_basis_pinv = torch.pinverse(self.mel_basis.to(mel_spec.device))
        linear_spec = torch.matmul(mel_basis_pinv, mel_spec)
        return linear_spec
    
    def griffin_lim(self, 
                    mel_spec: torch.Tensor, 
                    n_iters: int = 32) -> torch.Tensor:
        """Convert mel spectrogram to audio using Griffin-Lim algorithm"""
        # Convert mel to linear spectrogram
        linear_spec = self.mel_to_linear(mel_spec)
        
        # Griffin-Lim reconstruction
        audio = librosa.griffinlim(
            linear_spec.cpu().numpy(),
            n_iter=n_iters,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=self.center
        )
        
        return torch.from_numpy(audio).float()
    
    def preprocess_audio(self, 
                        audio_path: Union[str, Path],
                        max_length: Optional[float] = None,
                        min_length: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess audio file for training
        Returns: (audio_tensor, mel_spectrogram)
        """
        # Load audio
        audio = self.load_audio(audio_path)
        
        # Filter by length if specified
        if max_length is not None:
            max_samples = int(max_length * self.sampling_rate)
            if len(audio) > max_samples:
                # Random crop
                start = torch.randint(0, len(audio) - max_samples + 1, (1,)).item()
                audio = audio[start:start + max_samples]
                
        if min_length is not None:
            min_samples = int(min_length * self.sampling_rate)
            if len(audio) < min_samples:
                # Pad with zeros
                pad_length = min_samples - len(audio)
                audio = torch.cat([audio, torch.zeros(pad_length)])
        
        # Get mel spectrogram
        mel_spec = self.get_mel_spectrogram(audio)
        
        return audio, mel_spec
    
    def apply_spec_augment(self,
                          mel_spec: torch.Tensor,
                          freq_mask_N: int = 2,
                          freq_mask_F: int = 10,
                          time_mask_N: int = 2,
                          time_mask_T: int = 50,
                          time_mask_p: float = 0.2) -> torch.Tensor:
        """Apply SpecAugment to mel spectrogram"""
        mel_spec = mel_spec.clone()
        
        # Time masking
        for _ in range(time_mask_N):
            t = torch.randint(0, min(time_mask_T, int(mel_spec.size(1) * time_mask_p)), (1,)).item()
            if t > 0:
                t0 = torch.randint(0, mel_spec.size(1) - t, (1,)).item()
                mel_spec[:, t0:t0+t] = 0
        
        # Frequency masking
        for _ in range(freq_mask_N):
            f = torch.randint(0, min(freq_mask_F, mel_spec.size(0) // 4), (1,)).item()
            if f > 0:
                f0 = torch.randint(0, mel_spec.size(0) - f, (1,)).item()
                mel_spec[f0:f0+f, :] = 0
                
        return mel_spec

def dynamic_range_compression(x: torch.Tensor, 
                            C: float = 1, 
                            clip_val: float = 1e-5) -> torch.Tensor:
    """Dynamic range compression"""
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression(x: torch.Tensor, C: float = 1) -> torch.Tensor:
    """Dynamic range decompression"""
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    """Spectral normalization"""
    output = dynamic_range_compression(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    """Spectral denormalization"""
    output = dynamic_range_decompression(magnitudes)
    return output

class MelSpectrogramTransform:
    """Mel spectrogram transform for torchaudio compatibility"""
    
    def __init__(self, audio_config):
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_config.sampling_rate,
            n_fft=audio_config.n_fft,
            hop_length=audio_config.hop_length,
            win_length=audio_config.win_length,
            n_mels=audio_config.n_mel_channels,
            f_min=audio_config.mel_fmin,
            f_max=audio_config.mel_fmax,
            center=audio_config.center
        )
        
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        mel_spec = self.transform(audio)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        return mel_spec.squeeze(0)