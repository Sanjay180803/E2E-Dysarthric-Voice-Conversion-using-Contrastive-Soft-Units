"""
Advanced audio preprocessing utilities for dysarthric voice conversion
"""

import os
import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
from typing import Tuple, Optional, List, Dict, Union
from pathlib import Path
import warnings
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

class AudioPreprocessor:
    """
    Advanced audio preprocessing for speech analysis
    """
    
    def __init__(self, 
                 target_sr: int = 16000,
                 normalize: bool = True,
                 trim_silence: bool = True,
                 filter_noise: bool = True):
        
        self.target_sr = target_sr
        self.normalize = normalize
        self.trim_silence = trim_silence
        self.filter_noise = filter_noise
    
    def load_and_preprocess(self, 
                           audio_path: Union[str, Path],
                           max_duration: Optional[float] = None,
                           min_duration: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
        """
        Load and preprocess audio file with comprehensive processing
        
        Args:
            audio_path: Path to audio file
            max_duration: Maximum duration in seconds
            min_duration: Minimum duration in seconds
            
        Returns:
            Tuple of (processed_audio, metadata)
        """
        try:
            # Load audio
            audio, sr = sf.read(str(audio_path))
            
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            # Initialize metadata
            metadata = {
                "original_sr": sr,
                "original_duration": len(audio) / sr,
                "original_samples": len(audio),
                "preprocessing_steps": []
            }
            
            # Resample if necessary
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                metadata["preprocessing_steps"].append("resampled")
            
            # Validate duration
            duration = len(audio) / self.target_sr
            
            if min_duration and duration < min_duration:
                # Pad with silence if too short
                target_samples = int(min_duration * self.target_sr)
                padding = target_samples - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')
                metadata["preprocessing_steps"].append("padded")
            
            if max_duration and duration > max_duration:
                # Trim if too long
                target_samples = int(max_duration * self.target_sr)
                audio = audio[:target_samples]
                metadata["preprocessing_steps"].append("trimmed")
            
            # Noise filtering
            if self.filter_noise:
                audio = self._apply_noise_filter(audio)
                metadata["preprocessing_steps"].append("noise_filtered")
            
            # Trim silence
            if self.trim_silence:
                audio, trim_info = self._trim_silence(audio)
                metadata["preprocessing_steps"].append("silence_trimmed")
                metadata["trim_info"] = trim_info
            
            # Normalize
            if self.normalize:
                audio = self._normalize_audio(audio)
                metadata["preprocessing_steps"].append("normalized")
            
            # Final metadata
            metadata.update({
                "final_sr": self.target_sr,
                "final_duration": len(audio) / self.target_sr,
                "final_samples": len(audio),
                "rms": np.sqrt(np.mean(audio**2)),
                "max_amplitude": np.max(np.abs(audio)),
                "dynamic_range": np.max(audio) - np.min(audio)
            })
            
            return audio, metadata
            
        except Exception as e:
            raise ValueError(f"Error processing {audio_path}: {str(e)}")
    
    def _apply_noise_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise reduction filter"""
        # High-pass filter to remove low-frequency noise
        nyquist = self.target_sr / 2
        low_cutoff = 80  # Hz
        high = low_cutoff / nyquist
        
        b, a = butter(N=5, Wn=high, btype='high')
        filtered_audio = filtfilt(b, a, audio)
        
        return filtered_audio
    
    def _trim_silence(self, 
                     audio: np.ndarray,
                     top_db: int = 20,
                     frame_length: int = 2048,
                     hop_length: int = 512) -> Tuple[np.ndarray, Dict]:
        """
        Trim silence from beginning and end of audio
        """
        # Use librosa's trim function
        trimmed_audio, index = librosa.effects.trim(
            audio,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        trim_info = {
            "start_sample": index[0],
            "end_sample": index[1],
            "trimmed_start": index[0] / self.target_sr,
            "trimmed_end": (len(audio) - index[1]) / self.target_sr,
            "original_length": len(audio),
            "trimmed_length": len(trimmed_audio)
        }
        
        return trimmed_audio, trim_info
    
    def _normalize_audio(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """
        Normalize audio to target dB level
        """
        # RMS normalization
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            # Convert target dB to linear scale
            target_rms = 10**(target_db / 20)
            audio = audio * (target_rms / rms)
        
        # Ensure no clipping
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val * 0.99
        
        return audio

class SpeechQualityAnalyzer:
    """
    Analyze speech quality metrics for dysarthric speech
    """
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
    
    def analyze_speech_quality(self, audio: np.ndarray) -> Dict:
        """
        Comprehensive speech quality analysis
        """
        metrics = {}
        
        # Basic signal metrics
        metrics.update(self._compute_signal_metrics(audio))
        
        # Spectral metrics
        metrics.update(self._compute_spectral_metrics(audio))
        
        # Prosodic metrics
        metrics.update(self._compute_prosodic_metrics(audio))
        
        # Voice quality metrics
        metrics.update(self._compute_voice_quality_metrics(audio))
        
        return metrics
    
    def _compute_signal_metrics(self, audio: np.ndarray) -> Dict:
        """Compute basic signal metrics"""
        return {
            "duration": len(audio) / self.sr,
            "rms": np.sqrt(np.mean(audio**2)),
            "energy": np.sum(audio**2),
            "zcr": np.mean(librosa.feature.zero_crossing_rate(audio)[0]),
            "max_amplitude": np.max(np.abs(audio)),
            "snr_estimate": self._estimate_snr(audio)
        }
    
    def _compute_spectral_metrics(self, audio: np.ndarray) -> Dict:
        """Compute spectral characteristics"""
        # Compute spectrum
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        # Spectral centroid, rolloff, flatness
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
        
        return {
            "spectral_centroid_mean": np.mean(spectral_centroids),
            "spectral_centroid_std": np.std(spectral_centroids),
            "spectral_rolloff_mean": np.mean(spectral_rolloff),
            "spectral_flatness_mean": np.mean(spectral_flatness),
            "bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)[0])
        }
    
    def _compute_prosodic_metrics(self, audio: np.ndarray) -> Dict:
        """Compute prosodic features"""
        # Fundamental frequency (F0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=self.sr
        )
        
        # Remove NaN values
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) > 0:
            f0_metrics = {
                "f0_mean": np.mean(f0_clean),
                "f0_std": np.std(f0_clean),
                "f0_min": np.min(f0_clean),
                "f0_max": np.max(f0_clean),
                "f0_range": np.max(f0_clean) - np.min(f0_clean),
                "voiced_percentage": np.mean(voiced_flag) * 100
            }
        else:
            f0_metrics = {
                "f0_mean": 0, "f0_std": 0, "f0_min": 0, 
                "f0_max": 0, "f0_range": 0, "voiced_percentage": 0
            }
        
        # Rhythm analysis
        rhythm_metrics = self._analyze_rhythm(audio)
        
        return {**f0_metrics, **rhythm_metrics}
    
    def _compute_voice_quality_metrics(self, audio: np.ndarray) -> Dict:
        """Compute voice quality indicators"""
        # Jitter and shimmer approximation
        f0, _, _ = librosa.pyin(audio, fmin=50, fmax=400, sr=self.sr)
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) > 1:
            # Approximate jitter (F0 variability)
            jitter = np.std(np.diff(f0_clean)) / np.mean(f0_clean) if np.mean(f0_clean) > 0 else 0
        else:
            jitter = 0
        
        # Harmonics-to-noise ratio approximation
        hnr = self._estimate_hnr(audio)
        
        return {
            "jitter_approx": jitter,
            "hnr_estimate": hnr,
            "voicing_consistency": len(f0_clean) / len(f0) if len(f0) > 0 else 0
        }
    
    def _analyze_rhythm(self, audio: np.ndarray) -> Dict:
        """Analyze speech rhythm patterns"""
        # Energy-based rhythm analysis
        frame_length = int(0.025 * self.sr)  # 25ms frames
        hop_length = int(0.010 * self.sr)    # 10ms hop
        
        # Compute frame energy
        energy = librosa.feature.rms(
            y=audio, frame_length=frame_length, hop_length=hop_length
        )[0]
        
        # Find energy peaks (syllable nuclei approximation)
        peaks, _ = find_peaks(energy, height=np.mean(energy), distance=int(0.1 * self.sr / hop_length))
        
        if len(peaks) > 1:
            # Inter-peak intervals
            intervals = np.diff(peaks) * hop_length / self.sr
            rhythm_metrics = {
                "syllable_rate": len(peaks) / (len(audio) / self.sr),
                "rhythm_regularity": 1 / (1 + np.std(intervals)) if len(intervals) > 0 else 0,
                "pause_count": len(np.where(energy < 0.1 * np.mean(energy))[0])
            }
        else:
            rhythm_metrics = {
                "syllable_rate": 0,
                "rhythm_regularity": 0,
                "pause_count": 0
            }
        
        return rhythm_metrics
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio"""
        # Simple SNR estimation based on energy distribution
        frame_length = int(0.025 * self.sr)
        hop_length = int(0.010 * self.sr)
        
        # Compute frame energies
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        frame_energies = np.sum(frames**2, axis=0)
        
        # Assume top 20% are signal, bottom 20% are noise
        sorted_energies = np.sort(frame_energies)
        signal_energy = np.mean(sorted_energies[-int(0.2 * len(sorted_energies)):])
        noise_energy = np.mean(sorted_energies[:int(0.2 * len(sorted_energies))])
        
        if noise_energy > 0:
            snr_db = 10 * np.log10(signal_energy / noise_energy)
        else:
            snr_db = float('inf')
        
        return snr_db
    
    def _estimate_hnr(self, audio: np.ndarray) -> float:
        """Estimate harmonics-to-noise ratio"""
        # Autocorrelation-based HNR estimation
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find the first peak (fundamental period)
        if len(autocorr) > 1:
            max_lag = min(int(0.02 * self.sr), len(autocorr) - 1)  # 20ms max
            if max_lag > 1:
                peak_idx = np.argmax(autocorr[1:max_lag]) + 1
                hnr_ratio = autocorr[peak_idx] / autocorr[0] if autocorr[0] > 0 else 0
                hnr_db = 20 * np.log10(hnr_ratio / (1 - hnr_ratio + 1e-10))
                return np.clip(hnr_db, -40, 40)  # Reasonable HNR range
        
        return 0.0

class DatasetValidator:
    """
    Validate dataset quality and consistency
    """
    
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        self.analyzer = SpeechQualityAnalyzer(target_sr)
    
    def validate_dataset(self, 
                        audio_files: List[str],
                        min_duration: float = 0.5,
                        max_duration: float = 20.0,
                        min_snr: float = 5.0) -> Dict:
        """
        Validate entire dataset and return quality report
        """
        validation_results = {
            "total_files": len(audio_files),
            "valid_files": [],
            "invalid_files": [],
            "warnings": [],
            "statistics": {},
            "quality_metrics": []
        }
        
        print(f"Validating {len(audio_files)} audio files...")
        
        for audio_file in tqdm(audio_files, desc="Validating"):
            try:
                # Load audio
                audio, sr = sf.read(audio_file)
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                
                # Check basic requirements
                duration = len(audio) / sr
                
                is_valid = True
                issues = []
                
                # Duration check
                if duration < min_duration:
                    is_valid = False
                    issues.append(f"Too short: {duration:.2f}s")
                elif duration > max_duration:
                    is_valid = False
                    issues.append(f"Too long: {duration:.2f}s")
                
                # Silence check
                if np.max(np.abs(audio)) < 0.001:
                    is_valid = False
                    issues.append("Audio too quiet")
                
                # Quality analysis
                if is_valid:
                    quality_metrics = self.analyzer.analyze_speech_quality(audio)
                    
                    # SNR check
                    if quality_metrics["snr_estimate"] < min_snr:
                        validation_results["warnings"].append(
                            f"{audio_file}: Low SNR ({quality_metrics['snr_estimate']:.1f} dB)"
                        )
                    
                    validation_results["quality_metrics"].append(quality_metrics)
                    validation_results["valid_files"].append(audio_file)
                else:
                    validation_results["invalid_files"].append({
                        "file": audio_file,
                        "issues": issues
                    })
                
            except Exception as e:
                validation_results["invalid_files"].append({
                    "file": audio_file,
                    "issues": [f"Loading error: {str(e)}"]
                })
        
        # Compute dataset statistics
        validation_results["statistics"] = self._compute_dataset_statistics(
            validation_results["quality_metrics"]
        )
        
        return validation_results
    
    def _compute_dataset_statistics(self, quality_metrics: List[Dict]) -> Dict:
        """Compute aggregate dataset statistics"""
        if not quality_metrics:
            return {}
        
        stats = {}
        
        # Aggregate each metric
        for metric_name in quality_metrics[0].keys():
            values = [m[metric_name] for m in quality_metrics if not np.isnan(m[metric_name])]
            if values:
                stats[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values)
                }
        
        return stats

def create_fairseq_manifest(audio_files: List[str], 
                          output_path: str,
                          root_dir: str) -> None:
    """
    Create fairseq-compatible manifest file
    """
    with open(output_path, 'w') as f:
        f.write(f"{root_dir}\n")
        
        for audio_file in audio_files:
            try:
                # Get frame count
                audio, sr = sf.read(audio_file)
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                
                frame_count = len(audio)
                rel_path = os.path.relpath(audio_file, root_dir)
                
                f.write(f"{rel_path}\t{frame_count}\n")
                
            except Exception as e:
                print(f"Warning: Could not process {audio_file}: {e}")

# Convenience functions
def preprocess_audio_file(input_path: str, 
                         output_path: str,
                         target_sr: int = 16000,
                         **kwargs) -> Dict:
    """
    Preprocess single audio file and save result
    """
    preprocessor = AudioPreprocessor(target_sr=target_sr, **kwargs)
    audio, metadata = preprocessor.load_and_preprocess(input_path)
    
    # Save processed audio
    sf.write(output_path, audio, target_sr)
    
    return metadata

def analyze_audio_quality(audio_path: str, sr: int = 16000) -> Dict:
    """
    Analyze quality of single audio file
    """
    analyzer = SpeechQualityAnalyzer(sr=sr)
    audio, _ = sf.read(audio_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    return analyzer.analyze_speech_quality(audio)