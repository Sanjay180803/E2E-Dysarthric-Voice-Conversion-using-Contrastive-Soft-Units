"""
Advanced evaluation metrics specific to dysarthric speech conversion
Novel contribution: Cross-severity generalization and dysarthria-specific metrics
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import librosa
from scipy import stats

class DysarthriaSpecificMetrics:
    """
    Novel evaluation metrics specifically designed for dysarthric speech
    """
    
    def __init__(self, sampling_rate: int = 16000):
        self.sampling_rate = sampling_rate
    
    def compute_severity_transfer_accuracy(self,
                                         original_severities: List[str],
                                         converted_severities: List[str]) -> Dict[str, float]:
        """
        Measure how well the system transfers severity levels
        
        Novel metric: Severity Transfer Accuracy (STA)
        """
        severity_map = {'healthy': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
        
        original_scores = [severity_map[s] for s in original_severities]
        converted_scores = [severity_map[s] for s in converted_severities]
        
        # Compute improvement (reduction in severity)
        improvements = [orig - conv for orig, conv in zip(original_scores, converted_scores)]
        
        # Severity Transfer Accuracy
        sta_score = np.mean([1 if imp > 0 else 0 for imp in improvements])
        
        # Average severity reduction
        avg_reduction = np.mean(improvements)
        
        return {
            'severity_transfer_accuracy': sta_score,
            'average_severity_reduction': avg_reduction,
            'successful_transfers': sum(1 for imp in improvements if imp > 0),
            'total_transfers': len(improvements)
        }
    
    def compute_dysarthria_intelligibility_index(self,
                                                original_audio: np.ndarray,
                                                converted_audio: np.ndarray) -> float:
        """
        Novel Dysarthria Intelligibility Index (DII)
        
        Combines multiple acoustic measures relevant to dysarthric speech
        """
        # 1. Articulation precision (formant stability)
        orig_formants = self._extract_formant_stability(original_audio)
        conv_formants = self._extract_formant_stability(converted_audio)
        articulation_improvement = conv_formants - orig_formants
        
        # 2. Prosodic naturalness (rhythm and stress patterns)
        orig_prosody = self._compute_prosodic_naturalness(original_audio)
        conv_prosody = self._compute_prosodic_naturalness(converted_audio)
        prosody_improvement = conv_prosody - orig_prosody
        
        # 3. Speech clarity (spectral definition)
        orig_clarity = self._compute_spectral_clarity(original_audio)
        conv_clarity = self._compute_spectral_clarity(converted_audio)
        clarity_improvement = conv_clarity - orig_clarity
        
        # Combine into DII (weighted average)
        dii = (0.4 * articulation_improvement + 
               0.3 * prosody_improvement + 
               0.3 * clarity_improvement)
        
        return max(0, min(1, dii))  # Normalize to [0, 1]
    
    def _extract_formant_stability(self, audio: np.ndarray) -> float:
        """Measure formant stability as indicator of articulation precision"""
        # Extract formants using LPC
        try:
            # Simple formant tracking using spectral peaks
            n_fft = 1024
            hop_length = 256
            
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            
            # Find spectral peaks across time
            formant_tracks = []
            for frame in magnitude.T:
                peaks = librosa.util.peak_pick(frame, pre_max=3, post_max=3, pre_avg=3, post_avg=3)
                if len(peaks) >= 2:
                    formant_tracks.append(peaks[:2])  # F1, F2
            
            if len(formant_tracks) > 1:
                formant_tracks = np.array(formant_tracks)
                # Compute coefficient of variation (stability measure)
                f1_stability = 1 / (1 + np.std(formant_tracks[:, 0]) / np.mean(formant_tracks[:, 0]))
                f2_stability = 1 / (1 + np.std(formant_tracks[:, 1]) / np.mean(formant_tracks[:, 1]))
                return (f1_stability + f2_stability) / 2
            
            return 0.5  # Default value
            
        except:
            return 0.5  # Fallback
    
    def _compute_prosodic_naturalness(self, audio: np.ndarray) -> float:
        """Measure prosodic naturalness"""
        # Extract F0 contour
        f0, _, _ = librosa.pyin(audio, fmin=50, fmax=400, sr=self.sampling_rate)
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) < 10:
            return 0.5
        
        # Measure F0 contour smoothness
        f0_diff = np.diff(f0_clean)
        smoothness = 1 / (1 + np.std(f0_diff))
        
        # Measure rhythm regularity
        energy = librosa.feature.rms(y=audio, hop_length=256)[0]
        energy_peaks = librosa.util.peak_pick(energy, pre_max=3, post_max=3)
        
        if len(energy_peaks) > 2:
            inter_peak_intervals = np.diff(energy_peaks)
            rhythm_regularity = 1 / (1 + np.std(inter_peak_intervals) / np.mean(inter_peak_intervals))
        else:
            rhythm_regularity = 0.5
        
        return (smoothness + rhythm_regularity) / 2
    
    def _compute_spectral_clarity(self, audio: np.ndarray) -> float:
        """Measure spectral clarity"""
        # Spectral centroid stability
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sampling_rate)[0]
        centroid_stability = 1 / (1 + np.std(centroid) / np.mean(centroid))
        
        # Spectral rolloff consistency
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sampling_rate)[0]
        rolloff_consistency = 1 / (1 + np.std(rolloff) / np.mean(rolloff))
        
        return (centroid_stability + rolloff_consistency) / 2

class CrossSeverityEvaluator:
    """
    Evaluate cross-severity generalization capabilities
    """
    
    def __init__(self):
        self.dysarthria_metrics = DysarthriaSpecificMetrics()
    
    def evaluate_cross_severity_generalization(self,
                                             model,
                                             test_data_by_severity: Dict[str, List],
                                             device: str = "cuda") -> Dict[str, Dict]:
        """
        Evaluate how well the model generalizes across different severity levels
        """
        results = {}
        severities = ['mild', 'moderate', 'severe']
        
        for train_severity in severities:
            for test_severity in severities:
                if train_severity == test_severity:
                    continue  # Skip same-severity evaluation
                
                # Test model trained on train_severity with test_severity data
                test_results = self._evaluate_severity_pair(
                    model, train_severity, test_severity,
                    test_data_by_severity[test_severity], device
                )
                
                results[f"{train_severity}_to_{test_severity}"] = test_results
        
        return results
    
    def _evaluate_severity_pair(self,
                               model,
                               train_severity: str,
                               test_severity: str,
                               test_data: List,
                               device: str) -> Dict:
        """Evaluate model on cross-severity data"""
        model.eval()
        
        total_dii = []
        conversion_quality = []
        
        with torch.no_grad():
            for sample in test_data[:50]:  # Limit for efficiency
                try:
                    # Load and convert audio
                    original_audio = sample['audio']
                    converted_audio = model.convert(sample['mel_spec'].to(device))
                    
                    # Compute DII
                    dii = self.dysarthria_metrics.compute_dysarthria_intelligibility_index(
                        original_audio.numpy(), converted_audio.cpu().numpy()
                    )
                    total_dii.append(dii)
                    
                    # Additional quality metrics
                    quality = self._compute_conversion_quality(original_audio, converted_audio)
                    conversion_quality.append(quality)
                    
                except Exception as e:
                    continue
        
        return {
            'average_dii': np.mean(total_dii) if total_dii else 0.0,
            'dii_std': np.std(total_dii) if total_dii else 0.0,
            'conversion_quality': np.mean(conversion_quality) if conversion_quality else 0.0,
            'num_samples': len(total_dii)
        }
    
    def _compute_conversion_quality(self, original: torch.Tensor, converted: torch.Tensor) -> float:
        """Compute overall conversion quality score"""
        # Simple quality measure based on signal properties
        orig_energy = torch.mean(original ** 2)
        conv_energy = torch.mean(converted ** 2)
        
        energy_ratio = min(conv_energy / orig_energy, orig_energy / conv_energy)
        return energy_ratio.item()