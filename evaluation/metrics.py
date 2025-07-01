"""
Evaluation metrics for dysarthric voice conversion
Based on the paper's evaluation methodology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import soundfile as sf
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import pesq
from sentence_transformers import SentenceTransformer

try:
    import pyworld as pw
except ImportError:
    pw = None
    print("Warning: pyworld not available. Some metrics may not work.")

class IntelligibilityMetrics:
    """
    Intelligibility evaluation metrics following the paper:
    - Mean Opinion Score (MOS)
    - Degraded MOS (DMOS) 
    - BERT-based similarity score
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
        # Initialize BERT model for semantic similarity
        try:
            self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.bert_model.to(device)
        except:
            print("Warning: BERT model not available for semantic similarity")
            self.bert_model = None
    
    def compute_bert_similarity(self, 
                               original_text: List[str],
                               converted_text: List[str]) -> float:
        """
        Compute BERT-based semantic similarity between original and converted speech
        Note: Requires ASR transcriptions of both original and converted speech
        """
        if self.bert_model is None:
            return 0.0
        
        if len(original_text) != len(converted_text):
            raise ValueError("Text lists must have same length")
        
        similarities = []
        
        for orig, conv in zip(original_text, converted_text):
            # Get embeddings
            orig_emb = self.bert_model.encode([orig], convert_to_tensor=True)
            conv_emb = self.bert_model.encode([conv], convert_to_tensor=True)
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(orig_emb, conv_emb, dim=-1)
            similarities.append(similarity.item())
        
        return np.mean(similarities)
    
    def compute_word_error_rate(self, 
                               reference: List[str],
                               hypothesis: List[str]) -> float:
        """
        Compute Word Error Rate (WER) between reference and hypothesis
        """
        total_words = 0
        total_errors = 0
        
        for ref, hyp in zip(reference, hypothesis):
            ref_words = ref.lower().split()
            hyp_words = hyp.lower().split()
            
            # Simple edit distance computation
            errors = self._edit_distance(ref_words, hyp_words)
            
            total_errors += errors
            total_words += len(ref_words)
        
        return total_errors / total_words if total_words > 0 else 0.0
    
    def _edit_distance(self, ref: List[str], hyp: List[str]) -> int:
        """Compute edit distance between two word sequences"""
        m, n = len(ref), len(hyp)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref[i-1] == hyp[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )
        
        return dp[m][n]

class AudioQualityMetrics:
    """
    Audio quality evaluation metrics
    """
    
    def __init__(self, sampling_rate: int = 16000):
        self.sampling_rate = sampling_rate
    
    def compute_pesq(self, 
                    clean_audio: np.ndarray,
                    enhanced_audio: np.ndarray) -> float:
        """
        Compute PESQ (Perceptual Evaluation of Speech Quality)
        """
        try:
            # Ensure same length
            min_len = min(len(clean_audio), len(enhanced_audio))
            clean_audio = clean_audio[:min_len]
            enhanced_audio = enhanced_audio[:min_len]
            
            # Compute PESQ
            pesq_score = pesq.pesq(
                self.sampling_rate,
                clean_audio,
                enhanced_audio,
                'wb'  # wideband
            )
            
            return pesq_score
        except Exception as e:
            print(f"PESQ computation failed: {e}")
            return 0.0
    
    def compute_stoi(self,
                    clean_audio: np.ndarray,
                    enhanced_audio: np.ndarray) -> float:
        """
        Compute STOI (Short-Time Objective Intelligibility)
        Simplified version - in practice, use pystoi library
        """
        # This is a simplified version
        # For actual STOI, install and use pystoi library
        
        # Compute correlation as proxy
        min_len = min(len(clean_audio), len(enhanced_audio))
        clean_audio = clean_audio[:min_len]
        enhanced_audio = enhanced_audio[:min_len]
        
        correlation, _ = pearsonr(clean_audio, enhanced_audio)
        return max(0, correlation)  # STOI is between 0 and 1
    
    def compute_spectral_convergence(self,
                                   clean_audio: np.ndarray,
                                   enhanced_audio: np.ndarray) -> float:
        """
        Compute spectral convergence
        """
        # Compute spectrograms
        clean_spec = np.abs(librosa.stft(clean_audio))
        enhanced_spec = np.abs(librosa.stft(enhanced_audio))
        
        # Ensure same shape
        min_frames = min(clean_spec.shape[1], enhanced_spec.shape[1])
        clean_spec = clean_spec[:, :min_frames]
        enhanced_spec = enhanced_spec[:, :min_frames]
        
        # Compute spectral convergence
        numerator = np.linalg.norm(clean_spec - enhanced_spec, 'fro')
        denominator = np.linalg.norm(clean_spec, 'fro')
        
        return numerator / denominator if denominator > 0 else float('inf')
    
    def compute_mel_cepstral_distortion(self,
                                      clean_audio: np.ndarray,
                                      enhanced_audio: np.ndarray) -> float:
        """
        Compute Mel-Cepstral Distortion (MCD)
        """
        # Extract MFCC features
        clean_mfcc = librosa.feature.mfcc(
            y=clean_audio, 
            sr=self.sampling_rate,
            n_mfcc=13
        )
        
        enhanced_mfcc = librosa.feature.mfcc(
            y=enhanced_audio,
            sr=self.sampling_rate, 
            n_mfcc=13
        )
        
        # Ensure same number of frames
        min_frames = min(clean_mfcc.shape[1], enhanced_mfcc.shape[1])
        clean_mfcc = clean_mfcc[:, :min_frames]
        enhanced_mfcc = enhanced_mfcc[:, :min_frames]
        
        # Compute MCD
        diff = clean_mfcc - enhanced_mfcc
        mcd = np.sqrt(2) * np.mean(np.sqrt(np.sum(diff**2, axis=0)))
        
        return mcd

class ProsodyMetrics:
    """
    Prosody evaluation metrics
    """
    
    def __init__(self, sampling_rate: int = 16000):
        self.sampling_rate = sampling_rate
    
    def extract_f0(self, audio: np.ndarray) -> np.ndarray:
        """Extract fundamental frequency (F0) using pyworld"""
        if pw is None:
            # Fallback to librosa
            f0, _, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sampling_rate
            )
            return f0
        
        # Use pyworld for better F0 extraction
        f0, _ = pw.harvest(
            audio.astype(np.float64),
            self.sampling_rate
        )
        
        return f0
    
    def compute_f0_rmse(self,
                       clean_audio: np.ndarray,
                       enhanced_audio: np.ndarray) -> float:
        """
        Compute F0 RMSE between clean and enhanced audio
        """
        f0_clean = self.extract_f0(clean_audio)
        f0_enhanced = self.extract_f0(enhanced_audio)
        
        # Remove NaN values and align lengths
        f0_clean = f0_clean[~np.isnan(f0_clean)]
        f0_enhanced = f0_enhanced[~np.isnan(f0_enhanced)]
        
        min_len = min(len(f0_clean), len(f0_enhanced))
        if min_len == 0:
            return float('inf')
        
        f0_clean = f0_clean[:min_len]
        f0_enhanced = f0_enhanced[:min_len]
        
        # Compute RMSE
        rmse = np.sqrt(np.mean((f0_clean - f0_enhanced)**2))
        return rmse
    
    def compute_rhythm_similarity(self,
                                clean_audio: np.ndarray,
                                enhanced_audio: np.ndarray) -> float:
        """
        Compute rhythm similarity based on energy patterns
        """
        # Extract energy
        clean_energy = librosa.feature.rms(y=clean_audio, frame_length=2048, hop_length=512)[0]
        enhanced_energy = librosa.feature.rms(y=enhanced_audio, frame_length=2048, hop_length=512)[0]
        
        # Align lengths
        min_len = min(len(clean_energy), len(enhanced_energy))
        clean_energy = clean_energy[:min_len]
        enhanced_energy = enhanced_energy[:min_len]
        
        # Compute correlation
        if min_len > 1:
            correlation, _ = pearsonr(clean_energy, enhanced_energy)
            return max(0, correlation)
        else:
            return 0.0

class ComprehensiveEvaluator:
    """
    Comprehensive evaluator combining all metrics
    """
    
    def __init__(self, sampling_rate: int = 16000, device: str = "cuda"):
        self.sampling_rate = sampling_rate
        self.device = device
        
        self.intelligibility_metrics = IntelligibilityMetrics(device)
        self.audio_quality_metrics = AudioQualityMetrics(sampling_rate)
        self.prosody_metrics = ProsodyMetrics(sampling_rate)
    
    def evaluate_pair(self,
                     original_audio: np.ndarray,
                     converted_audio: np.ndarray,
                     original_text: Optional[str] = None,
                     converted_text: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate a pair of original and converted audio
        """
        results = {}
        
        # Audio quality metrics
        results['pesq'] = self.audio_quality_metrics.compute_pesq(
            original_audio, converted_audio
        )
        
        results['stoi'] = self.audio_quality_metrics.compute_stoi(
            original_audio, converted_audio
        )
        
        results['spectral_convergence'] = self.audio_quality_metrics.compute_spectral_convergence(
            original_audio, converted_audio
        )
        
        results['mcd'] = self.audio_quality_metrics.compute_mel_cepstral_distortion(
            original_audio, converted_audio
        )
        
        # Prosody metrics
        results['f0_rmse'] = self.prosody_metrics.compute_f0_rmse(
            original_audio, converted_audio
        )
        
        results['rhythm_similarity'] = self.prosody_metrics.compute_rhythm_similarity(
            original_audio, converted_audio
        )
        
        # Intelligibility metrics (if text available)
        if original_text and converted_text:
            results['bert_similarity'] = self.intelligibility_metrics.compute_bert_similarity(
                [original_text], [converted_text]
            )
            
            results['wer'] = self.intelligibility_metrics.compute_word_error_rate(
                [original_text], [converted_text]
            )
        
        return results
    
    def evaluate_dataset(self,
                        original_audio_files: List[str],
                        converted_audio_files: List[str],
                        original_texts: Optional[List[str]] = None,
                        converted_texts: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate entire dataset
        """
        all_results = []
        
        for i, (orig_file, conv_file) in enumerate(zip(original_audio_files, converted_audio_files)):
            # Load audio
            orig_audio, _ = librosa.load(orig_file, sr=self.sampling_rate)
            conv_audio, _ = librosa.load(conv_file, sr=self.sampling_rate)
            
            # Get texts if available
            orig_text = original_texts[i] if original_texts else None
            conv_text = converted_texts[i] if converted_texts else None
            
            # Evaluate pair
            result = self.evaluate_pair(
                orig_audio, conv_audio, orig_text, conv_text
            )
            
            all_results.append(result)
        
        # Aggregate results
        aggregated = {}
        for key in all_results[0].keys():
            values = [r[key] for r in all_results if not np.isnan(r[key]) and not np.isinf(r[key])]
            if values:
                aggregated[f"{key}_mean"] = np.mean(values)
                aggregated[f"{key}_std"] = np.std(values)
            else:
                aggregated[f"{key}_mean"] = 0.0
                aggregated[f"{key}_std"] = 0.0
        
        return aggregated

