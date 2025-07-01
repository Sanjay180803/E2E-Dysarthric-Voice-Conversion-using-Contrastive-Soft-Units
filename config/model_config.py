"""
Model configuration for Dysarthric Voice Conversion
Based on the paper: End-to-End Dysarthric Voice Conversion for Low-Resource Languages Using Contrastive Soft Units
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class FairseqUnitsConfig:
    """Configuration for Fairseq unit extraction"""
    # Using fairseq speech2unit pipeline with mHuBERT-147
    model_name: str = "mhubert-147"
    feature_type: str = "hubert"  # for fairseq
    kmeans_clusters: int = 1000
    layer_for_units: int = 11  # 11th layer for unit extraction
    
    # Fairseq paths (will be set during setup)
    acoustic_model_path: str = ""  # path to downloaded mHuBERT-147
    kmeans_model_path: str = ""    # path to trained K-means model
    
@dataclass
class SoftEncoderConfig:
    """Configuration for Soft Content Encoder"""
    # Input: discrete units from fairseq (integer indices)
    # Output: soft unit distributions
    input_dim: int = 1000  # Number of discrete units (K-means clusters)
    output_dim: int = 1000  # Same as input for soft units
    embedding_dim: int = 256  # Embedding dimension for discrete units
    hidden_dim: int = 512
    dropout: float = 0.1
    
@dataclass
class S2UTConfig:
    """Configuration for Speech-to-Unit Translation model"""
    # Encoder config
    encoder_layers: int = 12
    encoder_attention_heads: int = 4
    encoder_embed_dim: int = 256
    encoder_ffn_embed_dim: int = 1024
    encoder_normalize_before: bool = True
    encoder_learned_pos: bool = True
    
    # Decoder config  
    decoder_layers: int = 6
    decoder_attention_heads: int = 8
    decoder_embed_dim: int = 256
    decoder_ffn_embed_dim: int = 1024
    decoder_normalize_before: bool = True
    decoder_learned_pos: bool = True
    
    # Input/Output config
    input_feat_per_channel: int = 80  # Mel filterbank features
    input_channels: int = 1
    output_dim: int = 1000  # Soft unit dimension
    
    # Training config
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    activation_fn: str = "relu"
    
    # Contrastive loss
    contrastive_weight: float = 0.1
    temperature: float = 0.07
    
    # Conv1d subsampling
    conv_kernel_sizes: List[int] = None
    conv_channels: int = 256
    
    def __post_init__(self):
        if self.conv_kernel_sizes is None:
            self.conv_kernel_sizes = [5, 5]

@dataclass
class HiFiGANSoftConfig:
    """Configuration for HiFi-GAN Soft Unit Vocoder"""
    # Generator config
    soft_unit_dim: int = 1000
    embedding_dim: int = 256
    upsample_rates: List[int] = None
    upsample_kernel_sizes: List[int] = None
    upsample_initial_channel: int = 512
    resblock_kernel_sizes: List[int] = None
    resblock_dilation_sizes: List[List[int]] = None
    
    # Discriminator config
    periods: List[int] = None
    
    # Loss weights
    lambda_feat: float = 2.0
    lambda_mel: float = 45.0
    lambda_adv: float = 1.0
    
    # Audio config
    sampling_rate: int = 16000
    hop_size: int = 256
    win_size: int = 1024
    n_mel_channels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0
    
    def __post_init__(self):
        if self.upsample_rates is None:
            self.upsample_rates = [8, 8, 2, 2]
        if self.upsample_kernel_sizes is None:
            self.upsample_kernel_sizes = [16, 16, 4, 4]
        if self.resblock_kernel_sizes is None:
            self.resblock_kernel_sizes = [3, 7, 11]
        if self.resblock_dilation_sizes is None:
            self.resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        if self.periods is None:
            self.periods = [2, 3, 5, 7, 11]

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sampling_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mel_channels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0
    center: bool = False
    
    # Normalization
    normalize: bool = True
    mean: float = 0.0
    std: float = 1.0

@dataclass
class ModelConfig:
    """Main model configuration"""
    fairseq_units: FairseqUnitsConfig = FairseqUnitsConfig()
    soft_encoder: SoftEncoderConfig = SoftEncoderConfig()
    s2ut: S2UTConfig = S2UTConfig()
    hifigan: HiFiGANSoftConfig = HiFiGANSoftConfig()
    audio: AudioConfig = AudioConfig()
    
    # Paths
    data_dir: str = "data_processed"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Device
    device: str = "cuda"
    num_workers: int = 4
    
    def create_dirs(self):
        """Create necessary directories"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create subdirectories for each model
        for model_name in ["soft_encoder", "s2ut", "hifigan"]:
            os.makedirs(os.path.join(self.checkpoint_dir, model_name), exist_ok=True)