"""
Enhanced configuration with novel parameters for dysarthric voice conversion
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch

@dataclass 
class EnhancedS2UTConfig:
    """Enhanced S2UT configuration with innovative features"""
    
    # Base model architecture
    encoder_layers: int = 12
    decoder_layers: int = 6
    hidden_dim: int = 768
    num_heads: int = 12
    dropout: float = 0.1
    
    # Novel dysarthria-specific parameters
    severity_aware_contrastive: bool = True
    prosody_consistency_weight: float = 0.1
    disfluency_detection_weight: float = 0.05
    articulatory_feature_weight: float = 0.08
    
    # Adaptive learning parameters
    severity_adaptive_margins: Dict[str, float] = field(default_factory=lambda: {
        'mild': 0.05,
        'moderate': 0.1, 
        'severe': 0.15
    })
    cross_severity_regularization: float = 0.02
    temporal_consistency_weight: float = 0.03
    
    # Contrastive learning enhancements
    temperature_scheduling: bool = True
    initial_temperature: float = 0.07
    final_temperature: float = 0.01
    temperature_decay_steps: int = 10000
    
    # Multi-scale feature extraction
    use_multi_scale_features: bool = True
    feature_scales: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    scale_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.2, 0.1])
    
    # Enhanced evaluation
    compute_dii: bool = True
    cross_severity_evaluation: bool = True
    perceptual_evaluation: bool = True
    
    # Training dynamics
    gradient_clipping: float = 1.0
    warmup_steps: int = 4000
    label_smoothing: float = 0.1
    
    # Data augmentation
    spec_augment: bool = True
    time_masking: bool = True
    freq_masking: bool = True
    noise_injection_prob: float = 0.1

@dataclass
class EnhancedTrainingConfig:
    """Enhanced training configuration"""
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 200
    patience: int = 20
    
    # Optimizer settings
    optimizer_type: str = "adamw"
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    
    # Learning rate scheduling
    lr_scheduler: str = "cosine_with_warmup"
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.01
    
    # Loss function weights
    reconstruction_weight: float = 1.0
    contrastive_weight: float = 0.5
    severity_classification_weight: float = 0.3
    prosody_weight: float = 0.2
    
    # Validation and checkpointing
    validation_frequency: int = 1000
    checkpoint_frequency: int = 5000
    save_top_k: int = 3
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

@dataclass
class EnhancedDataConfig:
    """Enhanced data configuration"""
    
    # Audio processing
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    
    # Sequence lengths
    max_source_length: int = 1024
    max_target_length: int = 1024
    
    # Data augmentation
    augmentation_prob: float = 0.3
    noise_std: float = 0.01
    pitch_shift_range: Tuple[float, float] = (-2.0, 2.0)
    time_stretch_range: Tuple[float, float] = (0.8, 1.2)
    
    # Severity detection
    severity_detection_model: str = "wav2vec2"
    severity_threshold_mild: float = 0.3
    severity_threshold_severe: float = 0.7
