"""
Training configuration for Dysarthric Voice Conversion models
"""

from dataclasses import dataclass
from typing import Dict, Optional

"""
Training configuration for Dysarthric Voice Conversion models
"""

from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class SoftEncoderTrainingConfig:
    """Training configuration for Soft Content Encoder"""
    # Optimizer
    optimizer: str = "adam"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    
    # Training
    epochs: int = 25
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Validation
    eval_steps: int = 200
    save_steps: int = 500
    logging_steps: int = 50
    
    # Loss
    label_smoothing: float = 0.0

@dataclass
class S2UTTrainingConfig:
    """Training configuration for S2UT model"""
    # Optimizer
    optimizer: str = "adam"
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1e-9
    
    # Scheduler
    lr_scheduler: str = "inverse_sqrt"
    warmup_steps: int = 10000
    
    # Training
    epochs: int = 235
    max_steps: int = 36000
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Validation
    eval_steps: int = 1000
    save_steps: int = 2000
    logging_steps: int = 100
    
    # Loss weights
    ce_weight: float = 1.0
    contrastive_weight: float = 0.1
    
    # SpecAugment
    freq_mask_N: int = 2
    freq_mask_F: int = 10
    time_mask_N: int = 2
    time_mask_T: int = 50
    time_mask_p: float = 0.2
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

@dataclass
class HiFiGANTrainingConfig:
    """Training configuration for HiFi-GAN Soft Unit Vocoder"""
    # Optimizer (separate for generator and discriminator)
    gen_optimizer: str = "adam"
    disc_optimizer: str = "adam"
    gen_learning_rate: float = 2e-4
    disc_learning_rate: float = 2e-4
    gen_weight_decay: float = 0.0
    disc_weight_decay: float = 0.0
    adam_beta1: float = 0.8
    adam_beta2: float = 0.99
    
    # Scheduler
    lr_scheduler: str = "exponential"
    lr_decay: float = 0.999
    
    # Training
    epochs: int = 3000
    max_steps: int = 500000
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    gen_max_grad_norm: float = 1.0
    disc_max_grad_norm: float = 1.0
    
    # Training schedule
    gen_train_start_step: int = 0
    disc_train_start_step: int = 2500
    
    # Validation
    eval_steps: int = 5000
    save_steps: int = 10000
    logging_steps: int = 100
    
    # Loss weights
    lambda_adv: float = 1.0
    lambda_feat: float = 2.0
    lambda_mel: float = 45.0
    
    # Audio validation
    generate_audio_every: int = 5000
    num_audio_samples: int = 5

@dataclass
class DataConfig:
    """Data configuration for training"""
    # Dataset
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Audio processing
    max_audio_length: float = 10.0  # seconds
    min_audio_length: float = 0.5   # seconds
    
    # Filtering
    filter_by_duration: bool = True
    normalize_audio: bool = True
    
    # Augmentation
    use_spec_augment: bool = True
    use_audio_augment: bool = False
    
    # Loading
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

@dataclass
class TrainingConfig:
    """Main training configuration"""
    soft_encoder: SoftEncoderTrainingConfig = SoftEncoderTrainingConfig()
    s2ut: S2UTTrainingConfig = S2UTTrainingConfig()
    hifigan: HiFiGANTrainingConfig = HiFiGANTrainingConfig()
    data: DataConfig = DataConfig()
    
    # General settings
    seed: int = 42
    device: str = "cuda"
    mixed_precision: bool = True
    compile_model: bool = False
    
    # Logging
    project_name: str = "dysarthric_voice_conversion"
    experiment_name: Optional[str] = None
    log_to_wandb: bool = True
    log_to_tensorboard: bool = True
    
    # Checkpointing
    save_top_k: int = 3
    monitor_metric: str = "val_loss"
    mode: str = "min"
    
    def get_experiment_name(self, model_name: str) -> str:
        """Generate experiment name"""
        if self.experiment_name:
            return f"{self.experiment_name}_{model_name}"
        return f"dvc_{model_name}"