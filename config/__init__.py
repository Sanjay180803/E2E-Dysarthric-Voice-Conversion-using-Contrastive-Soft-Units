"""
Configuration module for dysarthric voice conversion
"""

from .model_config import ModelConfig, FairseqUnitsConfig, SoftEncoderConfig, S2UTConfig, HiFiGANSoftConfig, AudioConfig
from .training_config import TrainingConfig, SoftEncoderTrainingConfig, S2UTTrainingConfig, HiFiGANTrainingConfig, DataConfig

__all__ = [
    "ModelConfig",
    "FairseqUnitsConfig", 
    "SoftEncoderConfig",
    "S2UTConfig",
    "HiFiGANSoftConfig",
    "AudioConfig",
    "TrainingConfig",
    "SoftEncoderTrainingConfig",
    "S2UTTrainingConfig", 
    "HiFiGANTrainingConfig",
    "DataConfig"
]