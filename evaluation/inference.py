"""
End-to-end inference pipeline for dysarthric voice conversion
Using fairseq-extracted units approach
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import soundfile as sf
import librosa
from typing import Dict, Optional, Union, List
import json
import time

from config.model_config import ModelConfig
from models.soft_encoder import SoftContentEncoder
from models.s2ut_model import S2UTModel
from models.hifigan_soft import SoftHiFiGANGenerator
from data.audio_utils import AudioProcessor

class DysarthricVoiceConverter:
    """
    Complete end-to-end dysarthric voice conversion pipeline
    
    Pipeline: Dysarthric Speech -> S2UT -> Healthy Soft Units -> HiFi-GAN -> Healthy Speech
    
    Note: This uses pre-extracted units from fairseq speech2unit pipeline
    """
    
    def __init__(self, 
                 config: ModelConfig,
                 model_checkpoints: Dict[str, str],
                 device: str = "cuda"):
        """
        Initialize the voice conversion pipeline
        
        Args:
            config: Model configuration
            model_checkpoints: Dictionary with paths to model checkpoints
            device: Device to run inference on
        """
        self.config = config
        self.device = torch.device(device)
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(**config.audio.__dict__)
        
        # Load models
        self.s2ut_model = self._load_s2ut_model(model_checkpoints["s2ut"])
        self.hifigan_generator = self._load_hifigan_model(model_checkpoints["hifigan"])
        
        # Set models to evaluation mode
        self.s2ut_model.eval()
        self.hifigan_generator.eval()
        
        print("Dysarthric Voice Conversion Pipeline initialized successfully!")
        print("Note: This pipeline uses pre-extracted units from fairseq speech2unit")
    
    def _load_s2ut_model(self, checkpoint_path: str) -> S2UTModel:
        """Load S2UT model from checkpoint"""
        print(f"Loading S2UT model from {checkpoint_path}")
        
        model = S2UTModel(self.config.s2ut)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        
        return model
    
    def _load_hifigan_model(self, checkpoint_path: str) -> SoftHiFiGANGenerator:
        """Load HiFi-GAN generator from checkpoint"""
        print(f"Loading HiFi-GAN model from {checkpoint_path}")
        
        generator = SoftHiFiGANGenerator(self.config.hifigan)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if "generator" in checkpoint:
            generator.load_state_dict(checkpoint["generator"])
        else:
            generator.load_state_dict(checkpoint["model_state_dict"])
        
        generator.to(self.device)
        
        # Remove weight norm for inference
        generator.remove_weight_norm()
        
        return generator
    
    def convert_speech(self, 
                      dysarthric_audio_path: Union[str, Path],
                      output_path: Optional[Union[str, Path]] = None,
                      return_intermediate: bool = False) -> Dict:
        """
        Convert dysarthric speech to healthy speech
        
        Args:
            dysarthric_audio_path: Path to dysarthric audio file
            output_path: Path to save converted audio
            return_intermediate: Whether to return intermediate outputs
            
        Returns:
            Dictionary containing conversion results
        """
        start_time = time.time()
        
        # Load and preprocess dysarthric audio
        dysarthric_audio, mel_spec = self.audio_processor.preprocess_audio(
            dysarthric_audio_path,
            max_length=10.0  # Limit to 10 seconds for memory
        )
        
        # Convert to batch format
        mel_spec = mel_spec.unsqueeze(0).to(self.device)  # [1, n_mels, seq_len]
        
        results = {"processing_time": {}}
        
        with torch.no_grad():
            # S2UT - Convert dysarthric mel to healthy soft units
            s2ut_start = time.time()
            s2ut_outputs = self.s2ut_model(
                dysarthric_mel=mel_spec,
                target_soft_units=None,  # Inference mode
                src_mask=None,
                tgt_mask=None
            )
            
            healthy_soft_units = s2ut_outputs["predicted_soft_units"]  # [1, seq_len, soft_unit_dim]
            results["processing_time"]["s2ut"] = time.time() - s2ut_start
            
            # HiFi-GAN - Convert soft units to audio
            hifigan_start = time.time()
            converted_audio = self.hifigan_generator(healthy_soft_units)  # [1, 1, audio_len]
            converted_audio = converted_audio.squeeze().cpu().numpy()  # [audio_len]
            results["processing_time"]["hifigan"] = time.time() - hifigan_start
        
        # Post-process audio
        converted_audio = self._post_process_audio(converted_audio)
        
        # Save converted audio if output path provided
        if output_path:
            self.audio_processor.save_audio(
                torch.from_numpy(converted_audio),
                output_path,
                normalize=True
            )
            results["output_path"] = str(output_path)
        
        # Store results
        results["converted_audio"] = converted_audio
        results["processing_time"]["total"] = time.time() - start_time
        results["original_duration"] = len(dysarthric_audio) / self.config.audio.sampling_rate
        results["converted_duration"] = len(converted_audio) / self.config.audio.sampling_rate
        
        # Store intermediate outputs if requested
        if return_intermediate:
            results["intermediate"] = {
                "dysarthric_mel": mel_spec.cpu().numpy(),
                "healthy_soft_units": healthy_soft_units.cpu().numpy(),
                "original_audio": dysarthric_audio.numpy()
            }
        
        return results
    
    def _post_process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Post-process generated audio"""
        # Trim silence from beginning and end
        audio = librosa.effects.trim(audio, top_db=20)[0]
        
        # Normalize amplitude
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
        
        return audio
    
    def convert_batch(self,
                     dysarthric_audio_paths: List[Union[str, Path]],
                     output_dir: Union[str, Path],
                     batch_size: int = 4) -> List[Dict]:
        """
        Convert multiple dysarthric audio files in batches
        
        Args:
            dysarthric_audio_paths: List of dysarthric audio file paths
            output_dir: Directory to save converted audio files
            batch_size: Batch size for processing
            
        Returns:
            List of conversion results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        
        for i in range(0, len(dysarthric_audio_paths), batch_size):
            batch_paths = dysarthric_audio_paths[i:i + batch_size]
            
            for audio_path in batch_paths:
                audio_path = Path(audio_path)
                output_path = output_dir / f"converted_{audio_path.stem}.wav"
                
                print(f"Converting {audio_path.name}...")
                
                try:
                    result = self.convert_speech(
                        dysarthric_audio_path=audio_path,
                        output_path=output_path,
                        return_intermediate=False
                    )
                    
                    result["input_file"] = str(audio_path)
                    result["success"] = True
                    
                except Exception as e:
                    print(f"Error converting {audio_path.name}: {e}")
                    result = {
                        "input_file": str(audio_path),
                        "success": False,
                        "error": str(e)
                    }
                
                all_results.append(result)
        
        return all_results

def load_model_checkpoints(checkpoint_dir: str) -> Dict[str, str]:
    """
    Load model checkpoint paths from checkpoint directory
    
    Args:
        checkpoint_dir: Directory containing model checkpoints
        
    Returns:
        Dictionary mapping model names to checkpoint paths
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    checkpoints = {}
    
    # Look for latest checkpoints
    for model_name in ["soft_encoder", "s2ut", "hifigan"]:
        model_dir = checkpoint_dir / model_name
        
        if model_dir.exists():
            # Look for latest checkpoint
            latest_checkpoint = model_dir / f"{model_name}_latest.pt"
            
            if latest_checkpoint.exists():
                checkpoints[model_name] = str(latest_checkpoint)
            else:
                # Look for best checkpoint
                best_checkpoints = list(model_dir.glob(f"{model_name}_best_*.pt"))
                if best_checkpoints:
                    # Get the most recent best checkpoint
                    best_checkpoint = max(best_checkpoints, key=lambda x: x.stat().st_mtime)
                    checkpoints[model_name] = str(best_checkpoint)
    
    return checkpoints

def create_inference_pipeline(checkpoint_dir: str, 
                            config_path: Optional[str] = None,
                            device: str = "cuda") -> DysarthricVoiceConverter:
    """
    Create inference pipeline from saved checkpoints
    
    Args:
        checkpoint_dir: Directory containing model checkpoints
        config_path: Path to config file (optional)
        device: Device to run inference on
        
    Returns:
        Initialized voice conversion pipeline
    """
    # Load configuration
    if config_path:
        from training.trainer_utils import load_config
        config_dict = load_config(config_path)
        # Convert dict back to ModelConfig (simplified)
        config = ModelConfig()
    else:
        config = ModelConfig()
    
    # Load model checkpoints
    model_checkpoints = load_model_checkpoints(checkpoint_dir)
    
    # Check required checkpoints
    required_checkpoints = ["s2ut", "hifigan"]
    
    missing_checkpoints = [name for name in required_checkpoints if name not in model_checkpoints]
    if missing_checkpoints:
        raise FileNotFoundError(f"Missing checkpoints: {missing_checkpoints}")
    
    # Create pipeline
    pipeline = DysarthricVoiceConverter(
        config=config,
        model_checkpoints=model_checkpoints,
        device=device
    )
    
    return pipeline