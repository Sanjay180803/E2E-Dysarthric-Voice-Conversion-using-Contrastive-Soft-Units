"""
Complete training pipeline for Dysarthric Voice Conversion
Orchestrates all training phases from unit extraction to final model training
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from scripts.extract_units_fairseq import main as extract_units_main
from training.train_soft_encoder import main as train_soft_encoder_main
from training.train_s2ut import main as train_s2ut_main
from training.train_hifigan import main as train_hifigan_main

class TrainingPipeline:
    """
    Complete training pipeline for dysarthric voice conversion
    
    Pipeline stages:
    1. Data validation and preprocessing check
    2. Unit extraction using fairseq speech2unit
    3. Soft encoder training 
    4. S2UT model training
    5. HiFi-GAN vocoder training
    6. Final validation and cleanup
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 training_config_path: Optional[str] = None,
                 resume_from_stage: Optional[str] = None,
                 stages_to_run: Optional[List[str]] = None):
        """
        Initialize training pipeline
        
        Args:
            config_path: Path to model config file
            training_config_path: Path to training config file  
            resume_from_stage: Stage to resume from
            stages_to_run: Specific stages to run
        """
        # Load configurations
        if config_path:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.config = ModelConfig(**config_dict)
        else:
            self.config = ModelConfig()
            
        if training_config_path:
            with open(training_config_path, 'r') as f:
                training_config_dict = json.load(f)
            self.training_config = TrainingConfig(**training_config_dict)
        else:
            self.training_config = TrainingConfig()
        
        # Create necessary directories
        self.config.create_dirs()
        
        # Define pipeline stages
        self.all_stages = [
            "validate_data",
            "extract_units", 
            "train_soft_encoder",
            "train_s2ut",
            "train_hifigan",
            "final_validation"
        ]
        
        # Determine stages to run
        if stages_to_run:
            self.stages_to_run = stages_to_run
        elif resume_from_stage:
            start_idx = self.all_stages.index(resume_from_stage)
            self.stages_to_run = self.all_stages[start_idx:]
        else:
            self.stages_to_run = self.all_stages
        
        # Pipeline state
        self.pipeline_state = self._load_pipeline_state()
        self.start_time = time.time()
        
        print(f"Training Pipeline initialized")
        print(f"Stages to run: {self.stages_to_run}")
        print(f"Data directory: {self.config.data_dir}")
        print(f"Checkpoint directory: {self.config.checkpoint_dir}")
    
    def _load_pipeline_state(self) -> Dict:
        """Load pipeline state from file"""
        state_file = Path(self.config.log_dir) / "pipeline_state.json"
        
        if state_file.exists():
            with open(state_file, 'r') as f:
                return json.load(f)
        
        return {
            "completed_stages": [],
            "failed_stages": [],
            "stage_durations": {},
            "total_start_time": None
        }
    
    def _save_pipeline_state(self):
        """Save pipeline state to file"""
        state_file = Path(self.config.log_dir) / "pipeline_state.json"
        
        with open(state_file, 'w') as f:
            json.dump(self.pipeline_state, f, indent=2)
    
    def _check_stage_completion(self, stage: str) -> bool:
        """Check if a stage has been completed successfully"""
        if stage == "validate_data":
            return (Path(self.config.data_dir) / "metadata" / "dataset_statistics.json").exists()
        
        elif stage == "extract_units":
            return (Path(self.config.data_dir) / "units" / "discrete_units.txt").exists()
        
        elif stage == "train_soft_encoder":
            return (Path(self.config.checkpoint_dir) / "soft_encoder" / "soft_encoder_latest.pt").exists()
        
        elif stage == "train_s2ut":
            return (Path(self.config.checkpoint_dir) / "s2ut" / "s2ut_latest.pt").exists()
        
        elif stage == "train_hifigan":
            return (Path(self.config.checkpoint_dir) / "hifigan" / "hifigan_latest.pt").exists()
        
        elif stage == "final_validation":
            return (Path(self.config.log_dir) / "pipeline_completed.json").exists()
        
        return False
    
    def validate_data(self) -> bool:
        """Validate data preparation and preprocessing"""
        print("\n" + "="*50)
        print("STAGE 1: Data Validation")
        print("="*50)
        
        stage_start = time.time()
        
        try:
            # Check if data directories exist
            data_dir = Path(self.config.data_dir)
            
            required_dirs = [
                data_dir / "healthy",
                data_dir / "dysarthric" / "mild",
                data_dir / "dysarthric" / "moderate",
                data_dir / "metadata"
            ]
            
            missing_dirs = [d for d in required_dirs if not d.exists()]
            if missing_dirs:
                print(f"ERROR: Missing data directories: {missing_dirs}")
                print("Please run scripts/prepare_data.py first")
                return False
            
            # Check metadata files
            metadata_file = data_dir / "metadata" / "complete_metadata.json"
            if not metadata_file.exists():
                print(f"ERROR: Metadata file not found: {metadata_file}")
                return False
            
            # Load and validate metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Collect statistics
            stats = {
                "healthy_speakers": set(),
                "dysarthric_speakers": set(),
                "total_utterances": len(metadata),
                "severities": {},
                "total_duration": 0.0
            }
            
            for item in metadata:
                if item["severity"] == "healthy":
                    stats["healthy_speakers"].add(item["speaker_id"])
                else:
                    stats["dysarthric_speakers"].add(item["speaker_id"])
                    
                severity = item["severity"]
                if severity not in stats["severities"]:
                    stats["severities"][severity] = 0
                stats["severities"][severity] += 1
                
                stats["total_duration"] += item.get("duration", 0.0)
            
            # Convert sets to counts
            stats["healthy_speakers"] = len(stats["healthy_speakers"])
            stats["dysarthric_speakers"] = len(stats["dysarthric_speakers"])
            
            # Save statistics
            stats_file = data_dir / "metadata" / "dataset_statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Print summary
            print(f"✓ Data validation completed")
            print(f"  - Healthy speakers: {stats['healthy_speakers']}")
            print(f"  - Dysarthric speakers: {stats['dysarthric_speakers']}")
            print(f"  - Total utterances: {stats['total_utterances']}")
            print(f"  - Total duration: {stats['total_duration']:.1f} hours")
            print(f"  - Severities: {stats['severities']}")
            
            duration = time.time() - stage_start
            self.pipeline_state["stage_durations"]["validate_data"] = duration
            
            return True
            
        except Exception as e:
            print(f"ERROR in data validation: {e}")
            return False
    
    def extract_units(self) -> bool:
        """Extract discrete units using fairseq speech2unit pipeline"""
        print("\n" + "="*50)
        print("STAGE 2: Unit Extraction (Fairseq)")
        print("="*50)
        
        stage_start = time.time()
        
        try:
            # Check if mHuBERT model exists
            mhubert_path = Path("models/pretrained/mhubert_base_vp_en_es_fr_it3.pt")
            if not mhubert_path.exists():
                print(f"ERROR: mHuBERT model not found: {mhubert_path}")
                print("Please download the model using fairseq_setup/download_models.py")
                return False
            
            # Set up arguments for unit extraction
            extract_args = [
                "--data_dir", str(self.config.data_dir),
                "--acoustic_model_path", str(mhubert_path),
                "--output_dir", str(Path(self.config.data_dir) / "units"),
                "--fit_kmeans",
                "--extract_units",
                "--n_clusters", str(self.config.fairseq_units.kmeans_clusters),
                "--layer", str(self.config.fairseq_units.layer_for_units),
                "--severity_filter", "mild", "moderate"
            ]
            
            # Run unit extraction
            print("Running fairseq unit extraction...")
            
            # Simulate extract_units_main call (replace with actual implementation)
            success = self._run_extract_units_fairseq(extract_args)
            
            if success:
                print("✓ Unit extraction completed successfully")
                duration = time.time() - stage_start
                self.pipeline_state["stage_durations"]["extract_units"] = duration
                return True
            else:
                print("ERROR: Unit extraction failed")
                return False
                
        except Exception as e:
            print(f"ERROR in unit extraction: {e}")
            return False
    
    def _run_extract_units_fairseq(self, args: List[str]) -> bool:
        """Run fairseq unit extraction"""
        try:
            # Import and run the extraction script
            from scripts.extract_units_fairseq import main as extract_main
            
            # Mock arguments object
            class Args:
                def __init__(self, arg_list):
                    self.data_dir = arg_list[arg_list.index("--data_dir") + 1]
                    self.acoustic_model_path = arg_list[arg_list.index("--acoustic_model_path") + 1]
                    self.output_dir = arg_list[arg_list.index("--output_dir") + 1]
                    self.fit_kmeans = "--fit_kmeans" in arg_list
                    self.extract_units = "--extract_units" in arg_list
                    self.n_clusters = int(arg_list[arg_list.index("--n_clusters") + 1])
                    self.layer = int(arg_list[arg_list.index("--layer") + 1])
                    severity_idx = arg_list.index("--severity_filter")
                    self.severity_filter = arg_list[severity_idx + 1:severity_idx + 3]
            
            mock_args = Args(args)
            extract_main(mock_args)
            return True
            
        except Exception as e:
            print(f"Unit extraction error: {e}")
            return False
    
    def train_soft_encoder(self) -> bool:
        """Train soft content encoder"""
        print("\n" + "="*50)
        print("STAGE 3: Soft Encoder Training")
        print("="*50)
        
        stage_start = time.time()
        
        try:
            print("Starting soft encoder training...")
            
            # Run training
            success = self._run_soft_encoder_training()
            
            if success:
                print("✓ Soft encoder training completed")
                duration = time.time() - stage_start
                self.pipeline_state["stage_durations"]["train_soft_encoder"] = duration
                return True
            else:
                print("ERROR: Soft encoder training failed")
                return False
                
        except Exception as e:
            print(f"ERROR in soft encoder training: {e}")
            return False
    
    def _run_soft_encoder_training(self) -> bool:
        """Run soft encoder training"""
        try:
            from training.train_soft_encoder import SoftEncoderTrainer
            
            # Create trainer and run
            trainer = SoftEncoderTrainer(self.config, self.training_config)
            trainer.train()
            
            return True
            
        except Exception as e:
            print(f"Soft encoder training error: {e}")
            return False
    
    def train_s2ut(self) -> bool:
        """Train Speech-to-Unit Translation model"""
        print("\n" + "="*50)
        print("STAGE 4: S2UT Training")
        print("="*50)
        
        stage_start = time.time()
        
        try:
            print("Starting S2UT training...")
            print("This is the core model for dysarthric-to-healthy conversion")
            
            # Run training
            success = self._run_s2ut_training()
            
            if success:
                print("✓ S2UT training completed")
                duration = time.time() - stage_start
                self.pipeline_state["stage_durations"]["train_s2ut"] = duration
                return True
            else:
                print("ERROR: S2UT training failed")
                return False
                
        except Exception as e:
            print(f"ERROR in S2UT training: {e}")
            return False
    
    def _run_s2ut_training(self) -> bool:
        """Run S2UT training"""
        try:
            from training.train_s2ut import S2UTTrainer
            
            # Create trainer and run
            trainer = S2UTTrainer(self.config, self.training_config)
            trainer.train()
            
            return True
            
        except Exception as e:
            print(f"S2UT training error: {e}")
            return False
    
    def train_hifigan(self) -> bool:
        """Train HiFi-GAN vocoder"""
        print("\n" + "="*50)
        print("STAGE 5: HiFi-GAN Training")
        print("="*50)
        
        stage_start = time.time()
        
        try:
            print("Starting HiFi-GAN training...")
            print("This will take 3-5 days for full training")
            
            # Run training
            success = self._run_hifigan_training()
            
            if success:
                print("✓ HiFi-GAN training completed")
                duration = time.time() - stage_start
                self.pipeline_state["stage_durations"]["train_hifigan"] = duration
                return True
            else:
                print("ERROR: HiFi-GAN training failed")
                return False
                
        except Exception as e:
            print(f"ERROR in HiFi-GAN training: {e}")
            return False
    
    def _run_hifigan_training(self) -> bool:
        """Run HiFi-GAN training"""
        try:
            from training.train_hifigan import HiFiGANTrainer
            
            # Create trainer and run
            trainer = HiFiGANTrainer(self.config, self.training_config)
            trainer.train()
            
            return True
            
        except Exception as e:
            print(f"HiFi-GAN training error: {e}")
            return False
    
    def final_validation(self) -> bool:
        """Perform final validation and create completion marker"""
        print("\n" + "="*50)
        print("STAGE 6: Final Validation")
        print("="*50)
        
        stage_start = time.time()
        
        try:
            # Check all model checkpoints exist
            required_checkpoints = [
                self.config.checkpoint_dir + "/soft_encoder/soft_encoder_latest.pt",
                self.config.checkpoint_dir + "/s2ut/s2ut_latest.pt", 
                self.config.checkpoint_dir + "/hifigan/hifigan_latest.pt"
            ]
            
            missing_checkpoints = [cp for cp in required_checkpoints if not Path(cp).exists()]
            if missing_checkpoints:
                print(f"ERROR: Missing checkpoints: {missing_checkpoints}")
                return False
            
            # Test inference pipeline
            print("Testing inference pipeline...")
            success = self._test_inference_pipeline()
            
            if not success:
                print("ERROR: Inference pipeline test failed")
                return False
            
            # Create completion marker
            completion_info = {
                "pipeline_completed": True,
                "completion_time": time.time(),
                "total_duration_hours": (time.time() - self.start_time) / 3600,
                "stage_durations": self.pipeline_state["stage_durations"],
                "model_checkpoints": {
                    "soft_encoder": str(Path(self.config.checkpoint_dir) / "soft_encoder" / "soft_encoder_latest.pt"),
                    "s2ut": str(Path(self.config.checkpoint_dir) / "s2ut" / "s2ut_latest.pt"),
                    "hifigan": str(Path(self.config.checkpoint_dir) / "hifigan" / "hifigan_latest.pt")
                }
            }
            
            completion_file = Path(self.config.log_dir) / "pipeline_completed.json"
            with open(completion_file, 'w') as f:
                json.dump(completion_info, f, indent=2)
            
            print("✓ Final validation completed")
            print("✓ Training pipeline completed successfully!")
            
            duration = time.time() - stage_start
            self.pipeline_state["stage_durations"]["final_validation"] = duration
            
            return True
            
        except Exception as e:
            print(f"ERROR in final validation: {e}")
            return False
    
    def _test_inference_pipeline(self) -> bool:
        """Test inference pipeline with a sample"""
        try:
            from evaluation.inference import create_inference_pipeline
            
            # Create inference pipeline
            pipeline = create_inference_pipeline(
                checkpoint_dir=self.config.checkpoint_dir,
                device=self.config.device
            )
            
            # Find a test audio file
            test_audio_dir = Path(self.config.data_dir) / "dysarthric" / "mild"
            test_files = list(test_audio_dir.rglob("*.wav"))
            
            if not test_files:
                print("Warning: No test files found for inference validation")
                return True  # Skip test if no files available
            
            test_file = test_files[0]
            output_path = Path(self.config.log_dir) / "test_conversion.wav"
            
            # Run conversion
            result = pipeline.convert_speech(
                dysarthric_audio_path=test_file,
                output_path=output_path
            )
            
            if result.get("success", True) and output_path.exists():
                print(f"✓ Inference test successful: {output_path}")
                return True
            else:
                print("✗ Inference test failed")
                return False
                
        except Exception as e:
            print(f"Inference test error: {e}")
            return False
    
    def run(self) -> bool:
        """Run the complete training pipeline"""
        print("Starting Dysarthric Voice Conversion Training Pipeline")
        print(f"Stages to run: {self.stages_to_run}")
        print("="*70)
        
        self.pipeline_state["total_start_time"] = self.start_time
        
        # Stage mapping
        stage_functions = {
            "validate_data": self.validate_data,
            "extract_units": self.extract_units,
            "train_soft_encoder": self.train_soft_encoder,
            "train_s2ut": self.train_s2ut,
            "train_hifigan": self.train_hifigan,
            "final_validation": self.final_validation
        }
        
        # Run stages
        for stage in self.stages_to_run:
            print(f"\nRunning stage: {stage}")
            
            # Skip if already completed (unless forced)
            if self._check_stage_completion(stage):
                print(f"✓ Stage {stage} already completed, skipping...")
                continue
            
            # Run stage
            stage_func = stage_functions[stage]
            success = stage_func()
            
            if success:
                self.pipeline_state["completed_stages"].append(stage)
                print(f"✓ Stage {stage} completed successfully")
            else:
                self.pipeline_state["failed_stages"].append(stage)
                print(f"✗ Stage {stage} failed")
                self._save_pipeline_state()
                return False
            
            # Save progress
            self._save_pipeline_state()
        
        total_duration = time.time() - self.start_time
        print(f"\n{'='*70}")
        print(f"Pipeline completed successfully in {total_duration/3600:.2f} hours")
        print(f"Completed stages: {self.pipeline_state['completed_stages']}")
        
        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Complete training pipeline for dysarthric voice conversion")
    
    parser.add_argument("--config", type=str, default=None,
                       help="Path to model config file")
    parser.add_argument("--training_config", type=str, default=None,
                       help="Path to training config file")
    parser.add_argument("--resume_from", type=str, default=None,
                       choices=["validate_data", "extract_units", "train_soft_encoder", 
                               "train_s2ut", "train_hifigan", "final_validation"],
                       help="Stage to resume from")
    parser.add_argument("--stages", type=str, nargs="+", default=None,
                       choices=["validate_data", "extract_units", "train_soft_encoder",
                               "train_s2ut", "train_hifigan", "final_validation"], 
                       help="Specific stages to run")
    parser.add_argument("--force", action="store_true",
                       help="Force re-run completed stages")
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = TrainingPipeline(
        config_path=args.config,
        training_config_path=args.training_config,
        resume_from_stage=args.resume_from,
        stages_to_run=args.stages
    )
    
    success = pipeline.run()
    
    if success:
        print("\n🎉 Training pipeline completed successfully!")
        print("Your dysarthric voice conversion system is ready!")
        print("\nNext steps:")
        print("1. Test the system with: python scripts/run_inference.py")
        print("2. Evaluate performance with: python evaluation/evaluate.py")
        print("3. Run MOS studies for subjective evaluation")
    else:
        print("\n❌ Training pipeline failed")
        print("Check logs for details and resume with --resume_from option")
        sys.exit(1)

if __name__ == "__main__":
    main()