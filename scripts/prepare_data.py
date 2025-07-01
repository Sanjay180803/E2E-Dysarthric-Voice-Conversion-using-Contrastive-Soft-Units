"""
Data preparation script for Tamil Dysarthric Speech Corpus (TDSC)
Organizes raw data into the required directory structure
"""

import os
import sys
import shutil
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm
import numpy as np
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.audio_utils import AudioProcessor
from config.model_config import ModelConfig

class TDSCDataPreparator:
    """
    Data preparator for Tamil Dysarthric Speech Corpus (TDSC)
    
    Expected raw data structure:
    raw_data/
        speakers/
            healthy/
                speaker_001/
                    utterance_001.wav
                    utterance_002.wav
                    ...
            dysarthric/
                mild/
                    speaker_101/
                        utterance_001.wav
                        ...
                moderate/
                    speaker_201/
                        utterance_001.wav
                        ...
                severe/
                    speaker_301/
                        utterance_001.wav
                        ...
        metadata.csv (optional)
    """
    
    def __init__(self, 
                 raw_data_dir: str,
                 output_dir: str,
                 audio_config=None):
        
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.audio_processor = AudioProcessor() if audio_config is None else AudioProcessor(**audio_config.__dict__)
        
        # Create output directory structure
        self.create_output_structure()
        
        # Statistics
        self.stats = {
            "healthy": {"speakers": 0, "utterances": 0, "total_duration": 0.0},
            "mild": {"speakers": 0, "utterances": 0, "total_duration": 0.0},
            "moderate": {"speakers": 0, "utterances": 0, "total_duration": 0.0},
            "severe": {"speakers": 0, "utterances": 0, "total_duration": 0.0}
        }
    
    def create_output_structure(self):
        """Create output directory structure"""
        directories = [
            self.output_dir / "healthy",
            self.output_dir / "dysarthric" / "mild",
            self.output_dir / "dysarthric" / "moderate", 
            self.output_dir / "dysarthric" / "severe",
            self.output_dir / "metadata"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate_audio_file(self, audio_path: Path) -> Tuple[bool, Dict]:
        """
        Validate audio file and return metadata
        
        Returns:
            (is_valid, metadata_dict)
        """
        try:
            # Load audio to check validity
            audio, sr = sf.read(str(audio_path))
            
            # Check if mono/stereo
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # Convert to mono
            
            # Basic validation
            duration = len(audio) / sr
            
            if duration < 0.5:  # Minimum 0.5 seconds
                return False, {"error": "Too short"}
            
            if duration > 20.0:  # Maximum 20 seconds  
                return False, {"error": "Too long"}
            
            # Check for silence
            if np.max(np.abs(audio)) < 0.001:
                return False, {"error": "Too quiet"}
            
            metadata = {
                "duration": duration,
                "sample_rate": sr,
                "channels": audio.ndim,
                "max_amplitude": float(np.max(np.abs(audio))),
                "rms": float(np.sqrt(np.mean(audio**2)))
            }
            
            return True, metadata
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def process_speaker_directory(self, 
                                speaker_dir: Path,
                                severity: str,
                                output_speaker_dir: Path) -> List[Dict]:
        """
        Process all audio files for a single speaker
        
        Returns:
            List of file metadata
        """
        file_metadata = []
        speaker_id = speaker_dir.name
        
        # Get all audio files
        audio_files = list(speaker_dir.glob("*.wav")) + list(speaker_dir.glob("*.WAV"))
        
        if not audio_files:
            print(f"Warning: No audio files found in {speaker_dir}")
            return file_metadata
        
        # Create output speaker directory
        output_speaker_dir.mkdir(parents=True, exist_ok=True)
        
        for audio_file in tqdm(audio_files, desc=f"Processing {speaker_id}", leave=False):
            # Validate audio
            is_valid, metadata = self.validate_audio_file(audio_file)
            
            if not is_valid:
                print(f"Skipping invalid file {audio_file}: {metadata.get('error', 'Unknown error')}")
                continue
            
            # Generate output filename
            utterance_id = audio_file.stem
            output_filename = f"{speaker_id}_{utterance_id}.wav"
            output_path = output_speaker_dir / output_filename
            
            try:
                # Load, process, and save audio
                audio = self.audio_processor.load_audio(audio_file)
                
                # Resample if necessary
                if metadata["sample_rate"] != self.audio_processor.sampling_rate:
                    audio_np = audio.numpy()
                    audio_np = librosa.resample(
                        audio_np,
                        orig_sr=metadata["sample_rate"],
                        target_sr=self.audio_processor.sampling_rate
                    )
                    audio = torch.from_numpy(audio_np)
                
                # Save processed audio
                self.audio_processor.save_audio(audio, output_path)
                
                # Update metadata
                processed_metadata = {
                    "original_path": str(audio_file),
                    "processed_path": str(output_path),
                    "speaker_id": speaker_id,
                    "utterance_id": utterance_id,
                    "severity": severity,
                    "duration": metadata["duration"],
                    "sample_rate": self.audio_processor.sampling_rate,
                    "original_sample_rate": metadata["sample_rate"],
                    "max_amplitude": metadata["max_amplitude"],
                    "rms": metadata["rms"]
                }
                
                file_metadata.append(processed_metadata)
                
                # Update statistics
                self.stats[severity]["utterances"] += 1
                self.stats[severity]["total_duration"] += metadata["duration"]
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
        
        if file_metadata:
            self.stats[severity]["speakers"] += 1
        
        return file_metadata
    
    def process_healthy_speakers(self) -> List[Dict]:
        """Process healthy speakers"""
        print("Processing healthy speakers...")
        
        healthy_dir = self.raw_data_dir / "speakers" / "healthy"
        output_healthy_dir = self.output_dir / "healthy"
        
        if not healthy_dir.exists():
            print(f"Warning: Healthy speakers directory not found: {healthy_dir}")
            return []
        
        all_metadata = []
        speaker_dirs = [d for d in healthy_dir.iterdir() if d.is_dir()]
        
        for speaker_dir in tqdm(speaker_dirs, desc="Healthy speakers"):
            speaker_metadata = self.process_speaker_directory(
                speaker_dir=speaker_dir,
                severity="healthy",
                output_speaker_dir=output_healthy_dir / speaker_dir.name
            )
            all_metadata.extend(speaker_metadata)
        
        return all_metadata
    
    def process_dysarthric_speakers(self) -> List[Dict]:
        """Process dysarthric speakers"""
        print("Processing dysarthric speakers...")
        
        dysarthric_dir = self.raw_data_dir / "speakers" / "dysarthric"
        
        if not dysarthric_dir.exists():
            print(f"Warning: Dysarthric speakers directory not found: {dysarthric_dir}")
            return []
        
        all_metadata = []
        
        for severity in ["mild", "moderate", "severe"]:
            severity_dir = dysarthric_dir / severity
            output_severity_dir = self.output_dir / "dysarthric" / severity
            
            if not severity_dir.exists():
                print(f"Warning: {severity} dysarthric directory not found: {severity_dir}")
                continue
            
            speaker_dirs = [d for d in severity_dir.iterdir() if d.is_dir()]
            
            for speaker_dir in tqdm(speaker_dirs, desc=f"{severity.capitalize()} speakers"):
                speaker_metadata = self.process_speaker_directory(
                    speaker_dir=speaker_dir,
                    severity=severity,
                    output_speaker_dir=output_severity_dir / speaker_dir.name
                )
                all_metadata.extend(speaker_metadata)
        
        return all_metadata
    
    def create_train_val_test_splits(self, 
                                   all_metadata: List[Dict],
                                   train_ratio: float = 0.8,
                                   val_ratio: float = 0.1,
                                   test_ratio: float = 0.1) -> Dict[str, List[Dict]]:
        """
        Create train/validation/test splits ensuring speaker independence
        """
        print("Creating train/validation/test splits...")
        
        # Group by speaker and severity
        speakers_by_severity = {
            "healthy": {},
            "mild": {},
            "moderate": {},
            "severe": {}
        }
        
        for item in all_metadata:
            severity = item["severity"]
            speaker_id = item["speaker_id"]
            
            if speaker_id not in speakers_by_severity[severity]:
                speakers_by_severity[severity][speaker_id] = []
            
            speakers_by_severity[severity][speaker_id].append(item)
        
        # Create splits for each severity level
        splits = {"train": [], "val": [], "test": []}
        
        for severity, speakers in speakers_by_severity.items():
            if not speakers:
                continue
            
            speaker_ids = list(speakers.keys())
            np.random.seed(42)  # For reproducibility
            np.random.shuffle(speaker_ids)
            
            n_speakers = len(speaker_ids)
            n_train = int(n_speakers * train_ratio)
            n_val = int(n_speakers * val_ratio)
            
            train_speakers = speaker_ids[:n_train]
            val_speakers = speaker_ids[n_train:n_train + n_val]
            test_speakers = speaker_ids[n_train + n_val:]
            
            # Assign utterances to splits
            for speaker_id in train_speakers:
                splits["train"].extend(speakers[speaker_id])
            
            for speaker_id in val_speakers:
                splits["val"].extend(speakers[speaker_id])
            
            for speaker_id in test_speakers:
                splits["test"].extend(speakers[speaker_id])
        
        return splits
    
    def save_metadata(self, all_metadata: List[Dict], splits: Dict[str, List[Dict]]):
        """Save metadata files"""
        print("Saving metadata...")
        
        metadata_dir = self.output_dir / "metadata"
        
        # Save complete metadata
        with open(metadata_dir / "complete_metadata.json", 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        # Save splits
        for split_name, split_data in splits.items():
            with open(metadata_dir / f"{split_name}_metadata.json", 'w') as f:
                json.dump(split_data, f, indent=2)
        
        # Save statistics
        with open(metadata_dir / "dataset_statistics.json", 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Create CSV files for easier analysis
        df = pd.DataFrame(all_metadata)
        df.to_csv(metadata_dir / "complete_metadata.csv", index=False)
        
        for split_name, split_data in splits.items():
            if split_data:
                split_df = pd.DataFrame(split_data)
                split_df.to_csv(metadata_dir / f"{split_name}_metadata.csv", index=False)
    
    def print_statistics(self):
        """Print dataset statistics"""
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        
        total_speakers = sum(self.stats[severity]["speakers"] for severity in self.stats)
        total_utterances = sum(self.stats[severity]["utterances"] for severity in self.stats)
        total_duration = sum(self.stats[severity]["total_duration"] for severity in self.stats)
        
        print(f"Total Speakers: {total_speakers}")
        print(f"Total Utterances: {total_utterances}")
        print(f"Total Duration: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours)")
        print()
        
        for severity, stats in self.stats.items():
            print(f"{severity.upper()}:")
            print(f"  Speakers: {stats['speakers']}")
            print(f"  Utterances: {stats['utterances']}")
            print(f"  Duration: {stats['total_duration']:.2f}s ({stats['total_duration']/3600:.2f}h)")
            if stats['utterances'] > 0:
                avg_duration = stats['total_duration'] / stats['utterances']
                print(f"  Avg utterance duration: {avg_duration:.2f}s")
            print()
    
    def run(self):
        """Run the complete data preparation pipeline"""
        print("Starting data preparation...")
        print(f"Raw data directory: {self.raw_data_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Process healthy speakers
        healthy_metadata = self.process_healthy_speakers()
        
        # Process dysarthric speakers
        dysarthric_metadata = self.process_dysarthric_speakers()
        
        # Combine all metadata
        all_metadata = healthy_metadata + dysarthric_metadata
        
        if not all_metadata:
            print("Error: No valid audio files found!")
            return
        
        # Create splits
        splits = self.create_train_val_test_splits(all_metadata)
        
        # Save metadata
        self.save_metadata(all_metadata, splits)
        
        # Print statistics
        self.print_statistics()
        
        print("Data preparation completed successfully!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Prepare Tamil Dysarthric Speech Corpus")
    parser.add_argument("--raw_data_dir", type=str, required=True,
                       help="Directory containing raw TDSC data")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for processed data")
    parser.add_argument("--config_path", type=str, default=None,
                       help="Path to model config file")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_path:
        from training.trainer_utils import load_config
        config_dict = load_config(args.config_path)
        config = ModelConfig()
        # Update config with loaded values (simplified)
    else:
        config = ModelConfig()
    
    # Initialize data preparator
    preparator = TDSCDataPreparator(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        audio_config=config.audio
    )
    
    # Run data preparation
    preparator.run()

if __name__ == "__main__":
    main()