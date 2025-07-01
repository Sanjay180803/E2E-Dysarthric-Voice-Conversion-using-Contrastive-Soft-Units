"""
Dataset classes for Dysarthric Voice Conversion
"""

import os
import json
import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

from .audio_utils import AudioProcessor

class TamilDysarthricDataset(Dataset):
    """
    Tamil Dysarthric Speech Corpus (TDSC) Dataset
    Based on the paper specifications:
    - 20 dysarthric speakers (13 male, 7 female)
    - 10 healthy speakers (5 male, 5 female)
    - 365 utterances per speaker (103 words + 262 sentences)
    """
    
    def __init__(self,
                 data_dir: str,
                 split: str = "train",
                 audio_config=None,
                 transform=None,
                 severity_filter: List[str] = None,
                 max_audio_length: float = 10.0,
                 min_audio_length: float = 0.5):
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor() if audio_config is None else AudioProcessor(**audio_config.__dict__)
        
        # Load metadata
        self.metadata_file = self.data_dir / "metadata.json"
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = self._create_metadata()
            
        # Filter by severity if specified
        if severity_filter is not None:
            self.data_list = self._filter_by_severity(severity_filter)
        else:
            self.data_list = self.metadata[split]
            
    def _create_metadata(self) -> Dict:
        """Create metadata from directory structure"""
        metadata = {
            "healthy": [],
            "dysarthric": [],
            "train": [],
            "val": [],
            "test": []
        }
        
        # Expected directory structure:
        # data_dir/
        #   healthy/
        #     speaker_id/
        #       utterance_id.wav
        #   dysarthric/
        #     mild/
        #       speaker_id/
        #         utterance_id.wav
        #     moderate/
        #       speaker_id/
        #         utterance_id.wav
        #     severe/
        #       speaker_id/
        #         utterance_id.wav
        
        # Process healthy speakers
        healthy_dir = self.data_dir / "healthy"
        if healthy_dir.exists():
            for speaker_dir in healthy_dir.iterdir():
                if speaker_dir.is_dir():
                    speaker_id = speaker_dir.name
                    for audio_file in speaker_dir.glob("*.wav"):
                        metadata["healthy"].append({
                            "audio_path": str(audio_file),
                            "speaker_id": speaker_id,
                            "utterance_id": audio_file.stem,
                            "severity": "healthy",
                            "gender": self._infer_gender(speaker_id)
                        })
        
        # Process dysarthric speakers
        dysarthric_dir = self.data_dir / "dysarthric"
        if dysarthric_dir.exists():
            for severity_dir in dysarthric_dir.iterdir():
                if severity_dir.is_dir():
                    severity = severity_dir.name
                    for speaker_dir in severity_dir.iterdir():
                        if speaker_dir.is_dir():
                            speaker_id = speaker_dir.name
                            for audio_file in speaker_dir.glob("*.wav"):
                                metadata["dysarthric"].append({
                                    "audio_path": str(audio_file),
                                    "speaker_id": speaker_id,
                                    "utterance_id": audio_file.stem,
                                    "severity": severity,
                                    "gender": self._infer_gender(speaker_id)
                                })
        
        # Create train/val/test splits
        all_data = metadata["healthy"] + metadata["dysarthric"]
        
        # Group by speaker to ensure speaker-independent splits
        speakers = {}
        for item in all_data:
            speaker_id = item["speaker_id"]
            if speaker_id not in speakers:
                speakers[speaker_id] = []
            speakers[speaker_id].append(item)
        
        speaker_ids = list(speakers.keys())
        train_speakers, temp_speakers = train_test_split(speaker_ids, test_size=0.3, random_state=42)
        val_speakers, test_speakers = train_test_split(temp_speakers, test_size=0.5, random_state=42)
        
        for speaker_id in train_speakers:
            metadata["train"].extend(speakers[speaker_id])
        for speaker_id in val_speakers:
            metadata["val"].extend(speakers[speaker_id])
        for speaker_id in test_speakers:
            metadata["test"].extend(speakers[speaker_id])
        
        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return metadata
    
    def _infer_gender(self, speaker_id: str) -> str:
        """
        Infer gender from speaker ID based on TDSC naming convention
        
        Naming convention:
        - Files starting with 'F' = Female (e.g., FC01_1.wav, FC01_2.wav)
        - Files starting with 'M' = Male (e.g., MC05_104.wav, MC05_115.wav)
        """
        # Convert to uppercase to handle case variations
        speaker_id_upper = speaker_id.upper()
        
        if speaker_id_upper.startswith('F'):
            return "female"
        elif speaker_id_upper.startswith('M'):
            return "male"
        else:
            # Fallback for unexpected naming patterns
            return "unknown"
    
    def _filter_by_severity(self, severity_filter: List[str]) -> List[Dict]:
        """Filter data by dysarthria severity"""
        filtered_data = []
        for item in self.metadata[self.split]:
            if item["severity"] in severity_filter:
                filtered_data.append(item)
        return filtered_data
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample"""
        item = self.data_list[idx]
        
        # Load and preprocess audio
        audio, mel_spec = self.audio_processor.preprocess_audio(
            item["audio_path"],
            max_length=self.max_audio_length,
            min_length=self.min_audio_length
        )
        
        # Apply transforms if specified
        if self.transform is not None:
            mel_spec = self.transform(mel_spec)
        
        return {
            "audio": audio,
            "mel_spectrogram": mel_spec,
            "speaker_id": item["speaker_id"],
            "utterance_id": item["utterance_id"],
            "severity": item["severity"],
            "audio_path": item["audio_path"]
        }

class DysarthricS2UTDataset(Dataset):
    """
    Dataset for Speech-to-Unit Translation training
    Pairs dysarthric speech with healthy soft units
    Uses fairseq-extracted discrete units
    """
    
    def __init__(self,
                 dysarthric_data_dir: str,
                 fairseq_units_file: str,
                 soft_units_dir: str,
                 split: str = "train",
                 audio_config=None,
                 severity_filter: List[str] = ["mild", "moderate"],
                 max_audio_length: float = 10.0,
                 apply_spec_augment: bool = True):
        
        self.dysarthric_data_dir = Path(dysarthric_data_dir)
        self.fairseq_units_file = fairseq_units_file
        self.soft_units_dir = Path(soft_units_dir)
        self.split = split
        self.severity_filter = severity_filter
        self.max_audio_length = max_audio_length
        self.apply_spec_augment = apply_spec_augment and split == "train"
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor() if audio_config is None else AudioProcessor(**audio_config.__dict__)
        
        # Load fairseq units and create mappings
        self.fairseq_units = self._load_fairseq_units()
        
        # Create data pairs
        self.data_pairs = self._create_data_pairs()
        
    def _load_fairseq_units(self) -> Dict[int, List[int]]:
        """Load fairseq-extracted discrete units"""
        units = {}
        
        with open(self.fairseq_units_file, 'r') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    unit_sequence = [int(x) for x in line.split()]
                    units[idx] = unit_sequence
        
        print(f"Loaded {len(units)} unit sequences from fairseq")
        return units
    
    def _create_data_pairs(self) -> List[Dict]:
        """Create pairs of dysarthric speech and target units"""
        pairs = []
        
        # For this implementation, we'll use a simple mapping
        # In practice, you would need proper alignment between audio files and units
        
        # Get dysarthric audio files
        dysarthric_files = []
        for severity in self.severity_filter:
            severity_dir = self.dysarthric_data_dir / severity
            if severity_dir.exists():
                dysarthric_files.extend(list(severity_dir.glob("**/*.wav")))
        
        # Create pairs (simplified - assumes order matches fairseq units)
        for i, audio_file in enumerate(dysarthric_files):
            if i < len(self.fairseq_units):
                # Target healthy units (from fairseq extraction of healthy speech)
                target_units = self.fairseq_units.get(i % 100, [])  # Cycle through healthy units
                
                pairs.append({
                    "dysarthric_audio": str(audio_file),
                    "target_units": target_units,
                    "severity": self._get_severity_from_path(str(audio_file)),
                    "audio_idx": i
                })
        
        print(f"Created {len(pairs)} data pairs for {self.split}")
        return pairs
    
    def _get_severity_from_path(self, audio_path: str) -> str:
        """Extract severity from audio file path"""
        if "mild" in audio_path:
            return "mild"
        elif "moderate" in audio_path:
            return "moderate"
        elif "severe" in audio_path:
            return "severe"
        else:
            return "unknown"
    
    def __len__(self) -> int:
        return len(self.data_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data_pairs[idx]
        
        # Load dysarthric audio
        audio, mel_spec = self.audio_processor.preprocess_audio(
            item["dysarthric_audio"],
            max_length=self.max_audio_length
        )
        
        # Apply SpecAugment if enabled
        if self.apply_spec_augment:
            mel_spec = self.audio_processor.apply_spec_augment(mel_spec)
        
        # Convert target units to tensor
        target_units = torch.LongTensor(item["target_units"])
        
        # For contrastive learning, get a negative sample
        negative_idx = random.randint(0, len(self.data_pairs) - 1)
        while negative_idx == idx:
            negative_idx = random.randint(0, len(self.data_pairs) - 1)
        
        negative_units = torch.LongTensor(self.data_pairs[negative_idx]["target_units"])
        
        return {
            "dysarthric_mel": mel_spec,
            "target_units": target_units,
            "negative_units": negative_units,
            "severity": item["severity"],
            "audio_idx": item["audio_idx"]
        }

def collate_fn_s2ut(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for S2UT dataset with fairseq units"""
    # Pad sequences to the same length
    mel_specs = [item["dysarthric_mel"] for item in batch]
    target_units = [item["target_units"] for item in batch]
    negative_units = [item["negative_units"] for item in batch]
    
    # Pad mel spectrograms
    max_mel_len = max(mel.size(1) for mel in mel_specs)
    padded_mels = torch.zeros(len(batch), mel_specs[0].size(0), max_mel_len)
    mel_lengths = torch.LongTensor([mel.size(1) for mel in mel_specs])
    
    for i, mel in enumerate(mel_specs):
        padded_mels[i, :, :mel.size(1)] = mel
    
    # Pad unit sequences
    max_unit_len = max(units.size(0) for units in target_units)
    padded_target = torch.zeros(len(batch), max_unit_len, dtype=torch.long)
    padded_negative = torch.zeros(len(batch), max_unit_len, dtype=torch.long)
    unit_lengths = torch.LongTensor([units.size(0) for units in target_units])
    
    for i, (target, negative) in enumerate(zip(target_units, negative_units)):
        padded_target[i, :target.size(0)] = target
        # Handle negative units that might be different lengths
        neg_len = min(negative.size(0), max_unit_len)
        padded_negative[i, :neg_len] = negative[:neg_len]
    
    return {
        "dysarthric_mel": padded_mels,
        "target_units": padded_target,
        "negative_units": padded_negative,
        "mel_lengths": mel_lengths,
        "unit_lengths": unit_lengths,
        "severities": [item["severity"] for item in batch],
        "audio_indices": [item["audio_idx"] for item in batch]
    }

def create_dataloaders(config, model_type: str = "s2ut"):
    """Create dataloaders for different model types"""
    
    if model_type == "s2ut":
        # Use fairseq units
        fairseq_units_file = f"{config.data_dir}/units/discrete_units.txt"
        
        train_dataset = DysarthricS2UTDataset(
            dysarthric_data_dir=f"{config.data_dir}/dysarthric",
            fairseq_units_file=fairseq_units_file,
            soft_units_dir=f"{config.data_dir}/soft_units",
            split="train",
            audio_config=config.audio,
            severity_filter=["mild", "moderate"],
            apply_spec_augment=True
        )
        
        val_dataset = DysarthricS2UTDataset(
            dysarthric_data_dir=f"{config.data_dir}/dysarthric", 
            fairseq_units_file=fairseq_units_file,
            soft_units_dir=f"{config.data_dir}/soft_units",
            split="val",
            audio_config=config.audio,
            severity_filter=["mild", "moderate"],
            apply_spec_augment=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_fn_s2ut,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn_s2ut,
            pin_memory=True
        )
        
    else:  # Regular audio dataset
        train_dataset = TamilDysarthricDataset(
            data_dir=config.data_dir,
            split="train",
            audio_config=config.audio,
            severity_filter=["healthy"] if model_type == "healthy" else ["mild", "moderate"]
        )
        
        val_dataset = TamilDysarthricDataset(
            data_dir=config.data_dir,
            split="val", 
            audio_config=config.audio,
            severity_filter=["healthy"] if model_type == "healthy" else ["mild", "moderate"]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader