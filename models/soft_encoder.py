"""
Soft Content Encoder for converting discrete units to soft units
Based on the paper methodology for preserving fine phonetic detail
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
from pathlib import Path

class SoftContentEncoder(nn.Module):
    """
    Soft Content Encoder that converts discrete units to soft unit distributions
    
    From the paper:
    "To mitigate the loss of fine phonetic detail from purely discrete labels, 
    we adopt a soft-units approach. A small linear projection layer is trained 
    on top of mHuBERT such that, for each frame, it outputs a probability vector"
    
    Modified to work with fairseq-extracted discrete units
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding layer for discrete units
        self.unit_embedding = nn.Embedding(
            num_embeddings=config.input_dim,  # Number of discrete units (1000)
            embedding_dim=config.embedding_dim
        )
        
        # Projection layers
        if config.hidden_dim > 0:
            self.projection = nn.Sequential(
                nn.Linear(config.embedding_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.output_dim)
            )
        else:
            self.projection = nn.Linear(config.embedding_dim, config.output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, 
                discrete_units: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass converting discrete units to soft units
        
        Args:
            discrete_units: Discrete unit indices [batch_size, seq_len]
            mask: Optional mask for padding [batch_size, seq_len]
            
        Returns:
            Soft unit probability distributions [batch_size, seq_len, num_clusters]
        """
        # Embed discrete units
        unit_embeddings = self.unit_embedding(discrete_units)  # [B, T, embed_dim]
        
        # Apply dropout
        unit_embeddings = self.dropout(unit_embeddings)
        
        # Project to soft units
        soft_logits = self.projection(unit_embeddings)  # [B, T, output_dim]
        
        # Apply softmax to get probability distributions
        soft_units = F.softmax(soft_logits, dim=-1)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask to match soft_units dimensions
            mask_expanded = mask.unsqueeze(-1).expand_as(soft_units)
            soft_units = soft_units * mask_expanded
        
        return soft_units
    
    def compute_cross_entropy_loss(self,
                                  soft_units: torch.Tensor,
                                  discrete_targets: torch.Tensor,
                                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute cross-entropy loss against discrete unit targets
        
        Args:
            soft_units: Predicted soft unit distributions [batch_size, seq_len, num_clusters]
            discrete_targets: Discrete unit indices [batch_size, seq_len]
            mask: Optional mask for padding [batch_size, seq_len]
            
        Returns:
            Cross-entropy loss
        """
        batch_size, seq_len, num_clusters = soft_units.shape
        
        # Reshape for loss computation
        soft_units_flat = soft_units.view(-1, num_clusters)  # [B*T, num_clusters]
        targets_flat = discrete_targets.view(-1)  # [B*T]
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            soft_units_flat, 
            targets_flat, 
            reduction='none'
        )
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(-1)
            loss = loss * mask_flat
            loss = loss.sum() / mask_flat.sum()
        else:
            loss = loss.mean()
        
        return loss

class SoftUnitConverter:
    """
    Utility class for converting between discrete and soft units
    """
    
    def __init__(self, num_clusters: int = 1000):
        self.num_clusters = num_clusters
    
    def discrete_to_one_hot(self, 
                           discrete_units: torch.Tensor) -> torch.Tensor:
        """
        Convert discrete units to one-hot representations
        
        Args:
            discrete_units: Discrete unit indices [batch_size, seq_len]
            
        Returns:
            One-hot representations [batch_size, seq_len, num_clusters]
        """
        return F.one_hot(discrete_units, num_classes=self.num_clusters).float()
    
    def soft_to_discrete(self, 
                        soft_units: torch.Tensor) -> torch.Tensor:
        """
        Convert soft units to discrete by taking argmax
        
        Args:
            soft_units: Soft unit distributions [batch_size, seq_len, num_clusters]
            
        Returns:
            Discrete unit indices [batch_size, seq_len]
        """
        return torch.argmax(soft_units, dim=-1)
    
    def interpolate_soft_units(self,
                              soft_units_1: torch.Tensor,
                              soft_units_2: torch.Tensor,
                              alpha: float = 0.5) -> torch.Tensor:
        """
        Interpolate between two soft unit sequences
        
        Args:
            soft_units_1: First soft unit sequence
            soft_units_2: Second soft unit sequence  
            alpha: Interpolation weight (0 = first, 1 = second)
            
        Returns:
            Interpolated soft units
        """
        return (1 - alpha) * soft_units_1 + alpha * soft_units_2

class SoftUnitTrainer:
    """
    Trainer class for the Soft Content Encoder
    """
    
    def __init__(self, 
                 model: SoftContentEncoder,
                 optimizer: torch.optim.Optimizer,
                 device: str = "cuda"):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
        
        self.converter = SoftUnitConverter(model.config.output_dim)
    
    def train_step(self,
                   mhubert_features: torch.Tensor,
                   discrete_targets: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            mhubert_features: Features from mHuBERT
            discrete_targets: K-means cluster indices
            mask: Optional padding mask
            
        Returns:
            Dictionary with loss and metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        soft_units = self.model(mhubert_features, mask)
        
        # Compute loss
        loss = self.model.compute_cross_entropy_loss(
            soft_units, discrete_targets, mask
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Compute accuracy
        predicted_discrete = self.converter.soft_to_discrete(soft_units)
        if mask is not None:
            correct = (predicted_discrete == discrete_targets) * mask
            accuracy = correct.sum().float() / mask.sum()
        else:
            accuracy = (predicted_discrete == discrete_targets).float().mean()
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy.item()
        }
    
    def validate_step(self,
                     mhubert_features: torch.Tensor,
                     discrete_targets: torch.Tensor,
                     mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Single validation step
        
        Args:
            mhubert_features: Features from mHuBERT
            discrete_targets: K-means cluster indices
            mask: Optional padding mask
            
        Returns:
            Dictionary with loss and metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            soft_units = self.model(mhubert_features, mask)
            
            # Compute loss
            loss = self.model.compute_cross_entropy_loss(
                soft_units, discrete_targets, mask
            )
            
            # Compute accuracy
            predicted_discrete = self.converter.soft_to_discrete(soft_units)
            if mask is not None:
                correct = (predicted_discrete == discrete_targets) * mask
                accuracy = correct.sum().float() / mask.sum()
            else:
                accuracy = (predicted_discrete == discrete_targets).float().mean()
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy.item()
        }
    
    def extract_soft_units(self,
                          mhubert_features: torch.Tensor,
                          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract soft units from mHuBERT features
        
        Args:
            mhubert_features: Features from mHuBERT
            mask: Optional padding mask
            
        Returns:
            Soft unit distributions
        """
        self.model.eval()
        
        with torch.no_grad():
            soft_units = self.model(mhubert_features, mask)
        
        return soft_units

class SoftUnitDataset(torch.utils.data.Dataset):
    """
    Dataset for training the Soft Content Encoder
    """
    
    def __init__(self,
                 mhubert_features_dir: str,
                 discrete_units_dir: str,
                 split: str = "train"):
        
        self.mhubert_features_dir = Path(mhubert_features_dir)
        self.discrete_units_dir = Path(discrete_units_dir)
        
        # Load file lists
        self.feature_files = sorted(list(self.mhubert_features_dir.glob(f"{split}_*.pt")))
        self.unit_files = sorted(list(self.discrete_units_dir.glob(f"{split}_*.pt")))
        
        assert len(self.feature_files) == len(self.unit_files), \
            "Number of feature and unit files must match"
    
    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        # Load mHuBERT features
        features = torch.load(self.feature_files[idx])
        
        # Load discrete units
        units = torch.load(self.unit_files[idx])
        
        # Ensure same sequence length
        min_len = min(features.size(0), units.size(0))
        features = features[:min_len]
        units = units[:min_len]
        
        return {
            "mhubert_features": features,
            "discrete_units": units
        }

def collate_fn_soft_encoder(batch):
    """Collate function for soft encoder dataset"""
    features = [item["mhubert_features"] for item in batch]
    units = [item["discrete_units"] for item in batch]
    
    # Pad to same length
    max_len = max(f.size(0) for f in features)
    feature_dim = features[0].size(1)
    
    padded_features = torch.zeros(len(batch), max_len, feature_dim)
    padded_units = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len)
    
    for i, (feat, unit) in enumerate(zip(features, units)):
        length = feat.size(0)
        padded_features[i, :length] = feat
        padded_units[i, :length] = unit
        mask[i, :length] = 1
    
    return {
        "mhubert_features": padded_features,
        "discrete_units": padded_units,
        "mask": mask
    }