"""
Enhanced training pipeline with severity-aware learning and novel features
"""

import os
import sys
import time
import json
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.enhanced_config import EnhancedS2UTConfig, EnhancedTrainingConfig, EnhancedDataConfig
from models.severity_aware_s2ut import SeverityAwareS2UTModel
from data.enhanced_preprocessing import EnhancedDysarthriaDataset
from evaluation.advanced_metrics import AdvancedMetricsCalculator
from training.trainer_utils import EarlyStopping, ModelCheckpoint, LearningRateScheduler

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

class EnhancedTrainingPipeline:
    """Enhanced training pipeline with innovative dysarthric voice conversion features"""
    
    def __init__(self, config_path: str, data_dir: str, output_dir: str):
        self.config_path = config_path
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load configurations
        self.model_config = EnhancedS2UTConfig()
        self.training_config = EnhancedTrainingConfig()
        self.data_config = EnhancedDataConfig()
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.metrics_calculator = None
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "training.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def prepare_data(self):
        """Prepare enhanced datasets with severity awareness"""
        self.logger.info("Preparing enhanced datasets...")
        
        # Training dataset
        train_dataset = EnhancedDysarthriaDataset(
            data_dir=self.data_dir / "train",
            config=self.data_config,
            mode="train",
            apply_augmentation=True,
            severity_aware=True
        )
        
        # Validation dataset
        val_dataset = EnhancedDysarthriaDataset(
            data_dir=self.data_dir / "val",
            config=self.data_config,
            mode="val",
            apply_augmentation=False,
            severity_aware=True
        )
        
        # Data loaders with enhanced sampling
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=self.training_config.num_workers,
            pin_memory=self.training_config.pin_memory,
            persistent_workers=self.training_config.persistent_workers,
            collate_fn=self._collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=self.training_config.num_workers,
            pin_memory=self.training_config.pin_memory,
            persistent_workers=self.training_config.persistent_workers,
            collate_fn=self._collate_fn
        )
        
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")
        
    def _collate_fn(self, batch):
        """Enhanced collate function for batching"""
        # Separate different components
        source_audio = [item['source_audio'] for item in batch]
        target_audio = [item['target_audio'] for item in batch]
        severity_labels = [item['severity_label'] for item in batch]
        prosody_features = [item['prosody_features'] for item in batch]
        
        # Pad sequences
        source_audio = self._pad_sequences(source_audio)
        target_audio = self._pad_sequences(target_audio)
        prosody_features = self._pad_sequences(prosody_features)
        
        return {
            'source_audio': torch.FloatTensor(source_audio),
            'target_audio': torch.FloatTensor(target_audio),
            'severity_labels': torch.LongTensor(severity_labels),
            'prosody_features': torch.FloatTensor(prosody_features),
        }
        
    def _pad_sequences(self, sequences):
        """Pad sequences to same length"""
        max_len = max(len(seq) for seq in sequences)
        padded = []
        for seq in sequences:
            if len(seq) < max_len:
                padding = max_len - len(seq)
                seq = np.pad(seq, (0, padding), mode='constant')
            padded.append(seq)
        return np.array(padded)
        
    def initialize_model(self):
        """Initialize the enhanced severity-aware model"""
        self.logger.info("Initializing enhanced S2UT model...")
        
        self.model = SeverityAwareS2UTModel(
            config=self.model_config,
            vocab_size=1000  # Adjust based on your tokenizer
        ).to(self.device)
        
        # Initialize metrics calculator
        self.metrics_calculator = AdvancedMetricsCalculator(
            sample_rate=self.data_config.sample_rate
        )
        
        self.logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
        
    def setup_training(self):
        """Setup optimizer, scheduler, and training utilities"""
        self.logger.info("Setting up training components...")
        
        # Optimizer
        if self.training_config.optimizer_type.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                betas=(self.training_config.beta1, self.training_config.beta2),
                weight_decay=self.training_config.weight_decay
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.training_config.learning_rate
            )
            
        # Learning rate scheduler
        total_steps = len(self.train_loader) * self.training_config.max_epochs
        warmup_steps = int(total_steps * self.training_config.warmup_ratio)
        
        self.scheduler = LearningRateScheduler(
            optimizer=self.optimizer,
            scheduler_type=self.training_config.lr_scheduler,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=self.training_config.min_lr_ratio
        )
        
        # Early stopping and checkpointing
        self.early_stopping = EarlyStopping(
            patience=self.training_config.patience,
            min_delta=1e-4,
            mode='min'
        )
        
        self.checkpoint_manager = ModelCheckpoint(
            dirpath=self.output_dir / "checkpoints",
            filename="enhanced_s2ut_{epoch:03d}_{val_loss:.4f}",
            save_top_k=self.training_config.save_top_k,
            mode='min'
        )
        
    def compute_enhanced_loss(self, batch, outputs):
        """Compute enhanced loss with multiple components"""
        device = batch['source_audio'].device
        
        # Reconstruction loss
        recon_loss = nn.MSELoss()(outputs['reconstructed'], batch['target_audio'])
        
        # Contrastive loss with severity awareness
        contrastive_loss = self.model.compute_severity_aware_contrastive_loss(
            outputs['embeddings'], 
            batch['severity_labels']
        )
        
        # Severity classification loss
        severity_loss = nn.CrossEntropyLoss()(
            outputs['severity_predictions'], 
            batch['severity_labels']
        )
        
        # Prosody consistency loss
        prosody_loss = nn.MSELoss()(
            outputs['prosody_features'], 
            batch['prosody_features']
        )
        
        # Combine losses
        total_loss = (
            self.training_config.reconstruction_weight * recon_loss +
            self.training_config.contrastive_weight * contrastive_loss +
            self.training_config.severity_classification_weight * severity_loss +
            self.training_config.prosody_weight * prosody_loss
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'contrastive_loss': contrastive_loss,
            'severity_loss': severity_loss,
            'prosody_loss': prosody_loss
        }
        
    def train_epoch(self, epoch: int):
        """Enhanced training epoch with comprehensive logging"""
        self.model.train()
        total_losses = {'total_loss': 0, 'reconstruction_loss': 0, 'contrastive_loss': 0, 
                       'severity_loss': 0, 'prosody_loss': 0}
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            
            # Compute loss
            losses = self.compute_enhanced_loss(batch, outputs)
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.training_config.gradient_clipping
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update running losses
            for key, value in losses.items():
                total_losses[key] += value.item()
                
            # Update progress bar
            if batch_idx % 100 == 0:
                avg_loss = total_losses['total_loss'] / (batch_idx + 1)
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
        # Average losses
        avg_losses = {k: v / len(self.train_loader) for k, v in total_losses.items()}
        return avg_losses
        
    def validate_epoch(self, epoch: int):
        """Enhanced validation with comprehensive metrics"""
        self.model.eval()
        total_losses = {'total_loss': 0, 'reconstruction_loss': 0, 'contrastive_loss': 0, 
                       'severity_loss': 0, 'prosody_loss': 0}
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                
                # Compute loss
                losses = self.compute_enhanced_loss(batch, outputs)
                
                # Update running losses
                for key, value in losses.items():
                    total_losses[key] += value.item()
                    
                # Collect predictions for metrics
                all_predictions.extend(outputs['reconstructed'].cpu().numpy())
                all_targets.extend(batch['target_audio'].cpu().numpy())
                
        # Average losses
        avg_losses = {k: v / len(self.val_loader) for k, v in total_losses.items()}
        
        # Compute advanced metrics
        advanced_metrics = self.metrics_calculator.compute_comprehensive_metrics(
            predictions=all_predictions[:10],  # Sample for speed
            targets=all_targets[:10],
            severity_labels=[0, 1, 2] * 3 + [0]  # Sample severity labels
        )
        
        return avg_losses, advanced_metrics
        
    def train(self):
        """Main enhanced training loop"""
        self.logger.info("Starting enhanced training pipeline...")
        
        # Prepare everything
        self.prepare_data()
        self.initialize_model()
        self.setup_training()
        
        best_val_loss = float('inf')
        training_history = []
        
        for epoch in range(self.training_config.max_epochs):
            start_time = time.time()
            
            # Training phase
            train_losses = self.train_epoch(epoch)
            
            # Validation phase
            val_losses, val_metrics = self.validate_epoch(epoch)
            
            epoch_time = time.time() - start_time
            
            # Logging
            self.logger.info(f"Epoch {epoch+1}/{self.training_config.max_epochs}")
            self.logger.info(f"Train Loss: {train_losses['total_loss']:.4f}")
            self.logger.info(f"Val Loss: {val_losses['total_loss']:.4f}")
            self.logger.info(f"PESQ: {val_metrics.get('pesq', 0):.3f}")
            self.logger.info(f"STOI: {val_metrics.get('stoi', 0):.3f}")
            self.logger.info(f"Epoch Time: {epoch_time:.2f}s")
            
            # Save training history
            history_entry = {
                'epoch': epoch + 1,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_metrics': val_metrics,
                'epoch_time': epoch_time,
                'lr': self.scheduler.get_last_lr()[0]
            }
            training_history.append(history_entry)
            
            # Checkpointing
            current_val_loss = val_losses['total_loss']
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                val_loss=current_val_loss,
                metrics=val_metrics
            )
            
            # Early stopping check
            if self.early_stopping.should_stop(current_val_loss):
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
                
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                
        # Save training history
        with open(self.output_dir / "training_history.json", 'w') as f:
            json.dump(training_history, f, indent=2)
            
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
def main():
    parser = argparse.ArgumentParser(description="Enhanced Dysarthric Voice Conversion Training")
    parser.add_argument("--config", type=str, default="config/enhanced_config.py",
                       help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing processed data")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for models and logs")
    parser.add_argument("--stages", type=str, default="all",
                       choices=["all", "encoder", "s2ut", "vocoder"],
                       help="Training stages to run")
    parser.add_argument("--use_severity_awareness", action="store_true",
                       help="Enable severity-aware training")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = EnhancedTrainingPipeline(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    pipeline.train()

if __name__ == "__main__":
    main()