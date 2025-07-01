"""
Training script for Soft Content Encoder using fairseq-extracted discrete units
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from models.soft_encoder import SoftContentEncoder
from training.trainer_utils import (
    Logger, EarlyStopping, AverageMeter, ModelCheckpoint,
    set_seed, get_lr_scheduler, save_config, count_parameters
)

class FairseqUnitsDataset(Dataset):
    """
    Dataset for training soft encoder with fairseq-extracted discrete units
    """
    
    def __init__(self, 
                 discrete_units_file: str,
                 max_sequence_length: int = 1000):
        
        self.max_sequence_length = max_sequence_length
        
        # Load discrete units from fairseq output
        self.units_data = self._load_fairseq_units(discrete_units_file)
        
        print(f"Loaded {len(self.units_data)} unit sequences")
        
    def _load_fairseq_units(self, units_file: str) -> list:
        """
        Load discrete units from fairseq output file
        
        Format: Each line contains space-separated integers representing unit sequence
        """
        units_data = []
        
        with open(units_file, 'r') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse unit sequence
                    units = [int(x) for x in line.split()]
                    
                    # Skip very short or very long sequences
                    if len(units) < 10 or len(units) > self.max_sequence_length:
                        continue
                    
                    units_data.append(units)
                    
                except ValueError as e:
                    print(f"Warning: Could not parse line {line_idx}: {e}")
                    continue
        
        return units_data
    
    def __len__(self):
        return len(self.units_data)
    
    def __getitem__(self, idx):
        units = self.units_data[idx]
        
        # Convert to tensor
        units_tensor = torch.LongTensor(units)
        
        return {
            "discrete_units": units_tensor
        }

def collate_fn_soft_encoder(batch):
    """Collate function for soft encoder dataset"""
    unit_sequences = [item["discrete_units"] for item in batch]
    
    # Pad to same length
    max_len = max(seq.size(0) for seq in unit_sequences)
    batch_size = len(batch)
    
    padded_units = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len)
    
    for i, seq in enumerate(unit_sequences):
        length = seq.size(0)
        padded_units[i, :length] = seq
        mask[i, :length] = 1
    
    return {
        "discrete_units": padded_units,
        "mask": mask
    }

class SoftEncoderTrainer:
    """Trainer for Soft Content Encoder with fairseq units"""
    
    def __init__(self, config: ModelConfig, training_config: TrainingConfig):
        self.config = config
        self.training_config = training_config
        self.device = torch.device(config.device)
        
        # Set random seed
        set_seed(training_config.seed)
        
        # Initialize model
        self.model = SoftContentEncoder(config.soft_encoder)
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=training_config.soft_encoder.learning_rate,
            weight_decay=training_config.soft_encoder.weight_decay,
            betas=(training_config.soft_encoder.adam_beta1, training_config.soft_encoder.adam_beta2)
        )
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # Initialize utilities
        self.logger = Logger(
            log_dir=config.log_dir,
            experiment_name=training_config.get_experiment_name("soft_encoder"),
            use_tensorboard=training_config.log_to_tensorboard,
            use_wandb=training_config.log_to_wandb,
            wandb_project=training_config.project_name
        )
        
        self.early_stopping = EarlyStopping(
            patience=10,
            min_delta=1e-4,
            mode="min"
        )
        
        self.checkpoint = ModelCheckpoint(
            checkpoint_dir=config.checkpoint_dir + "/soft_encoder",
            model_name="soft_encoder",
            save_top_k=training_config.save_top_k,
            monitor_metric=training_config.monitor_metric,
            mode=training_config.mode
        )
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        # Log model info
        param_count = count_parameters(self.model)
        self.logger.info(f"Soft Encoder parameters: {param_count:,}")
        
        # Save configs
        save_config(config, f"{config.log_dir}/model_config.json")
        save_config(training_config, f"{config.log_dir}/training_config.json")
    
    def _create_dataloaders(self):
        """Create train and validation dataloaders"""
        # Look for fairseq units file
        units_file = Path(self.config.data_dir) / "units" / "discrete_units.txt"
        
        if not units_file.exists():
            raise FileNotFoundError(f"Fairseq units file not found: {units_file}")
        
        # Create dataset
        full_dataset = FairseqUnitsDataset(str(units_file))
        
        # Split into train/val (80/20)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.soft_encoder.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn_soft_encoder,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config.soft_encoder.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn_soft_encoder,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self) -> dict:
        """Train for one epoch"""
        self.model.train()
        
        # Metrics
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}/{self.training_config.soft_encoder.epochs}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            discrete_units = batch["discrete_units"].to(self.device)
            mask = batch["mask"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            try:
                # Get soft units
                soft_units = self.model(discrete_units, mask)
                
                # Compute loss (self-supervised: predict discrete units from soft units)
                loss = self.model.compute_cross_entropy_loss(
                    soft_units, discrete_units, mask
                )
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.soft_encoder.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                
                # Compute accuracy
                predicted_units = torch.argmax(soft_units, dim=-1)
                correct = (predicted_units == discrete_units) * mask
                accuracy = correct.sum().float() / mask.sum()
                
                # Update metrics
                batch_size = discrete_units.size(0)
                loss_meter.update(loss.item(), batch_size)
                accuracy_meter.update(accuracy.item(), batch_size)
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss_meter.avg:.4f}",
                    "acc": f"{accuracy_meter.avg:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # Log training metrics
                if self.global_step % self.training_config.soft_encoder.logging_steps == 0:
                    self.logger.log_scalars({
                        "train/loss": loss_meter.avg,
                        "train/accuracy": accuracy_meter.avg,
                        "train/learning_rate": self.optimizer.param_groups[0]['lr']
                    }, self.global_step)
                
                self.global_step += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.logger.info(f"OOM at step {self.global_step}, skipping batch")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        return {
            "train_loss": loss_meter.avg,
            "train_accuracy": accuracy_meter.avg
        }
    
    def validate(self) -> dict:
        """Validate model"""
        self.model.eval()
        
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                discrete_units = batch["discrete_units"].to(self.device)
                mask = batch["mask"].to(self.device)
                
                # Forward pass
                soft_units = self.model(discrete_units, mask)
                
                # Compute loss
                loss = self.model.compute_cross_entropy_loss(
                    soft_units, discrete_units, mask
                )
                
                # Compute accuracy
                predicted_units = torch.argmax(soft_units, dim=-1)
                correct = (predicted_units == discrete_units) * mask
                accuracy = correct.sum().float() / mask.sum()
                
                # Update metrics
                batch_size = discrete_units.size(0)
                loss_meter.update(loss.item(), batch_size)
                accuracy_meter.update(accuracy.item(), batch_size)
        
        return {
            "val_loss": loss_meter.avg,
            "val_accuracy": accuracy_meter.avg
        }
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting Soft Encoder training...")
        self.logger.info(f"Training on {len(self.train_loader)} batches")
        self.logger.info(f"Validation on {len(self.val_loader)} batches")
        
        start_time = time.time()
        best_val_loss = float('inf')
        
        for epoch in range(self.training_config.soft_encoder.epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            if epoch % self.training_config.soft_encoder.eval_steps == 0:
                val_metrics = self.validate()
                
                # Log metrics
                all_metrics = {**train_metrics, **val_metrics}
                self.logger.log_scalars(all_metrics, self.global_step)
                
                # Check for best model
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    self.logger.info(f"New best validation loss: {best_val_loss:.4f}")
                
                # Save checkpoint
                self.checkpoint.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=None,
                    epoch=epoch,
                    step=self.global_step,
                    metrics=val_metrics
                )
                
                # Early stopping
                if self.early_stopping(val_metrics["val_loss"]):
                    self.logger.info("Early stopping triggered")
                    break
                
                # Log epoch summary
                elapsed_time = time.time() - start_time
                self.logger.info(
                    f"Epoch {epoch}: "
                    f"train_loss={train_metrics['train_loss']:.4f}, "
                    f"val_loss={val_metrics['val_loss']:.4f}, "
                    f"train_acc={train_metrics['train_accuracy']:.4f}, "
                    f"val_acc={val_metrics['val_accuracy']:.4f}, "
                    f"time={elapsed_time:.1f}s"
                )
        
        # Training completed
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.1f}s")
        self.logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        # Close logger
        self.logger.close()

def main():
    """Main training function"""
    # Load configs
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Create directories
    model_config.create_dirs()
    
    # Check if fairseq units exist
    units_file = Path(model_config.data_dir) / "units" / "discrete_units.txt"
    if not units_file.exists():
        print(f"Error: Fairseq units file not found: {units_file}")
        print("Please run: python scripts/extract_units_fairseq.py first")
        return
    
    # Initialize trainer
    trainer = SoftEncoderTrainer(model_config, training_config)
    
    # Train model
    trainer.train()

if __name__ == "__main__":
    main()