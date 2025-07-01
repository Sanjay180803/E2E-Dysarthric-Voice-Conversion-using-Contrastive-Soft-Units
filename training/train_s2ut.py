"""
Training script for Speech-to-Unit Translation (S2UT) model
Core component for dysarthric voice conversion
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from models.s2ut_model import S2UTModel
from data.dataset import DysarthricS2UTDataset, collate_fn_s2ut
from data.audio_utils import AudioProcessor
from training.trainer_utils import (
    Logger, EarlyStopping, AverageMeter, ModelCheckpoint,
    set_seed, get_lr_scheduler, save_config, count_parameters
)

class S2UTTrainer:
    """Trainer for Speech-to-Unit Translation model"""
    
    def __init__(self, config: ModelConfig, training_config: TrainingConfig):
        self.config = config
        self.training_config = training_config
        self.device = torch.device(config.device)
        
        # Set random seed
        set_seed(training_config.seed)
        
        # Initialize model
        self.model = S2UTModel(config.s2ut)
        self.model.to(self.device)
        
        # Load soft encoder if available
        self.soft_encoder = self._load_soft_encoder()
        if self.soft_encoder is not None:
            self.soft_encoder.to(self.device)
            self.soft_encoder.eval()
        
        # Initialize optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=training_config.s2ut.learning_rate,
            weight_decay=training_config.s2ut.weight_decay,
            betas=(training_config.s2ut.adam_beta1, training_config.s2ut.adam_beta2),
            eps=training_config.s2ut.adam_epsilon
        )
        
        # Initialize data loaders
        self.train_loader = self._create_dataloader("train")
        self.val_loader = self._create_dataloader("val")
        
        # Initialize scheduler
        if training_config.s2ut.max_steps > 0:
            total_steps = training_config.s2ut.max_steps
        else:
            total_steps = len(self.train_loader) * training_config.s2ut.epochs
        
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            training_config.s2ut.lr_scheduler,
            warmup_steps=training_config.s2ut.warmup_steps,
            total_steps=total_steps
        )
        
        # Initialize utilities
        self.logger = Logger(
            log_dir=config.log_dir,
            experiment_name=training_config.get_experiment_name("s2ut"),
            use_tensorboard=training_config.log_to_tensorboard,
            use_wandb=training_config.log_to_wandb,
            wandb_project=training_config.project_name
        )
        
        self.early_stopping = EarlyStopping(
            patience=training_config.s2ut.patience,
            min_delta=training_config.s2ut.min_delta,
            mode="min"
        )
        
        self.checkpoint = ModelCheckpoint(
            checkpoint_dir=config.checkpoint_dir + "/s2ut",
            model_name="s2ut",
            save_top_k=training_config.save_top_k,
            monitor_metric=training_config.monitor_metric,
            mode=training_config.mode
        )
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        # Audio processor for SpecAugment
        self.audio_processor = AudioProcessor(**config.audio.__dict__)
        
        # Log model info
        param_count = count_parameters(self.model)
        self.logger.info(f"S2UT Model parameters: {param_count:,}")
        
        # Save configs
        save_config(config, f"{config.log_dir}/model_config.json")
        save_config(training_config, f"{config.log_dir}/training_config.json")
    
    def _load_soft_encoder(self):
        """Load soft encoder if available"""
        try:
            from models.soft_encoder import SoftContentEncoder
            
            # Look for soft encoder checkpoint
            soft_encoder_path = Path(self.config.checkpoint_dir) / "soft_encoder" / "soft_encoder_latest.pt"
            
            if soft_encoder_path.exists():
                print(f"Loading soft encoder from {soft_encoder_path}")
                
                soft_encoder = SoftContentEncoder(self.config.soft_encoder)
                checkpoint = torch.load(soft_encoder_path, map_location=self.device)
                soft_encoder.load_state_dict(checkpoint["model_state_dict"])
                
                return soft_encoder
            else:
                print("Soft encoder not found, will use one-hot encoding")
                return None
                
        except Exception as e:
            print(f"Could not load soft encoder: {e}")
            return None
    
    def _create_dataloader(self, split: str) -> DataLoader:
        """Create data loader for given split"""
        fairseq_units_file = f"{self.config.data_dir}/units/discrete_units.txt"
        
        if not Path(fairseq_units_file).exists():
            raise FileNotFoundError(f"Fairseq units file not found: {fairseq_units_file}")
        
        dataset = DysarthricS2UTDataset(
            dysarthric_data_dir=f"{self.config.data_dir}/dysarthric",
            fairseq_units_file=fairseq_units_file,
            soft_units_dir=f"{self.config.data_dir}/soft_units",
            split=split,
            audio_config=self.config.audio,
            severity_filter=["mild", "moderate"],
            apply_spec_augment=(split == "train")
        )
        
        return DataLoader(
            dataset,
            batch_size=self.training_config.s2ut.batch_size,
            shuffle=(split == "train"),
            num_workers=self.config.num_workers,
            collate_fn=collate_fn_s2ut,
            pin_memory=True,
            drop_last=(split == "train")
        )
    
    def apply_spec_augment(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to mel spectrogram"""
        if self.training_config.s2ut.freq_mask_N > 0 or self.training_config.s2ut.time_mask_N > 0:
            return self.audio_processor.apply_spec_augment(
                mel_spec,
                freq_mask_N=self.training_config.s2ut.freq_mask_N,
                freq_mask_F=self.training_config.s2ut.freq_mask_F,
                time_mask_N=self.training_config.s2ut.time_mask_N,
                time_mask_T=self.training_config.s2ut.time_mask_T,
                time_mask_p=self.training_config.s2ut.time_mask_p
            )
        return mel_spec
    
    def compute_loss(self, 
                    predicted_soft_units: torch.Tensor,
                    target_soft_units: torch.Tensor,
                    negative_soft_units: torch.Tensor,
                    mask: torch.Tensor) -> dict:
        """Compute total loss with cross-entropy and contrastive components"""
        
        losses = self.model.compute_total_loss(
            predicted_soft_units=predicted_soft_units,
            target_soft_units=target_soft_units,
            negative_soft_units=negative_soft_units,
            mask=mask
        )
        
        return losses
    
    def train_epoch(self) -> dict:
        """Train for one epoch"""
        self.model.train()
        
        # Metrics
        total_loss_meter = AverageMeter()
        ce_loss_meter = AverageMeter()
        contrastive_loss_meter = AverageMeter()
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}/{self.training_config.s2ut.epochs}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Check max steps
            if (self.training_config.s2ut.max_steps > 0 and 
                self.global_step >= self.training_config.s2ut.max_steps):
                break
            
            # Move to device
            dysarthric_mel = batch["dysarthric_mel"].to(self.device)
            target_units = batch["target_units"].to(self.device)  # Now discrete units
            negative_units = batch["negative_units"].to(self.device)  # Now discrete units
            mel_lengths = batch["mel_lengths"].to(self.device)
            unit_lengths = batch["unit_lengths"].to(self.device)
            
            # Convert discrete units to soft units using soft encoder
            if hasattr(self, 'soft_encoder') and self.soft_encoder is not None:
                with torch.no_grad():
                    target_soft_units = self.soft_encoder(target_units)
                    negative_soft_units = self.soft_encoder(negative_units)
            else:
                # Use one-hot encoding as fallback
                target_soft_units = F.one_hot(target_units, num_classes=1000).float()
                negative_soft_units = F.one_hot(negative_units, num_classes=1000).float()
            
            # Apply SpecAugment during training
            if self.model.training:
                for i in range(dysarthric_mel.size(0)):
                    dysarthric_mel[i] = self.apply_spec_augment(dysarthric_mel[i])
            
            # Create masks
            batch_size, max_mel_len = dysarthric_mel.shape[:2]
            max_unit_len = target_soft_units.size(1)
            
            src_mask = torch.arange(max_mel_len, device=self.device)[None, :] < mel_lengths[:, None]
            tgt_mask = torch.arange(max_unit_len, device=self.device)[None, :] < unit_lengths[:, None]
            
            # Forward pass
            self.optimizer.zero_grad()
            
            try:
                outputs = self.model(
                    dysarthric_mel=dysarthric_mel,
                    target_soft_units=target_soft_units,
                    src_mask=src_mask,
                    tgt_mask=tgt_mask
                )
                
                predicted_soft_units = outputs["predicted_soft_units"]
                
                # Compute losses
                losses = self.compute_loss(
                    predicted_soft_units=predicted_soft_units,
                    target_soft_units=target_soft_units,
                    negative_soft_units=negative_soft_units,
                    mask=tgt_mask
                )
                
                total_loss = losses["total_loss"]
                
                # Backward pass
                total_loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.s2ut.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                
                # Update metrics
                batch_size = dysarthric_mel.size(0)
                total_loss_meter.update(total_loss.item(), batch_size)
                ce_loss_meter.update(losses["ce_loss"].item(), batch_size)
                contrastive_loss_meter.update(losses["contrastive_loss"].item(), batch_size)
                
                # Update progress bar
                progress_bar.set_postfix({
                    "total_loss": f"{total_loss_meter.avg:.4f}",
                    "ce_loss": f"{ce_loss_meter.avg:.4f}",
                    "cont_loss": f"{contrastive_loss_meter.avg:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # Log training metrics
                if self.global_step % self.training_config.s2ut.logging_steps == 0:
                    self.logger.log_scalars({
                        "train/total_loss": total_loss_meter.avg,
                        "train/ce_loss": ce_loss_meter.avg,
                        "train/contrastive_loss": contrastive_loss_meter.avg,
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
            "train_total_loss": total_loss_meter.avg,
            "train_ce_loss": ce_loss_meter.avg,
            "train_contrastive_loss": contrastive_loss_meter.avg
        }
    
    def validate(self) -> dict:
        """Validate model"""
        self.model.eval()
        
        total_loss_meter = AverageMeter()
        ce_loss_meter = AverageMeter()
        contrastive_loss_meter = AverageMeter()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                # Move to device
                dysarthric_mel = batch["dysarthric_mel"].to(self.device)
                target_units = batch["target_units"].to(self.device)
                negative_units = batch["negative_units"].to(self.device)
                mel_lengths = batch["mel_lengths"].to(self.device)
                unit_lengths = batch["unit_lengths"].to(self.device)
                
                # Convert discrete units to soft units
                if hasattr(self, 'soft_encoder') and self.soft_encoder is not None:
                    target_soft_units = self.soft_encoder(target_units)
                    negative_soft_units = self.soft_encoder(negative_units)
                else:
                    target_soft_units = F.one_hot(target_units, num_classes=1000).float()
                    negative_soft_units = F.one_hot(negative_units, num_classes=1000).float()
                
                # Create masks
                batch_size, max_mel_len = dysarthric_mel.shape[:2]
                max_unit_len = target_soft_units.size(1)
                
                src_mask = torch.arange(max_mel_len, device=self.device)[None, :] < mel_lengths[:, None]
                tgt_mask = torch.arange(max_unit_len, device=self.device)[None, :] < unit_lengths[:, None]
                
                # Forward pass
                outputs = self.model(
                    dysarthric_mel=dysarthric_mel,
                    target_soft_units=target_soft_units,
                    src_mask=src_mask,
                    tgt_mask=tgt_mask
                )
                
                predicted_soft_units = outputs["predicted_soft_units"]
                
                # Compute losses
                losses = self.compute_loss(
                    predicted_soft_units=predicted_soft_units,
                    target_soft_units=target_soft_units,
                    negative_soft_units=negative_soft_units,
                    mask=tgt_mask
                )
                
                # Update metrics
                batch_size = dysarthric_mel.size(0)
                total_loss_meter.update(losses["total_loss"].item(), batch_size)
                ce_loss_meter.update(losses["ce_loss"].item(), batch_size)
                contrastive_loss_meter.update(losses["contrastive_loss"].item(), batch_size)
        
        return {
            "val_total_loss": total_loss_meter.avg,
            "val_ce_loss": ce_loss_meter.avg,
            "val_contrastive_loss": contrastive_loss_meter.avg
        }
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting S2UT training...")
        self.logger.info(f"Training on {len(self.train_loader)} batches")
        self.logger.info(f"Validation on {len(self.val_loader)} batches")
        
        start_time = time.time()
        best_val_loss = float('inf')
        
        max_epochs = self.training_config.s2ut.epochs
        if self.training_config.s2ut.max_steps > 0:
            max_epochs = min(max_epochs, 
                           self.training_config.s2ut.max_steps // len(self.train_loader) + 1)
        
        for epoch in range(max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Check max steps
            if (self.training_config.s2ut.max_steps > 0 and 
                self.global_step >= self.training_config.s2ut.max_steps):
                self.logger.info(f"Reached max steps: {self.training_config.s2ut.max_steps}")
                break
            
            # Validate
            if epoch % self.training_config.s2ut.eval_steps == 0:
                val_metrics = self.validate()
                
                # Log metrics
                all_metrics = {**train_metrics, **val_metrics}
                self.logger.log_scalars(all_metrics, self.global_step)
                
                # Check for best model
                val_loss = val_metrics["val_total_loss"]
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.logger.info(f"New best validation loss: {best_val_loss:.4f}")
                
                # Save checkpoint
                self.checkpoint.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    step=self.global_step,
                    metrics={"val_loss": val_loss}
                )
                
                # Early stopping
                if self.early_stopping(val_loss):
                    self.logger.info("Early stopping triggered")
                    break
                
                # Log epoch summary
                elapsed_time = time.time() - start_time
                self.logger.info(
                    f"Epoch {epoch}: "
                    f"train_loss={train_metrics['train_total_loss']:.4f}, "
                    f"val_loss={val_metrics['val_total_loss']:.4f}, "
                    f"time={elapsed_time:.1f}s"
                )
        
        # Training completed
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.1f}s")
        self.logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        # Close logger
        self.logger.close()
    
    def inference(self, 
                 dysarthric_mel: torch.Tensor,
                 src_mask: torch.Tensor = None) -> torch.Tensor:
        """Run inference on dysarthric speech"""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(
                dysarthric_mel=dysarthric_mel,
                target_soft_units=None,  # No teacher forcing
                src_mask=src_mask,
                tgt_mask=None
            )
            
            return outputs["predicted_soft_units"]

def main():
    """Main training function"""
    # Load configs
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Create directories
    model_config.create_dirs()
    
    # Initialize trainer
    trainer = S2UTTrainer(model_config, training_config)
    
    # Train model
    trainer.train()

if __name__ == "__main__":
    main()