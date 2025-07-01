"""
Training script for HiFi-GAN Soft Unit Vocoder
Synthesizes waveforms from soft units
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from pathlib import Path
import torchaudio

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from models.hifigan_soft import SoftHiFiGAN, SoftHiFiGANLoss
from data.audio_utils import AudioProcessor
from training.trainer_utils import (
    Logger, EarlyStopping, AverageMeter, ModelCheckpoint,
    set_seed, get_lr_scheduler, save_config, count_parameters
)

class SoftUnitAudioDataset(torch.utils.data.Dataset):
    """Dataset for training HiFi-GAN with soft units"""
    
    def __init__(self,
                 soft_units_dir: str,
                 audio_dir: str,
                 split: str = "train",
                 audio_config=None,
                 max_audio_length: float = 4.0):
        
        self.soft_units_dir = Path(soft_units_dir)
        self.audio_dir = Path(audio_dir)
        self.split = split
        self.max_audio_length = max_audio_length
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor() if audio_config is None else AudioProcessor(**audio_config.__dict__)
        
        # Get file pairs
        self.file_pairs = self._get_file_pairs()
        
    def _get_file_pairs(self):
        """Get pairs of soft unit files and corresponding audio files"""
        pairs = []
        
        # Look for soft unit files
        soft_unit_files = list(self.soft_units_dir.glob(f"{self.split}_*.pt"))
        
        for soft_file in soft_unit_files:
            # Extract identifier from filename
            identifier = soft_file.stem.replace(f"{self.split}_", "")
            
            # Find corresponding audio file
            audio_file = self.audio_dir / f"{identifier}.wav"
            
            if audio_file.exists():
                pairs.append((str(soft_file), str(audio_file)))
        
        return pairs
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        soft_file, audio_file = self.file_pairs[idx]
        
        # Load soft units
        soft_units = torch.load(soft_file)
        
        # Load audio
        audio = self.audio_processor.load_audio(audio_file)
        
        # Trim to max length
        max_samples = int(self.max_audio_length * self.audio_processor.sampling_rate)
        if len(audio) > max_samples:
            start = torch.randint(0, len(audio) - max_samples + 1, (1,)).item()
            audio = audio[start:start + max_samples]
            
            # Adjust soft units accordingly (assuming frame rate)
            frame_rate = self.audio_processor.hop_length
            start_frame = start // frame_rate
            end_frame = start_frame + (max_samples // frame_rate)
            soft_units = soft_units[start_frame:end_frame]
        
        # Ensure audio and soft units are aligned
        expected_frames = audio.size(0) // self.audio_processor.hop_length
        if soft_units.size(0) > expected_frames:
            soft_units = soft_units[:expected_frames]
        elif soft_units.size(0) < expected_frames:
            # Pad soft units
            padding = expected_frames - soft_units.size(0)
            soft_units = torch.cat([
                soft_units,
                torch.zeros(padding, soft_units.size(1))
            ])
        
        return {
            "soft_units": soft_units,
            "audio": audio.unsqueeze(0)  # Add channel dimension
        }

def collate_fn_hifigan(batch):
    """Collate function for HiFi-GAN dataset"""
    soft_units = [item["soft_units"] for item in batch]
    audios = [item["audio"] for item in batch]
    
    # Pad to same length
    max_frames = max(units.size(0) for units in soft_units)
    max_audio_length = max(audio.size(1) for audio in audios)
    
    padded_units = torch.zeros(len(batch), max_frames, soft_units[0].size(1))
    padded_audios = torch.zeros(len(batch), 1, max_audio_length)
    
    for i, (units, audio) in enumerate(zip(soft_units, audios)):
        padded_units[i, :units.size(0)] = units
        padded_audios[i, :, :audio.size(1)] = audio
    
    return {
        "soft_units": padded_units,
        "audio": padded_audios
    }

class HiFiGANTrainer:
    """Trainer for HiFi-GAN Soft Unit Vocoder"""
    
    def __init__(self, config: ModelConfig, training_config: TrainingConfig):
        self.config = config
        self.training_config = training_config
        self.device = torch.device(config.device)
        
        # Set random seed
        set_seed(training_config.seed)
        
        # Initialize model
        self.model = SoftHiFiGAN(config.hifigan)
        self.model.to(self.device)
        
        # Initialize optimizers (separate for generator and discriminator)
        self.optimizer_g = Adam(
            self.model.generator.parameters(),
            lr=training_config.hifigan.gen_learning_rate,
            weight_decay=training_config.hifigan.gen_weight_decay,
            betas=(training_config.hifigan.adam_beta1, training_config.hifigan.adam_beta2)
        )
        
        self.optimizer_d = Adam(
            self.model.discriminator.parameters(),
            lr=training_config.hifigan.disc_learning_rate,
            weight_decay=training_config.hifigan.disc_weight_decay,
            betas=(training_config.hifigan.adam_beta1, training_config.hifigan.adam_beta2)
        )
        
        # Initialize data loaders
        self.train_loader = self._create_dataloader("train")
        self.val_loader = self._create_dataloader("val")
        
        # Initialize schedulers
        self.scheduler_g = get_lr_scheduler(
            self.optimizer_g,
            training_config.hifigan.lr_scheduler,
            lr_decay=training_config.hifigan.lr_decay
        )
        
        self.scheduler_d = get_lr_scheduler(
            self.optimizer_d,
            training_config.hifigan.lr_scheduler,
            lr_decay=training_config.hifigan.lr_decay
        )
        
        # Initialize loss function
        self.loss_fn = SoftHiFiGANLoss(config.hifigan)
        
        # Initialize utilities
        self.logger = Logger(
            log_dir=config.log_dir,
            experiment_name=training_config.get_experiment_name("hifigan"),
            use_tensorboard=training_config.log_to_tensorboard,
            use_wandb=training_config.log_to_wandb,
            wandb_project=training_config.project_name
        )
        
        self.checkpoint = ModelCheckpoint(
            checkpoint_dir=config.checkpoint_dir + "/hifigan",
            model_name="hifigan",
            save_top_k=training_config.save_top_k,
            monitor_metric="val_loss",
            mode="min"
        )
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        # Training schedule
        self.gen_train_start_step = training_config.hifigan.gen_train_start_step
        self.disc_train_start_step = training_config.hifigan.disc_train_start_step
        
        # Log model info
        gen_params = count_parameters(self.model.generator)
        disc_params = count_parameters(self.model.discriminator)
        self.logger.info(f"Generator parameters: {gen_params:,}")
        self.logger.info(f"Discriminator parameters: {disc_params:,}")
        
        # Save configs
        save_config(config, f"{config.log_dir}/model_config.json")
        save_config(training_config, f"{config.log_dir}/training_config.json")
    
    def _create_dataloader(self, split: str) -> DataLoader:
        """Create data loader for given split"""
        dataset = SoftUnitAudioDataset(
            soft_units_dir=f"{self.config.data_dir}/soft_units",
            audio_dir=f"{self.config.data_dir}/healthy",
            split=split,
            audio_config=self.config.audio,
            max_audio_length=4.0
        )
        
        return DataLoader(
            dataset,
            batch_size=self.training_config.hifigan.batch_size,
            shuffle=(split == "train"),
            num_workers=self.config.num_workers,
            collate_fn=collate_fn_hifigan,
            pin_memory=True,
            drop_last=(split == "train")
        )
    
    def train_generator(self, batch: dict) -> dict:
        """Train generator for one step"""
        soft_units = batch["soft_units"].to(self.device)
        real_audio = batch["audio"].to(self.device)
        
        self.optimizer_g.zero_grad()
        
        # Generate fake audio
        fake_audio = self.model.generator(soft_units)
        
        # Discriminator outputs for fake audio
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.model.discriminator(real_audio, fake_audio)
        
        # Compute generator loss
        loss_dict = self.loss_fn.compute_generator_loss(
            real_audio=real_audio,
            fake_audio=fake_audio,
            disc_outputs_fake=y_d_gs,
            fmap_real=fmap_rs,
            fmap_fake=fmap_gs
        )
        
        total_loss = loss_dict["total_loss"]
        
        # Backward pass
        total_loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.generator.parameters(),
            self.training_config.hifigan.gen_max_grad_norm
        )
        
        # Optimizer step
        self.optimizer_g.step()
        
        return {
            "gen_total_loss": total_loss.item(),
            "gen_adv_loss": loss_dict["adv_loss"].item(),
            "gen_feat_loss": loss_dict["feat_loss"].item(),
            "gen_mel_loss": loss_dict["mel_loss"].item()
        }
    
    def train_discriminator(self, batch: dict) -> dict:
        """Train discriminator for one step"""
        soft_units = batch["soft_units"].to(self.device)
        real_audio = batch["audio"].to(self.device)
        
        self.optimizer_d.zero_grad()
        
        # Generate fake audio (detached from generator)
        with torch.no_grad():
            fake_audio = self.model.generator(soft_units)
        
        # Discriminator outputs
        y_d_rs, y_d_gs, _, _ = self.model.discriminator(real_audio, fake_audio.detach())
        
        # Compute discriminator loss
        disc_loss, r_losses, g_losses = self.loss_fn.discriminator_loss(y_d_rs, y_d_gs)
        
        # Backward pass
        disc_loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.discriminator.parameters(),
            self.training_config.hifigan.disc_max_grad_norm
        )
        
        # Optimizer step
        self.optimizer_d.step()
        
        return {
            "disc_loss": disc_loss.item(),
            "disc_real_loss": np.mean(r_losses),
            "disc_fake_loss": np.mean(g_losses)
        }
    
    def train_epoch(self) -> dict:
        """Train for one epoch"""
        self.model.train()
        
        # Metrics
        gen_loss_meter = AverageMeter()
        disc_loss_meter = AverageMeter()
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}/{self.training_config.hifigan.epochs}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Check max steps
            if (self.training_config.hifigan.max_steps > 0 and 
                self.global_step >= self.training_config.hifigan.max_steps):
                break
            
            try:
                # Train generator
                if self.global_step >= self.gen_train_start_step:
                    gen_metrics = self.train_generator(batch)
                    gen_loss_meter.update(gen_metrics["gen_total_loss"], batch["audio"].size(0))
                else:
                    gen_metrics = {"gen_total_loss": 0.0}
                
                # Train discriminator
                if self.global_step >= self.disc_train_start_step:
                    disc_metrics = self.train_discriminator(batch)
                    disc_loss_meter.update(disc_metrics["disc_loss"], batch["audio"].size(0))
                else:
                    disc_metrics = {"disc_loss": 0.0}
                
                # Update schedulers
                if self.scheduler_g:
                    self.scheduler_g.step()
                if self.scheduler_d:
                    self.scheduler_d.step()
                
                # Update progress bar
                progress_bar.set_postfix({
                    "gen_loss": f"{gen_loss_meter.avg:.4f}",
                    "disc_loss": f"{disc_loss_meter.avg:.4f}",
                    "lr_g": f"{self.optimizer_g.param_groups[0]['lr']:.2e}",
                    "lr_d": f"{self.optimizer_d.param_groups[0]['lr']:.2e}"
                })
                
                # Log training metrics
                if self.global_step % self.training_config.hifigan.logging_steps == 0:
                    log_dict = {
                        "train/gen_learning_rate": self.optimizer_g.param_groups[0]['lr'],
                        "train/disc_learning_rate": self.optimizer_d.param_groups[0]['lr']
                    }
                    
                    if self.global_step >= self.gen_train_start_step:
                        log_dict.update({f"train/{k}": v for k, v in gen_metrics.items()})
                    
                    if self.global_step >= self.disc_train_start_step:
                        log_dict.update({f"train/{k}": v for k, v in disc_metrics.items()})
                    
                    self.logger.log_scalars(log_dict, self.global_step)
                
                # Generate audio samples
                if self.global_step % self.training_config.hifigan.generate_audio_every == 0:
                    self.generate_audio_samples(batch)
                
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
            "train_gen_loss": gen_loss_meter.avg,
            "train_disc_loss": disc_loss_meter.avg
        }
    
    def validate(self) -> dict:
        """Validate model"""
        self.model.eval()
        
        gen_loss_meter = AverageMeter()
        disc_loss_meter = AverageMeter()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                soft_units = batch["soft_units"].to(self.device)
                real_audio = batch["audio"].to(self.device)
                
                # Generate fake audio
                fake_audio = self.model.generator(soft_units)
                
                # Discriminator outputs
                y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.model.discriminator(real_audio, fake_audio)
                
                # Compute losses
                gen_loss_dict = self.loss_fn.compute_generator_loss(
                    real_audio=real_audio,
                    fake_audio=fake_audio,
                    disc_outputs_fake=y_d_gs,
                    fmap_real=fmap_rs,
                    fmap_fake=fmap_gs
                )
                
                disc_loss, _, _ = self.loss_fn.discriminator_loss(y_d_rs, y_d_gs)
                
                # Update metrics
                batch_size = real_audio.size(0)
                gen_loss_meter.update(gen_loss_dict["total_loss"].item(), batch_size)
                disc_loss_meter.update(disc_loss.item(), batch_size)
        
        return {
            "val_gen_loss": gen_loss_meter.avg,
            "val_disc_loss": disc_loss_meter.avg,
            "val_loss": gen_loss_meter.avg + disc_loss_meter.avg  # Combined for checkpointing
        }
    
    def generate_audio_samples(self, batch: dict):
        """Generate and log audio samples"""
        if self.global_step < self.gen_train_start_step:
            return
        
        self.model.eval()
        
        with torch.no_grad():
            soft_units = batch["soft_units"][:self.training_config.hifigan.num_audio_samples].to(self.device)
            real_audio = batch["audio"][:self.training_config.hifigan.num_audio_samples].to(self.device)
            
            # Generate fake audio
            fake_audio = self.model.generator(soft_units)
            
            # Log audio samples
            for i in range(min(soft_units.size(0), 3)):  # Log first 3 samples
                real_np = real_audio[i].cpu().numpy().flatten()
                fake_np = fake_audio[i].cpu().numpy().flatten()
                
                self.logger.log_audio(
                    f"audio_real/{i}",
                    real_np,
                    self.global_step,
                    self.config.audio.sampling_rate
                )
                
                self.logger.log_audio(
                    f"audio_fake/{i}",
                    fake_np,
                    self.global_step,
                    self.config.audio.sampling_rate
                )
        
        self.model.train()
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting HiFi-GAN training...")
        self.logger.info(f"Training on {len(self.train_loader)} batches")
        self.logger.info(f"Validation on {len(self.val_loader)} batches")
        
        start_time = time.time()
        best_val_loss = float('inf')
        
        max_epochs = self.training_config.hifigan.epochs
        if self.training_config.hifigan.max_steps > 0:
            max_epochs = min(max_epochs, 
                           self.training_config.hifigan.max_steps // len(self.train_loader) + 1)
        
        for epoch in range(max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Check max steps
            if (self.training_config.hifigan.max_steps > 0 and 
                self.global_step >= self.training_config.hifigan.max_steps):
                self.logger.info(f"Reached max steps: {self.training_config.hifigan.max_steps}")
                break
            
            # Validate
            if epoch % self.training_config.hifigan.eval_steps == 0:
                val_metrics = self.validate()
                
                # Log metrics
                all_metrics = {**train_metrics, **val_metrics}
                self.logger.log_scalars(all_metrics, self.global_step)
                
                # Check for best model
                val_loss = val_metrics["val_loss"]
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.logger.info(f"New best validation loss: {best_val_loss:.4f}")
                
                # Save checkpoint
                self.checkpoint.save(
                    model=self.model,
                    optimizer=self.optimizer_g,  # Save generator optimizer
                    scheduler=self.scheduler_g,
                    epoch=epoch,
                    step=self.global_step,
                    metrics=val_metrics,
                    extra_state={
                        "optimizer_d_state_dict": self.optimizer_d.state_dict(),
                        "scheduler_d_state_dict": self.scheduler_d.state_dict() if self.scheduler_d else None
                    }
                )
                
                # Log epoch summary
                elapsed_time = time.time() - start_time
                self.logger.info(
                    f"Epoch {epoch}: "
                    f"train_gen_loss={train_metrics['train_gen_loss']:.4f}, "
                    f"train_disc_loss={train_metrics['train_disc_loss']:.4f}, "
                    f"val_loss={val_metrics['val_loss']:.4f}, "
                    f"time={elapsed_time:.1f}s"
                )
        
        # Training completed
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.1f}s")
        self.logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        # Remove weight norm for inference
        self.model.generator.remove_weight_norm()
        
        # Close logger
        self.logger.close()

def main():
    """Main training function"""
    # Load configs
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Create directories
    model_config.create_dirs()
    
    # Initialize trainer
    trainer = HiFiGANTrainer(model_config, training_config)
    
    # Train model
    trainer.train()

if __name__ == "__main__":
    main()