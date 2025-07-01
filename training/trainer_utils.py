"""
Training utilities for Dysarthric Voice Conversion
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Any
import logging
from pathlib import Path
import json
import random
from tensorboardX import SummaryWriter
import wandb
 

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 1e-4,
                 mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif self._is_better(val_score):
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, score: float) -> bool:
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger:
    """Training logger with TensorBoard and Weights & Biases support"""
    
    def __init__(self, 
                 log_dir: str,
                 experiment_name: str,
                 use_tensorboard: bool = True,
                 use_wandb: bool = False,
                 wandb_project: Optional[str] = None):
        
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(str(self.log_dir / experiment_name))
        
        # Initialize Weights & Biases
        if self.use_wandb:
            wandb.init(
                project=wandb_project or "dysarthric_voice_conversion",
                name=experiment_name,
                dir=str(self.log_dir)
            )
        
        # Setup Python logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup Python logging"""
        log_file = self.log_dir / f"{self.experiment_name}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(self.experiment_name)
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value"""
        if self.use_tensorboard:
            self.tb_writer.add_scalar(tag, value, step)
        
        if self.use_wandb:
            wandb.log({tag: value}, step=step)
    
    def log_scalars(self, tag_value_dict: Dict[str, float], step: int):
        """Log multiple scalar values"""
        for tag, value in tag_value_dict.items():
            self.log_scalar(tag, value, step)
    
    def log_audio(self, tag: str, audio: np.ndarray, step: int, sample_rate: int = 16000):
        """Log audio sample"""
        if self.use_tensorboard:
            self.tb_writer.add_audio(tag, audio, step, sample_rate=sample_rate)
        
        if self.use_wandb:
            wandb.log({tag: wandb.Audio(audio, sample_rate=sample_rate)}, step=step)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text"""
        if self.use_tensorboard:
            self.tb_writer.add_text(tag, text, step)
        
        if self.use_wandb:
            wandb.log({tag: text}, step=step)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def close(self):
        """Close logger"""
        if self.use_tensorboard:
            self.tb_writer.close()
        
        if self.use_wandb:
            wandb.finish()

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_lr_scheduler(optimizer: torch.optim.Optimizer,
                    scheduler_type: str,
                    **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler"""
    
    if scheduler_type == "linear_warmup":
        from transformers import get_linear_schedule_with_warmup
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=kwargs.get("warmup_steps", 1000),
            num_training_steps=kwargs.get("total_steps", 10000)
        )
    
    elif scheduler_type == "inverse_sqrt":
        return InverseSqrtScheduler(
            optimizer,
            warmup_steps=kwargs.get("warmup_steps", 10000)
        )
    
    elif scheduler_type == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get("lr_decay", 0.999)
        )
    
    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("T_max", 1000)
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

class InverseSqrtScheduler:
    """Inverse square root learning rate scheduler"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int = 10000):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr_mult = self.step_count / self.warmup_steps
        else:
            # Inverse square root decay
            lr_mult = (self.warmup_steps / self.step_count) ** 0.5
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[i] * lr_mult

def save_config(config: Any, save_path: str):
    """Save configuration to JSON file"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert config to dictionary if it's a dataclass
    if hasattr(config, '__dict__'):
        config_dict = {}
        for key, value in config.__dict__.items():
            if hasattr(value, '__dict__'):
                config_dict[key] = value.__dict__
            else:
                config_dict[key] = value
    else:
        config_dict = config
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)

def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

class ModelCheckpoint:
    """Model checkpointing utility"""
    
    def __init__(self,
                 checkpoint_dir: str,
                 model_name: str,
                 save_top_k: int = 3,
                 monitor_metric: str = "val_loss",
                 mode: str = "min"):
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.save_top_k = save_top_k
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoints = []
    
    def save(self,
             model: nn.Module,
             optimizer: torch.optim.Optimizer,
             scheduler: Optional[Any],
             epoch: int,
             step: int,
             metrics: Dict[str, float],
             extra_state: Optional[Dict] = None):
        """Save model checkpoint"""
        
        metric_value = metrics.get(self.monitor_metric, 0.0)
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
            "metric_value": metric_value
        }
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        if extra_state is not None:
            checkpoint.update(extra_state)
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / f"{self.model_name}_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Check if this is a best checkpoint
        should_save = False
        if len(self.best_checkpoints) < self.save_top_k:
            should_save = True
        else:
            if self.mode == "min":
                worst_metric = max(self.best_checkpoints, key=lambda x: x[1])[1]
                should_save = metric_value < worst_metric
            else:
                worst_metric = min(self.best_checkpoints, key=lambda x: x[1])[1]
                should_save = metric_value > worst_metric
        
        if should_save:
            # Save best checkpoint
            best_path = self.checkpoint_dir / f"{self.model_name}_best_epoch_{epoch}.pt"
            torch.save(checkpoint, best_path)
            
            # Update best checkpoints list
            self.best_checkpoints.append((str(best_path), metric_value))
            
            # Remove worst checkpoint if exceeding save_top_k
            if len(self.best_checkpoints) > self.save_top_k:
                if self.mode == "min":
                    worst_checkpoint = max(self.best_checkpoints, key=lambda x: x[1])
                else:
                    worst_checkpoint = min(self.best_checkpoints, key=lambda x: x[1])
                
                # Remove worst checkpoint file
                if os.path.exists(worst_checkpoint[0]):
                    os.remove(worst_checkpoint[0])
                
                self.best_checkpoints.remove(worst_checkpoint)
        
        return str(latest_path)
    
    def load_latest(self, model: nn.Module, device: str = "cpu") -> Dict:
        """Load latest checkpoint"""
        latest_path = self.checkpoint_dir / f"{self.model_name}_latest.pt"
        
        if not latest_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {latest_path}")
        
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return checkpoint
    
    def load_best(self, model: nn.Module, device: str = "cpu") -> Dict:
        """Load best checkpoint"""
        if not self.best_checkpoints:
            raise ValueError("No best checkpoints available")
        
        if self.mode == "min":
            best_checkpoint_path = min(self.best_checkpoints, key=lambda x: x[1])[0]
        else:
            best_checkpoint_path = max(self.best_checkpoints, key=lambda x: x[1])[0]
        
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return checkpoint

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_time(seconds: float) -> str:
    """Format time in seconds to readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

class GradientClipping:
    """Gradient clipping utility"""
    
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
    
    def __call__(self, model: nn.Module) -> float:
        """Clip gradients and return gradient norm"""
        return torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)