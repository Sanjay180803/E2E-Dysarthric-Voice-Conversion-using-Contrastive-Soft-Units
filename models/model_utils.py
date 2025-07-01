"""
Model utilities for dysarthric voice conversion
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Union, List, Tuple, Any
from pathlib import Path
import json
import pickle
import warnings
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model
    
    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def get_model_size_mb(model: nn.Module) -> float:
    """
    Get model size in megabytes
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def save_model_checkpoint(model: nn.Module,
                         optimizer: Optional[torch.optim.Optimizer] = None,
                         scheduler: Optional[Any] = None,
                         epoch: int = 0,
                         step: int = 0,
                         metrics: Optional[Dict] = None,
                         save_path: Union[str, Path] = None,
                         model_name: str = "model",
                         extra_state: Optional[Dict] = None) -> str:
    """
    Save model checkpoint with comprehensive state information
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer state (optional)
        scheduler: Learning rate scheduler state (optional)
        epoch: Training epoch
        step: Training step
        metrics: Training/validation metrics
        save_path: Path to save checkpoint
        model_name: Name of the model
        extra_state: Additional state to save
        
    Returns:
        Path where checkpoint was saved
    """
    if save_path is None:
        save_path = f"{model_name}_checkpoint_epoch_{epoch}.pt"
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint dictionary
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "model_name": model_name,
        "model_config": getattr(model, 'config', None),
        "model_size_mb": get_model_size_mb(model),
        "num_parameters": count_parameters(model),
        "torch_version": torch.__version__,
    }
    
    # Add optimizer state
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        checkpoint["optimizer_type"] = type(optimizer).__name__
    
    # Add scheduler state
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        checkpoint["scheduler_type"] = type(scheduler).__name__
    
    # Add metrics
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    # Add extra state
    if extra_state is not None:
        checkpoint.update(extra_state)
    
    # Save checkpoint
    torch.save(checkpoint, save_path)
    
    logger.info(f"Saved checkpoint: {save_path}")
    logger.info(f"Model size: {checkpoint['model_size_mb']:.2f} MB")
    logger.info(f"Parameters: {checkpoint['num_parameters']:,}")
    
    return str(save_path)

def load_model_checkpoint(model: nn.Module,
                         checkpoint_path: Union[str, Path],
                         optimizer: Optional[torch.optim.Optimizer] = None,
                         scheduler: Optional[Any] = None,
                         device: str = "cpu",
                         strict: bool = True) -> Dict:
    """
    Load model checkpoint with error handling
    
    Args:
        model: PyTorch model to load state into
        checkpoint_path: Path to checkpoint file
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to map tensors to
        strict: Whether to strictly enforce state dict matching
        
    Returns:
        Checkpoint information dictionary
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        logger.info("Model state loaded successfully")
    except Exception as e:
        if strict:
            raise RuntimeError(f"Failed to load model state: {e}")
        else:
            logger.warning(f"Model state loading issues (non-strict): {e}")
    
    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Optimizer state loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load optimizer state: {e}")
    
    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("Scheduler state loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load scheduler state: {e}")
    
    # Return checkpoint info
    info = {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "metrics": checkpoint.get("metrics", {}),
        "model_name": checkpoint.get("model_name", "unknown"),
        "torch_version": checkpoint.get("torch_version", "unknown")
    }
    
    logger.info(f"Loaded checkpoint from epoch {info['epoch']}, step {info['step']}")
    
    return info

def initialize_weights(model: nn.Module, init_type: str = "xavier_uniform") -> None:
    """
    Initialize model weights using specified method
    
    Args:
        model: PyTorch model
        init_type: Initialization method
    """
    def init_func(m):
        classname = m.__class__.__name__
        
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight.data)
            elif init_type == 'xavier_normal':
                nn.init.xavier_normal_(m.weight.data)
            elif init_type == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            elif init_type == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
                
        elif 'Linear' in classname:
            if init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight.data)
            elif init_type == 'xavier_normal':
                nn.init.xavier_normal_(m.weight.data)
            elif init_type == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight.data)
            elif init_type == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight.data)
            elif init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
                
        elif 'BatchNorm' in classname:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
    
    model.apply(init_func)
    logger.info(f"Initialized model weights using {init_type}")

def freeze_model(model: nn.Module, freeze_bn: bool = True) -> None:
    """
    Freeze model parameters
    
    Args:
        model: PyTorch model to freeze
        freeze_bn: Whether to freeze batch norm layers
    """
    for param in model.parameters():
        param.requires_grad = False
    
    if not freeze_bn:
        # Unfreeze batch norm parameters
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                for param in module.parameters():
                    param.requires_grad = True
    
    frozen_params = sum(1 for p in model.parameters() if not p.requires_grad)
    total_params = sum(1 for p in model.parameters())
    
    logger.info(f"Frozen {frozen_params}/{total_params} parameters")

def unfreeze_model(model: nn.Module) -> None:
    """
    Unfreeze all model parameters
    
    Args:
        model: PyTorch model to unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True
    
    logger.info("Unfrozen all model parameters")

def get_model_summary(model: nn.Module, input_size: Optional[Tuple] = None) -> Dict:
    """
    Get comprehensive model summary
    
    Args:
        model: PyTorch model
        input_size: Input tensor size for forward pass analysis
        
    Returns:
        Model summary dictionary
    """
    summary = {
        "model_name": model.__class__.__name__,
        "total_parameters": count_parameters(model, trainable_only=False),
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "model_size_mb": get_model_size_mb(model),
        "layers": []
    }
    
    # Layer information
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            layer_info = {
                "name": name,
                "type": module.__class__.__name__,
                "parameters": count_parameters(module, trainable_only=False)
            }
            
            # Add layer-specific information
            if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                layer_info["in_features"] = module.in_features
                layer_info["out_features"] = module.out_features
            elif hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
                layer_info["in_channels"] = module.in_channels
                layer_info["out_channels"] = module.out_channels
                if hasattr(module, 'kernel_size'):
                    layer_info["kernel_size"] = module.kernel_size
            
            summary["layers"].append(layer_info)
    
    return summary

def save_model_for_inference(model: nn.Module,
                            save_path: Union[str, Path],
                            config: Optional[Dict] = None,
                            metadata: Optional[Dict] = None) -> None:
    """
    Save model in inference-ready format
    
    Args:
        model: PyTorch model
        save_path: Path to save the model
        config: Model configuration
        metadata: Additional metadata
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set model to eval mode
    model.eval()
    
    # Create inference checkpoint
    inference_checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
        "model_config": config,
        "metadata": metadata,
        "torch_version": torch.__version__,
        "inference_ready": True
    }
    
    torch.save(inference_checkpoint, save_path)
    logger.info(f"Saved inference model: {save_path}")

def convert_checkpoint_format(old_checkpoint_path: Union[str, Path],
                            new_checkpoint_path: Union[str, Path],
                            format_mapping: Optional[Dict] = None) -> None:
    """
    Convert checkpoint format (e.g., for compatibility)
    
    Args:
        old_checkpoint_path: Path to old checkpoint
        new_checkpoint_path: Path for new checkpoint
        format_mapping: Mapping for state dict key conversion
    """
    checkpoint = torch.load(old_checkpoint_path, map_location='cpu')
    
    if format_mapping:
        # Apply format mapping
        old_state_dict = checkpoint["model_state_dict"]
        new_state_dict = OrderedDict()
        
        for old_key, tensor in old_state_dict.items():
            new_key = format_mapping.get(old_key, old_key)
            new_state_dict[new_key] = tensor
        
        checkpoint["model_state_dict"] = new_state_dict
    
    torch.save(checkpoint, new_checkpoint_path)
    logger.info(f"Converted checkpoint format: {old_checkpoint_path} -> {new_checkpoint_path}")

def validate_checkpoint_compatibility(checkpoint_path: Union[str, Path],
                                    model: nn.Module) -> Dict:
    """
    Validate if checkpoint is compatible with model
    
    Args:
        checkpoint_path: Path to checkpoint
        model: PyTorch model
        
    Returns:
        Compatibility report
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if "model_state_dict" not in checkpoint:
        return {"compatible": False, "reason": "No model_state_dict found"}
    
    checkpoint_keys = set(checkpoint["model_state_dict"].keys())
    model_keys = set(model.state_dict().keys())
    
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    
    compatible = len(missing_keys) == 0 and len(unexpected_keys) == 0
    
    return {
        "compatible": compatible,
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
        "total_checkpoint_keys": len(checkpoint_keys),
        "total_model_keys": len(model_keys)
    }

class ModelRegistry:
    """
    Registry for managing multiple models
    """
    
    def __init__(self):
        self.models = {}
        self.metadata = {}
    
    def register_model(self, 
                      name: str, 
                      model: nn.Module,
                      config: Optional[Dict] = None,
                      metadata: Optional[Dict] = None) -> None:
        """Register a model"""
        self.models[name] = model
        self.metadata[name] = {
            "config": config,
            "metadata": metadata,
            "parameters": count_parameters(model),
            "size_mb": get_model_size_mb(model)
        }
        
        logger.info(f"Registered model: {name}")
    
    def get_model(self, name: str) -> nn.Module:
        """Get registered model"""
        if name not in self.models:
            raise KeyError(f"Model '{name}' not found in registry")
        return self.models[name]
    
    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self.models.keys())
    
    def get_model_info(self, name: str) -> Dict:
        """Get model information"""
        if name not in self.metadata:
            raise KeyError(f"Model '{name}' not found in registry")
        return self.metadata[name]
    
    def save_registry(self, save_dir: Union[str, Path]) -> None:
        """Save all models in registry"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        registry_info = {}
        
        for name, model in self.models.items():
            model_path = save_dir / f"{name}.pt"
            save_model_for_inference(
                model, 
                model_path,
                config=self.metadata[name]["config"],
                metadata=self.metadata[name]["metadata"]
            )
            registry_info[name] = {
                "path": str(model_path),
                "info": self.metadata[name]
            }
        
        # Save registry metadata
        with open(save_dir / "registry.json", 'w') as f:
            json.dump(registry_info, f, indent=2)
        
        logger.info(f"Saved model registry to: {save_dir}")

def analyze_model_complexity(model: nn.Module, 
                            input_tensor: torch.Tensor) -> Dict:
    """
    Analyze model computational complexity
    
    Args:
        model: PyTorch model
        input_tensor: Sample input tensor
        
    Returns:
        Complexity analysis
    """
    model.eval()
    
    # Count FLOPs (simplified)
    total_params = count_parameters(model)
    
    # Measure inference time
    with torch.no_grad():
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if torch.cuda.is_available():
            start_time.record()
            _ = model(input_tensor)
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time)  # milliseconds
        else:
            import time
            start = time.time()
            _ = model(input_tensor)
            end = time.time()
            inference_time = (end - start) * 1000  # convert to milliseconds
    
    return {
        "total_parameters": total_params,
        "model_size_mb": get_model_size_mb(model),
        "inference_time_ms": inference_time,
        "input_shape": list(input_tensor.shape),
        "memory_efficient": total_params < 50_000_000  # < 50M parameters
    }