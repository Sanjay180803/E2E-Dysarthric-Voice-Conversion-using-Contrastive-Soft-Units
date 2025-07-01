"""
HiFi-GAN Soft Unit Vocoder for synthesizing waveforms from soft units
Based on the paper: "Soft-unit HiFi-GAN vocoder"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import numpy as np
from typing import Dict, List, Optional, Tuple
import math
import torchaudio

def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Get padding size for 'same' convolution"""
    return int((kernel_size * dilation - dilation) / 2)

def init_weights(m, mean=0.0, std=0.01):
    """Initialize weights"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

class SoftUnitEmbedding(nn.Module):
    """
    Embedding layer for soft units
    Projects soft unit distributions to continuous embeddings
    """
    
    def __init__(self, 
                 soft_unit_dim: int = 1000,
                 embedding_dim: int = 256):
        super().__init__()
        self.soft_unit_dim = soft_unit_dim
        self.embedding_dim = embedding_dim
        
        # Learnable embedding matrix E ∈ R^(K×D)
        self.embedding_matrix = nn.Parameter(
            torch.randn(soft_unit_dim, embedding_dim) * 0.1
        )
        
    def forward(self, soft_units: torch.Tensor) -> torch.Tensor:
        """
        Convert soft units to continuous embeddings
        
        Args:
            soft_units: [batch_size, seq_len, soft_unit_dim]
            
        Returns:
            embeddings: [batch_size, seq_len, embedding_dim]
        """
        # z_t = s_t @ E (matrix multiplication)
        embeddings = torch.matmul(soft_units, self.embedding_matrix)
        return embeddings

class ResBlock(nn.Module):
    """Residual block for HiFi-GAN generator"""
    
    def __init__(self, 
                 channels: int,
                 kernel_size: int = 3,
                 dilation: Tuple[int, ...] = (1, 3, 5)):
        super().__init__()
        
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        
        for dil in dilation:
            self.convs1.append(
                weight_norm(nn.Conv1d(
                    channels, channels, kernel_size,
                    stride=1, dilation=dil,
                    padding=get_padding(kernel_size, dil)
                ))
            )
            
            self.convs2.append(
                weight_norm(nn.Conv1d(
                    channels, channels, kernel_size,
                    stride=1, dilation=1,
                    padding=get_padding(kernel_size, 1)
                ))
            )
        
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = conv1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = conv2(xt)
            x = xt + x
        return x
    
    def remove_weight_norm(self):
        for conv in self.convs1:
            remove_weight_norm(conv)
        for conv in self.convs2:
            remove_weight_norm(conv)

class SoftHiFiGANGenerator(nn.Module):
    """
    HiFi-GAN Generator adapted for soft units
    Based on the paper's soft-unit HiFi-GAN vocoder
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Soft unit embedding
        self.soft_embedding = SoftUnitEmbedding(
            config.soft_unit_dim,
            config.embedding_dim
        )
        
        # Initial convolution
        self.conv_pre = weight_norm(nn.Conv1d(
            config.embedding_dim,
            config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3
        ))
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (rate, kernel) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(nn.ConvTranspose1d(
                    config.upsample_initial_channel // (2**i),
                    config.upsample_initial_channel // (2**(i+1)),
                    kernel_size=kernel,
                    stride=rate,
                    padding=(kernel - rate) // 2
                ))
            )
        
        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = config.upsample_initial_channel // (2**(i+1))
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, kernel_size, dilation))
        
        # Final convolution
        self.conv_post = weight_norm(nn.Conv1d(
            ch, 1, kernel_size=7, stride=1, padding=3
        ))
        
        # Initialize weights
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        
    def forward(self, soft_units: torch.Tensor) -> torch.Tensor:
        """
        Generate waveform from soft units
        
        Args:
            soft_units: [batch_size, seq_len, soft_unit_dim]
            
        Returns:
            audio: [batch_size, 1, audio_length]
        """
        # Convert soft units to embeddings
        embeddings = self.soft_embedding(soft_units)  # [B, T, D]
        
        # Transpose for conv1d: [B, D, T]
        x = embeddings.transpose(1, 2)
        
        # Initial convolution
        x = self.conv_pre(x)
        x = F.leaky_relu(x, 0.1)
        
        # Upsampling with residual blocks
        for i, up in enumerate(self.ups):
            x = up(x)
            x = F.leaky_relu(x, 0.1)
            
            # Apply residual blocks
            xs = None
            for j in range(len(self.config.resblock_kernel_sizes)):
                idx = i * len(self.config.resblock_kernel_sizes) + j
                if xs is None:
                    xs = self.resblocks[idx](x)
                else:
                    xs += self.resblocks[idx](x)
            x = xs / len(self.config.resblock_kernel_sizes)
        
        # Final convolution
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x
    
    def remove_weight_norm(self):
        """Remove weight normalization for inference"""
        print('Removing weight norm...')
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

class DiscriminatorP(nn.Module):
    """Period discriminator"""
    
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period
        
        norm_f = weight_norm if period == 1 else spectral_norm
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        
        # 1D to 2D
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, fmap

class DiscriminatorS(nn.Module):
    """Scale discriminator"""
    
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, fmap

class MultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator"""
    
    def __init__(self, config):
        super().__init__()
        periods = config.periods
        
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
        ])
        
        self.discriminators.extend([
            DiscriminatorP(period) for period in periods
        ])
        
    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        
        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class SoftHiFiGAN(nn.Module):
    """
    Complete Soft HiFi-GAN model with generator and discriminator
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.generator = SoftHiFiGANGenerator(config)
        self.discriminator = MultiPeriodDiscriminator(config)
        
        # Mel spectrogram transform for loss computation
        self.mel_transform = nn.Module()
        
    def forward(self, soft_units: torch.Tensor) -> torch.Tensor:
        """Generate audio from soft units"""
        return self.generator(soft_units)
    
    def discriminate(self, real_audio: torch.Tensor, fake_audio: torch.Tensor):
        """Run discriminator on real and fake audio"""
        return self.discriminator(real_audio, fake_audio)

class SoftHiFiGANLoss:
    """
    Loss functions for Soft HiFi-GAN training
    """
    
    def __init__(self, config):
        self.config = config
        self.lambda_adv = config.lambda_adv
        self.lambda_feat = config.lambda_feat
        self.lambda_mel = config.lambda_mel
        
        # Mel spectrogram loss
        self.mel_loss = MelSpectrogramLoss(config)
        
    def generator_loss(self,
                      disc_outputs_fake: List[torch.Tensor]) -> torch.Tensor:
        """Adversarial loss for generator"""
        loss = 0
        for dg in disc_outputs_fake:
            loss += torch.mean((dg - 1)**2)
        return loss
    
    def discriminator_loss(self,
                          disc_outputs_real: List[torch.Tensor],
                          disc_outputs_fake: List[torch.Tensor]) -> torch.Tensor:
        """Adversarial loss for discriminator"""
        loss = 0
        r_losses = []
        g_losses = []
        
        for dr, dg in zip(disc_outputs_real, disc_outputs_fake):
            r_loss = torch.mean((dr - 1)**2)
            g_loss = torch.mean(dg**2)
            loss += r_loss + g_loss
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())
        
        return loss, r_losses, g_losses
    
    def feature_matching_loss(self,
                             fmap_real: List[List[torch.Tensor]],
                             fmap_fake: List[List[torch.Tensor]]) -> torch.Tensor:
        """Feature matching loss"""
        loss = 0
        for dr, dg in zip(fmap_real, fmap_fake):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        return loss * 2
    
    def compute_generator_loss(self,
                              real_audio: torch.Tensor,
                              fake_audio: torch.Tensor,
                              disc_outputs_fake: List[torch.Tensor],
                              fmap_real: List[List[torch.Tensor]],
                              fmap_fake: List[List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Complete generator loss"""
        # Adversarial loss
        adv_loss = self.generator_loss(disc_outputs_fake)
        
        # Feature matching loss
        feat_loss = self.feature_matching_loss(fmap_real, fmap_fake)
        
        # Mel spectrogram loss
        mel_loss = self.mel_loss(real_audio, fake_audio)
        
        # Total loss
        total_loss = (self.lambda_adv * adv_loss + 
                     self.lambda_feat * feat_loss + 
                     self.lambda_mel * mel_loss)
        
        return {
            "total_loss": total_loss,
            "adv_loss": adv_loss,
            "feat_loss": feat_loss,
            "mel_loss": mel_loss
        }

class MelSpectrogramLoss(nn.Module):
    """Mel spectrogram L1 loss"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sampling_rate,
            n_fft=config.win_size,
            hop_length=config.hop_size,
            n_mels=config.n_mel_channels,
            f_min=config.mel_fmin,
            f_max=config.mel_fmax
        )
        
    def forward(self, real_audio: torch.Tensor, fake_audio: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram L1 loss"""
        real_mel = self.mel_transform(real_audio.squeeze(1))
        fake_mel = self.mel_transform(fake_audio.squeeze(1))
        
        # Log scale
        real_mel = torch.log(torch.clamp(real_mel, min=1e-5))
        fake_mel = torch.log(torch.clamp(fake_mel, min=1e-5))
        
        return F.l1_loss(fake_mel, real_mel)

def load_checkpoint(filepath: str, device: str = "cpu") -> Dict:
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    return checkpoint

def save_checkpoint(filepath: str,
                   generator: nn.Module,
                   discriminator: nn.Module,
                   optimizer_g: torch.optim.Optimizer,
                   optimizer_d: torch.optim.Optimizer,
                   step: int,
                   epoch: int,
                   loss: float):
    """Save model checkpoint"""
    checkpoint = {
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "step": step,
        "epoch": epoch,
        "loss": loss
    }
    torch.save(checkpoint, filepath)