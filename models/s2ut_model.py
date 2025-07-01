"""
Speech-to-Unit Translation (S2UT) model for dysarthric voice conversion
Based on the paper: "Direct dysarthric speech-to-soft unit translation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple
import numpy as np

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class Conv1dSubsampling(nn.Module):
    """
    Conv1d subsampling module to reduce sequence length
    Following the paper's architecture
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 kernel_sizes: list = [5, 5],
                 strides: list = [2, 2]):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for kernel_size, stride in zip(kernel_sizes, strides):
            layers.extend([
                nn.Conv1d(in_dim, output_dim, kernel_size, stride=stride, padding=kernel_size//2),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = output_dim
        
        self.conv_layers = nn.Sequential(*layers)
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            mask: [batch_size, seq_len]
        Returns:
            (output, output_mask): ([batch_size, output_seq_len, output_dim], [batch_size, output_seq_len])
        """
        # Transpose for conv1d: [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)
        
        # Apply convolutions
        x = self.conv_layers(x)
        
        # Transpose back: [batch_size, seq_len, output_dim]
        x = x.transpose(1, 2)
        
        # Update mask if provided
        if mask is not None:
            # Compute output mask length based on convolution operations
            for i, layer in enumerate(self.conv_layers):
                if isinstance(layer, nn.Conv1d):
                    kernel_size = layer.kernel_size[0]
                    stride = layer.stride[0]
                    padding = layer.padding[0]
                    
                    mask_len = mask.sum(dim=1)  # Get actual lengths
                    mask_len = (mask_len + 2 * padding - kernel_size) // stride + 1
                    
                    # Create new mask
                    new_mask = torch.zeros(mask.size(0), x.size(1), device=mask.device)
                    for j, length in enumerate(mask_len):
                        new_mask[j, :length] = 1
                    mask = new_mask
        
        return x, mask

class S2UTTransformerEncoder(nn.Module):
    """
    Transformer encoder for S2UT model
    Based on the paper configuration: 12 encoder layers, 4 attention heads
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input projection from mel spectrogram
        self.input_projection = nn.Linear(
            config.input_feat_per_channel,  # 80 mel features
            config.encoder_embed_dim
        )
        
        # Conv1d subsampling
        self.subsampling = Conv1dSubsampling(
            input_dim=config.encoder_embed_dim,
            output_dim=config.encoder_embed_dim,
            kernel_sizes=config.conv_kernel_sizes
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.encoder_embed_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_embed_dim,
            nhead=config.encoder_attention_heads,
            dim_feedforward=config.encoder_ffn_embed_dim,
            dropout=config.dropout,
            activation=config.activation_fn,
            batch_first=True,
            norm_first=config.encoder_normalize_before
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(config.encoder_embed_dim)
        
    def forward(self,
                mel_spectrogram: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            mel_spectrogram: [batch_size, n_mels, seq_len]
            src_mask: [batch_size, seq_len]
        
        Returns:
            (encoder_output, output_mask)
        """
        # Transpose and project: [batch_size, seq_len, embed_dim]
        x = mel_spectrogram.transpose(1, 2)  # [batch_size, seq_len, n_mels]
        x = self.input_projection(x)
        
        # Apply subsampling
        x, src_mask = self.subsampling(x, src_mask)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, embed_dim]
        
        # Create attention mask for transformer
        if src_mask is not None:
            # Convert mask to attention mask (True = ignore)
            attn_mask = (src_mask == 0)
        else:
            attn_mask = None
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=attn_mask)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x, src_mask

class S2UTTransformerDecoder(nn.Module):
    """
    Transformer decoder for S2UT model
    Based on the paper configuration: 6 decoder layers, 8 attention heads
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding for target soft units (if using teacher forcing)
        self.target_embedding = nn.Linear(
            config.output_dim,  # 1000 soft unit dim
            config.decoder_embed_dim
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.decoder_embed_dim)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_embed_dim,
            nhead=config.decoder_attention_heads,
            dim_feedforward=config.decoder_ffn_embed_dim,
            dropout=config.dropout,
            activation=config.activation_fn,
            batch_first=True,
            norm_first=config.decoder_normalize_before
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_layers
        )
        
        # Output projection to soft units
        self.output_projection = nn.Linear(
            config.decoder_embed_dim,
            config.output_dim
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(config.decoder_embed_dim)
        
    def forward(self,
                encoder_output: torch.Tensor,
                target_soft_units: Optional[torch.Tensor] = None,
                encoder_mask: Optional[torch.Tensor] = None,
                target_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            encoder_output: [batch_size, src_seq_len, embed_dim]
            target_soft_units: [batch_size, tgt_seq_len, output_dim] (for teacher forcing)
            encoder_mask: [batch_size, src_seq_len]
            target_mask: [batch_size, tgt_seq_len]
        
        Returns:
            soft_units: [batch_size, tgt_seq_len, output_dim]
        """
        if target_soft_units is not None:
            # Teacher forcing mode
            batch_size, tgt_seq_len = target_soft_units.shape[:2]
            
            # Embed target soft units
            tgt_embed = self.target_embedding(target_soft_units)
            
            # Add positional encoding
            tgt_embed = tgt_embed.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
            tgt_embed = self.pos_encoding(tgt_embed)
            tgt_embed = tgt_embed.transpose(0, 1)  # [batch_size, seq_len, embed_dim]
            
            # Create causal mask for target
            tgt_causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(target_soft_units.device)
            
            # Create attention masks
            memory_key_padding_mask = (encoder_mask == 0) if encoder_mask is not None else None
            tgt_key_padding_mask = (target_mask == 0) if target_mask is not None else None
            
            # Apply transformer decoder
            output = self.transformer_decoder(
                tgt_embed,
                encoder_output,
                tgt_mask=tgt_causal_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
        else:
            # Inference mode - generate autoregressively
            batch_size = encoder_output.size(0)
            max_len = encoder_output.size(1) * 2  # Heuristic for max output length
            
            # Start with zeros
            outputs = []
            tgt_embed = torch.zeros(batch_size, 1, self.config.decoder_embed_dim, device=encoder_output.device)
            
            for step in range(max_len):
                # Add positional encoding
                pos_tgt = tgt_embed.transpose(0, 1)
                pos_tgt = self.pos_encoding(pos_tgt)
                pos_tgt = pos_tgt.transpose(0, 1)
                
                # Create causal mask
                seq_len = pos_tgt.size(1)
                causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(encoder_output.device)
                
                # Create attention masks
                memory_key_padding_mask = (encoder_mask == 0) if encoder_mask is not None else None
                
                # Apply transformer decoder
                output = self.transformer_decoder(
                    pos_tgt,
                    encoder_output,
                    tgt_mask=causal_mask,
                    memory_key_padding_mask=memory_key_padding_mask
                )
                
                # Project to soft units
                step_output = self.output_projection(output[:, -1:, :])
                step_output = F.softmax(step_output, dim=-1)
                outputs.append(step_output)
                
                # Update target embedding for next step
                next_embed = self.target_embedding(step_output)
                tgt_embed = torch.cat([tgt_embed, next_embed], dim=1)
                
                # Simple stopping criterion (you might want to improve this)
                if step > encoder_output.size(1):
                    break
            
            output = torch.cat(outputs, dim=1)
            return output
        
        # Layer normalization and projection
        output = self.layer_norm(output)
        soft_units = self.output_projection(output)
        soft_units = F.softmax(soft_units, dim=-1)
        
        return soft_units

class S2UTModel(nn.Module):
    """
    Complete Speech-to-Unit Translation model
    Translates dysarthric mel spectrograms to healthy soft units
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoder and decoder
        self.encoder = S2UTTransformerEncoder(config)
        self.decoder = S2UTTransformerDecoder(config)
        
        # For contrastive learning
        self.temperature = config.temperature
        
    def forward(self,
                dysarthric_mel: torch.Tensor,
                target_soft_units: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of S2UT model
        
        Args:
            dysarthric_mel: [batch_size, n_mels, seq_len]
            target_soft_units: [batch_size, tgt_seq_len, output_dim]
            src_mask: [batch_size, src_seq_len]
            tgt_mask: [batch_size, tgt_seq_len]
        
        Returns:
            Dictionary containing model outputs
        """
        # Encode dysarthric speech
        encoder_output, encoder_mask = self.encoder(dysarthric_mel, src_mask)
        
        # Decode to soft units
        if target_soft_units is not None:
            # Shift target for teacher forcing
            shifted_target = torch.cat([
                torch.zeros(target_soft_units.size(0), 1, target_soft_units.size(2), device=target_soft_units.device),
                target_soft_units[:, :-1, :]
            ], dim=1)
            
            predicted_soft_units = self.decoder(
                encoder_output,
                shifted_target,
                encoder_mask,
                tgt_mask
            )
        else:
            predicted_soft_units = self.decoder(
                encoder_output,
                None,
                encoder_mask,
                None
            )
        
        return {
            "predicted_soft_units": predicted_soft_units,
            "encoder_output": encoder_output,
            "encoder_mask": encoder_mask
        }
    
    def compute_cross_entropy_loss(self,
                                  predicted_soft_units: torch.Tensor,
                                  target_soft_units: torch.Tensor,
                                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute cross-entropy loss between predicted and target soft units
        """
        batch_size, seq_len, output_dim = predicted_soft_units.shape
        
        # Reshape for loss computation
        predicted_flat = predicted_soft_units.view(-1, output_dim)
        target_flat = target_soft_units.view(-1, output_dim)
        
        # Compute cross-entropy with soft targets
        loss = -torch.sum(target_flat * torch.log_softmax(predicted_flat, dim=-1), dim=-1)
        
        if mask is not None:
            mask_flat = mask.view(-1)
            loss = loss * mask_flat
            loss = loss.sum() / mask_flat.sum()
        else:
            loss = loss.mean()
        
        return loss
    
    def compute_contrastive_loss(self,
                                predicted_soft_units: torch.Tensor,
                                positive_soft_units: torch.Tensor,
                                negative_soft_units: torch.Tensor,
                                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute contrastive loss as described in the paper
        
        L_contrast = -log(exp(sim(s_t, s_t^+)) / (exp(sim(s_t, s_t^+)) + exp(sim(s_t, s_t^-))))
        """
        # Compute cosine similarities
        pos_sim = F.cosine_similarity(predicted_soft_units, positive_soft_units, dim=-1)
        neg_sim = F.cosine_similarity(predicted_soft_units, negative_soft_units, dim=-1)
        
        # Apply temperature scaling
        pos_sim = pos_sim / self.temperature
        neg_sim = neg_sim / self.temperature
        
        # Compute contrastive loss
        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
        
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()
        
        return loss
    
    def compute_total_loss(self,
                          predicted_soft_units: torch.Tensor,
                          target_soft_units: torch.Tensor,
                          negative_soft_units: torch.Tensor,
                          mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute total loss combining cross-entropy and contrastive losses
        
        L_total = L_CE + λ * L_contrast
        """
        # Cross-entropy loss
        ce_loss = self.compute_cross_entropy_loss(
            predicted_soft_units, target_soft_units, mask
        )
        
        # Contrastive loss
        contrastive_loss = self.compute_contrastive_loss(
            predicted_soft_units, target_soft_units, negative_soft_units, mask
        )
        
        # Total loss
        total_loss = ce_loss + self.config.contrastive_weight * contrastive_loss
        
        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "contrastive_loss": contrastive_loss
        }