"""
Enhanced S2UT model with severity-aware contrastive learning
Novel contribution: Dysarthria-specific adaptations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import numpy as np

class SeverityAwareContrastiveLoss(nn.Module):
    """
    Novel severity-aware contrastive learning for dysarthric speech
    
    Key Innovation: Distance-weighted contrastive learning that accounts for
    dysarthria severity hierarchy (mild -> moderate -> severe)
    """
    
    def __init__(self, 
                 temperature: float = 0.07,
                 severity_weights: Dict[str, float] = None,
                 adaptive_margin: bool = True):
        super().__init__()
        self.temperature = temperature
        self.adaptive_margin = adaptive_margin
        
        # Severity hierarchy weights
        self.severity_weights = severity_weights or {
            'mild': 1.0,
            'moderate': 1.5, 
            'severe': 2.0
        }
        
        # Learnable severity embeddings
        self.severity_embedding = nn.Embedding(4, 64)  # healthy, mild, moderate, severe
        self.severity_to_idx = {'healthy': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
        
    def compute_severity_aware_similarity(self,
                                        predicted_units: torch.Tensor,
                                        target_units: torch.Tensor,
                                        severities: List[str]) -> torch.Tensor:
        """
        Compute severity-aware similarity with adaptive margins
        """
        # Get severity embeddings
        severity_indices = [self.severity_to_idx[s] for s in severities]
        severity_embeds = self.severity_embedding(torch.tensor(severity_indices, device=predicted_units.device))
        
        # Base cosine similarity
        base_sim = F.cosine_similarity(predicted_units, target_units, dim=-1)
        
        if self.adaptive_margin:
            # Adaptive margin based on severity
            severity_factors = torch.tensor([self.severity_weights[s] for s in severities], 
                                          device=predicted_units.device)
            
            # Apply severity-dependent margin
            margin = 0.1 * severity_factors  # More margin for severe cases
            adapted_sim = base_sim - margin.unsqueeze(-1)
            
            return adapted_sim
        
        return base_sim
    
    def forward(self,
                predicted_soft_units: torch.Tensor,
                positive_soft_units: torch.Tensor, 
                negative_soft_units: torch.Tensor,
                severities: List[str],
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Severity-aware contrastive loss computation
        """
        # Positive similarities (severity-aware)
        pos_sim = self.compute_severity_aware_similarity(
            predicted_soft_units, positive_soft_units, severities
        )
        
        # Negative similarities (standard)
        neg_sim = F.cosine_similarity(predicted_soft_units, negative_soft_units, dim=-1)
        
        # Temperature scaling
        pos_sim = pos_sim / self.temperature
        neg_sim = neg_sim / self.temperature
        
        # Contrastive loss with severity awareness
        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
        
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()
        
        return loss

class DysarthriaSpecificEncoder(nn.Module):
    """
    Novel encoder with dysarthria-specific adaptations
    
    Key Innovation: Prosody-aware attention and disfluency modeling
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Standard transformer encoder
        self.base_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.encoder_embed_dim,
                nhead=config.encoder_attention_heads,
                dim_feedforward=config.encoder_ffn_embed_dim,
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=config.encoder_layers
        )
        
        # Prosody-specific branch
        self.prosody_extractor = nn.Sequential(
            nn.Conv1d(config.encoder_embed_dim, config.encoder_embed_dim // 2, 
                     kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(config.encoder_embed_dim // 2, config.encoder_embed_dim // 4,
                     kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Disfluency detection head
        self.disfluency_detector = nn.Sequential(
            nn.Linear(config.encoder_embed_dim, config.encoder_embed_dim // 2),
            nn.ReLU(),
            nn.Linear(config.encoder_embed_dim // 2, 2)  # fluent vs disfluent
        )
        
        # Fusion layer
        self.fusion = nn.MultiheadAttention(
            embed_dim=config.encoder_embed_dim,
            num_heads=4,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Forward pass with dysarthria-specific processing
        """
        # Base encoding
        encoded = self.base_encoder(x, src_key_padding_mask=mask)
        
        # Prosody extraction
        prosody_features = self.prosody_extractor(encoded.transpose(1, 2)).squeeze(-1)
        
        # Disfluency detection
        disfluency_scores = self.disfluency_detector(encoded)
        disfluency_weights = F.softmax(disfluency_scores, dim=-1)[:, :, 1:2]  # Take disfluent prob
        
        # Apply disfluency-aware weighting
        weighted_encoded = encoded * (1 + disfluency_weights)
        
        # Fusion with prosody
        prosody_expanded = prosody_features.unsqueeze(1).expand(-1, encoded.size(1), -1)
        fused_output, _ = self.fusion(weighted_encoded, prosody_expanded, prosody_expanded)
        
        return {
            'encoded': fused_output,
            'prosody_features': prosody_features,
            'disfluency_scores': disfluency_scores
        }

class EnhancedS2UTModel(nn.Module):
    """
    Enhanced S2UT with novel dysarthria-specific innovations
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Enhanced encoder
        self.encoder = DysarthriaSpecificEncoder(config)
        
        # Standard decoder (can be enhanced further)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_embed_dim,
            nhead=config.decoder_attention_heads,
            dim_feedforward=config.decoder_ffn_embed_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(config.decoder_embed_dim, config.output_dim)
        
        # Novel severity-aware contrastive loss
        self.severity_contrastive = SeverityAwareContrastiveLoss(
            temperature=config.temperature
        )
        
        # Auxiliary losses
        self.prosody_consistency_loss = nn.MSELoss()
        self.disfluency_classification_loss = nn.CrossEntropyLoss()
        
    def forward(self, 
                dysarthric_mel: torch.Tensor,
                target_soft_units: Optional[torch.Tensor] = None,
                severities: Optional[List[str]] = None,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with auxiliary outputs
        """
        # Enhanced encoding
        encoder_outputs = self.encoder(dysarthric_mel.transpose(1, 2), src_mask)
        encoded = encoder_outputs['encoded']
        
        # Decoding
        if target_soft_units is not None:
            # Teacher forcing
            tgt_input = torch.cat([
                torch.zeros(target_soft_units.size(0), 1, target_soft_units.size(2), 
                           device=target_soft_units.device),
                target_soft_units[:, :-1, :]
            ], dim=1)
            decoded = self.decoder(tgt_input, encoded, tgt_key_padding_mask=tgt_mask)
        else:
            # Autoregressive decoding (for inference)
            decoded = self._autoregressive_decode(encoded, src_mask)
        
        # Output projection
        predicted_soft_units = F.softmax(self.output_projection(decoded), dim=-1)
        
        return {
            'predicted_soft_units': predicted_soft_units,
            'encoder_output': encoded,
            'prosody_features': encoder_outputs['prosody_features'],
            'disfluency_scores': encoder_outputs['disfluency_scores']
        }
    
    def compute_enhanced_loss(self,
                            predicted_soft_units: torch.Tensor,
                            target_soft_units: torch.Tensor,
                            negative_soft_units: torch.Tensor,
                            severities: List[str],
                            prosody_target: Optional[torch.Tensor] = None,
                            disfluency_labels: Optional[torch.Tensor] = None,
                            mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Enhanced loss computation with auxiliary objectives
        """
        # Base cross-entropy loss
        ce_loss = -torch.sum(target_soft_units * torch.log_softmax(predicted_soft_units, dim=-1), dim=-1)
        if mask is not None:
            ce_loss = (ce_loss * mask).sum() / mask.sum()
        else:
            ce_loss = ce_loss.mean()
        
        # Severity-aware contrastive loss
        contrastive_loss = self.severity_contrastive(
            predicted_soft_units, target_soft_units, negative_soft_units, severities, mask
        )
        
        # Auxiliary losses
        aux_loss = 0.0
        if prosody_target is not None:
            aux_loss += 0.1 * self.prosody_consistency_loss(
                self.encoder.prosody_extractor(predicted_soft_units.transpose(1, 2)).squeeze(-1),
                prosody_target
            )
        
        if disfluency_labels is not None:
            disfluency_loss = self.disfluency_classification_loss(
                self.encoder.disfluency_detector(predicted_soft_units).view(-1, 2),
                disfluency_labels.view(-1)
            )
            aux_loss += 0.05 * disfluency_loss
        
        # Total loss
        total_loss = ce_loss + self.config.contrastive_weight * contrastive_loss + aux_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'contrastive_loss': contrastive_loss,
            'auxiliary_loss': aux_loss
        }
    
    def _autoregressive_decode(self, encoder_output: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        """Autoregressive decoding for inference"""
        batch_size = encoder_output.size(0)
        max_len = encoder_output.size(1)
        device = encoder_output.device
        
        # Start with zero vector
        decoded = torch.zeros(batch_size, 1, self.config.output_dim, device=device)
        
        for step in range(max_len):
            # Decode next step
            output = self.decoder(decoded, encoder_output)
            next_output = F.softmax(self.output_projection(output[:, -1:, :]), dim=-1)
            decoded = torch.cat([decoded, next_output], dim=1)
        
        return decoded[:, 1:, :]  # Remove initial zero vector