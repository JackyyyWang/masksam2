import math
from typing import Optional, Tuple, Type, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ...common import loralib as lora
from ...common import LayerNorm2d


class MultiModalFusion(nn.Module):
    """
    Memory-efficient module for fusing multiple imaging modalities.
    """

    def __init__(
            self,
            args,
            num_modalities: int = 3,
            out_channels: int = 3,
            fusion_type: str = 'cross_attention',
            embed_dim: int = 64,
            num_heads: int = 8,
            lora_rank: int = 4,
            use_modality_dropout: bool = True,
            modality_dropout_rate: float = 0.1,
            window_size: int = 8,
            reduction_factor: int = 4,
    ) -> None:
        super().__init__()
        self.args = args
        self.num_modalities = num_modalities
        self.out_channels = out_channels
        self.fusion_type = fusion_type
        self.use_modality_dropout = use_modality_dropout
        self.modality_dropout_rate = modality_dropout_rate
        self.window_size = window_size
        self.reduction_factor = reduction_factor

        # Modality-specific normalizations
        self.norm_layers = nn.ModuleList([
            nn.GroupNorm(1, 1) for _ in range(num_modalities)
        ])

        if self.fusion_type == 'linear':
            # Simple linear projection using LoRA
            self.fusion = lora.Conv2d(
                num_modalities, out_channels, kernel_size=1,
                stride=1, padding=0, bias=True, r=lora_rank
            )

        elif self.fusion_type == 'dynamic':
            # Dynamic weighted fusion with spatial and channel attention
            self.spatial_att = nn.Sequential(
                nn.Conv2d(num_modalities, num_modalities, kernel_size=3, padding=1, groups=num_modalities),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_modalities, num_modalities, kernel_size=1),
                nn.Sigmoid()
            )

            self.channel_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(num_modalities, num_modalities // 2, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_modalities // 2, num_modalities, kernel_size=1),
                nn.Sigmoid()
            )

            # Final projection to RGB-like output
            self.out_proj = lora.Conv2d(
                num_modalities, out_channels, kernel_size=1,
                stride=1, padding=0, bias=True, r=lora_rank
            )

        elif self.fusion_type == 'cross_attention':
            # ⚠️ SIMPLIFIED IMPLEMENTATION ⚠️
            # We'll use convolutions and simplified attention to avoid memory issues

            # First reduce spatial dimensions to save memory
            self.reduction = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(1, embed_dim, kernel_size=reduction_factor,
                              stride=reduction_factor, padding=0),
                    nn.LayerNorm([embed_dim, 1024 // reduction_factor, 1024 // reduction_factor]),
                    nn.GELU()
                ) for _ in range(num_modalities)
            ])

            # Simpler cross-attention mechanism
            self.q_projs = nn.ModuleList([
                lora.Conv2d(embed_dim, embed_dim, kernel_size=1, r=lora_rank)
                for _ in range(num_modalities)
            ])

            self.k_projs = nn.ModuleList([
                lora.Conv2d(embed_dim, embed_dim, kernel_size=1, r=lora_rank)
                for _ in range(num_modalities)
            ])

            self.v_projs = nn.ModuleList([
                lora.Conv2d(embed_dim, embed_dim, kernel_size=1, r=lora_rank)
                for _ in range(num_modalities)
            ])

            self.scale = embed_dim ** -0.5

            # Final upsampling and projection to RGB-like output
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=reduction_factor,
                                   stride=reduction_factor, padding=0),
                nn.LayerNorm([embed_dim, 1024, 1024]),
                nn.GELU()
            )

            self.out_proj = lora.Conv2d(
                embed_dim, out_channels, kernel_size=1,
                stride=1, padding=0, bias=True, r=lora_rank
            )

        elif self.fusion_type == 'corr_aware':
            # Simpler correlation-aware fusion with reduced memory footprint
            self.modality_encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(1, embed_dim, kernel_size=reduction_factor,
                              stride=reduction_factor, padding=0),
                    nn.LayerNorm([embed_dim, 1024 // reduction_factor, 1024 // reduction_factor]),
                    nn.GELU()
                ) for _ in range(num_modalities)
            ])

            # Correlation module (channel-wise instead of spatial)
            self.corr_module = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(num_modalities * embed_dim, num_modalities, kernel_size=1),
                nn.Sigmoid()
            )

            # Upsampling
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=reduction_factor,
                                   stride=reduction_factor, padding=0),
                nn.LayerNorm([embed_dim, 1024, 1024]),
                nn.GELU()
            )

            # Final projection to RGB-like output
            self.out_proj = lora.Conv2d(
                embed_dim, out_channels, kernel_size=1,
                stride=1, padding=0, bias=True, r=lora_rank
            )

        # Register ImageNet normalization parameters
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multimodal fusion
        Args:
            x: Tensor of shape [B, C, H, W] where C contains all modalities
        Returns:
            Fused tensor of shape [B, 3, H, W] (RGB-like)
        """
        # Split input into separate modalities
        modalities = []
        for i in range(self.num_modalities):
            if i < x.shape[1]:
                modalities.append(x[:, i:i + 1, :, :])
            else:
                # Handle case where there are fewer channels than expected modalities
                # by duplicating the last modality
                modalities.append(x[:, -1:, :, :])

        # Normalize each modality individually
        x_norm = [self.norm_layers[i](mod) for i, mod in enumerate(modalities)]

        # Apply modality dropout during training if enabled
        if self.training and self.use_modality_dropout:
            drop_mask = torch.rand(len(x_norm), device=x.device) > self.modality_dropout_rate
            for i in range(len(x_norm)):
                if not drop_mask[i]:
                    x_norm[i] = torch.zeros_like(x_norm[i])

        if self.fusion_type == 'linear':
            # Stack modalities along channel dimension
            x_stacked = torch.cat(x_norm, dim=1)
            # Apply 1x1 convolution to get RGB-like output
            fused = self.fusion(x_stacked)

        elif self.fusion_type == 'dynamic':
            # Stack modalities along channel dimension
            x_stacked = torch.cat(x_norm, dim=1)

            # Compute spatial and channel attention weights
            spatial_weights = self.spatial_att(x_stacked)
            channel_weights = self.channel_att(x_stacked)

            # Apply attention weights
            x_weighted = x_stacked * spatial_weights * channel_weights

            # Project to RGB-like output
            fused = self.out_proj(x_weighted)

        elif self.fusion_type == 'cross_attention':
            # ⚠️ SIMPLIFIED IMPLEMENTATION ⚠️

            # Apply spatial reduction to each modality
            reduced_features = [self.reduction[i](mod) for i, mod in enumerate(x_norm)]

            # Simple cross-modal attention
            fused_feature = self._simple_cross_attention(reduced_features)

            # Upsample back to original resolution
            fused_feature = self.upsample(fused_feature)

            # Project to RGB-like output
            fused = self.out_proj(fused_feature)

        elif self.fusion_type == 'corr_aware':
            # Encode each modality
            encoded_features = [encoder(mod) for encoder, mod in zip(self.modality_encoders, x_norm)]

            # Compute correlations using channel-wise pooling (memory efficient)
            pooled_features = [F.adaptive_avg_pool2d(feat, 1) for feat in encoded_features]
            pooled_concat = torch.cat(pooled_features, dim=1)

            # Get correlation weights
            corr_weights = self.corr_module(pooled_concat)
            corr_weights = corr_weights.view(corr_weights.size(0), -1, 1, 1)

            # Apply weights to features
            weighted_features = [feat * w for feat, w in
                                 zip(encoded_features, corr_weights.chunk(self.num_modalities, dim=1))]

            # Sum weighted features
            fused_feature = sum(weighted_features)

            # Upsample back to original resolution
            fused_feature = self.upsample(fused_feature)

            # Project to RGB-like output
            fused = self.out_proj(fused_feature)

        # Apply ImageNet normalization
        fused = (fused - self.mean) / self.std

        return fused

    def _simple_cross_attention(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Memory-efficient simplified cross-attention between modalities.
        This version uses channel-wise attention instead of spatial attention
        to avoid memory issues.

        Args:
            features: List of tensors, each of shape [B, C, H, W]
        Returns:
            Fused feature of shape [B, C, H, W]
        """
        num_modalities = len(features)
        batch_size = features[0].shape[0]

        # Process queries, keys, values for each modality
        q_features = [self.q_projs[i](feat) for i, feat in enumerate(features)]
        k_features = [self.k_projs[i](feat) for i, feat in enumerate(features)]
        v_features = [self.v_projs[i](feat) for i, feat in enumerate(features)]

        # Initialize output tensor
        output_features = []

        # For each modality as query
        for i in range(num_modalities):
            # Get query features
            q = q_features[i]  # [B, C, H, W]

            # Apply global average pooling to get a channel descriptor
            # This drastically reduces memory usage
            q_pool = F.adaptive_avg_pool2d(q, 1)  # [B, C, 1, 1]

            attended_features = []

            # For each modality as key/value
            for j in range(num_modalities):
                if i == j:
                    # Skip self-attention to save memory
                    continue

                # Get key and value features
                k = k_features[j]  # [B, C, H, W]
                v = v_features[j]  # [B, C, H, W]

                # Apply global average pooling to keys
                k_pool = F.adaptive_avg_pool2d(k, 1)  # [B, C, 1, 1]

                # Compute channel attention weights
                # Reshape for matrix multiplication
                q_flat = q_pool.flatten(2)  # [B, C, 1]
                k_flat = k_pool.flatten(2).transpose(1, 2)  # [B, 1, C]

                # Compute attention scores
                attn_weights = torch.matmul(q_flat, k_flat) * self.scale  # [B, C, C]
                attn_weights = F.softmax(attn_weights, dim=-1)  # [B, C, C]

                # Apply attention to values
                # First, reshape v to [B, C, H*W]
                v_flat = v.flatten(2)  # [B, C, H*W]

                # Apply attention
                attended_v = torch.matmul(attn_weights, v_flat)  # [B, C, H*W]

                # Reshape back to spatial dimensions
                attended_v = attended_v.view_as(v)  # [B, C, H, W]

                attended_features.append(attended_v)

            # Average attended features
            if attended_features:
                modality_output = sum(attended_features) / len(attended_features)
            else:
                # If no cross-attention, use the original features
                modality_output = q

            output_features.append(modality_output)

        # Average the outputs from all modalities
        final_output = sum(output_features) / len(output_features)

