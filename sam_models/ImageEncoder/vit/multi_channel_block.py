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
            self.fusion = nn.Conv2d(
                num_modalities, out_channels, kernel_size=1,
                stride=1, padding=0, bias=True
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
            self.out_proj = nn.Conv2d(
                num_modalities, out_channels, kernel_size=1,
                stride=1, padding=0, bias=True
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
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
                for _ in range(num_modalities)
            ])

            self.k_projs = nn.ModuleList([
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
                for _ in range(num_modalities)
            ])

            self.v_projs = nn.ModuleList([
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
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

            self.out_proj = nn.Conv2d(
                embed_dim, out_channels, kernel_size=1,
                stride=1, padding=0, bias=True
            )

        elif self.fusion_type == 'corr_aware':

                # Downsampling for memory efficiency
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=reduction_factor, stride=reduction_factor)
            )
            # Encoders (same as before)
            self.shared_encoder = nn.Sequential(
                nn.Conv2d(1, embed_dim // 2, kernel_size=3, padding=1, bias=True),
                nn.LayerNorm([embed_dim // 2, 1024 // reduction_factor, 1024 // reduction_factor]),
                nn.GELU()
            )
            
            self.unique_encoder = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(1, embed_dim // 2, kernel_size=3, padding=1, bias=True),
                    nn.LayerNorm([embed_dim // 2, 1024 // reduction_factor, 1024 // reduction_factor]),
                    nn.GELU()
                ) for _ in range(num_modalities)
            ])
            
            self.shared_corr_processor = nn.Sequential(
                # Multiply by (embed_dim // 2) to account for the channel dimension of each correlation pair
                nn.Conv2d((embed_dim // 2) * (num_modalities * (num_modalities - 1) // 2), embed_dim // 4, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=3, padding=1),
                nn.GELU()
            )
            
            # Attention mechanism for Unique-Shared interaction
            self.query_proj = nn.ModuleList([
                nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=1)
                for _ in range(num_modalities)
            ])
            
            self.key_proj = nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=1)
            self.value_proj = nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=1)
            
            # Weight Generator - combines correlation and attention information
            self.weight_generator = nn.Sequential(
                nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(embed_dim // 2, num_modalities * 2, kernel_size=1),  # Generate alpha and beta weights
                nn.Sigmoid()
            )
            # Add this to your initialization
            self.upsample =nn.Upsample(
                scale_factor=reduction_factor, 
                mode='bilinear', 
                align_corners=False
            )

            # Add output projection if not already defined
            self.out_proj = lora.Conv2d(
                embed_dim, out_channels, kernel_size=1, 
                stride=1, padding=0, bias=True, r=lora_rank
            )

        # Register ImageNet normalization parameters
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        # Add this to your __init__ method:
        self.out_norm = nn.BatchNorm2d(out_channels, affine=True)
        self.out_norm.weight.data.fill_(1.0)  # Initialize to identity transform
        self.out_norm.bias.data.fill_(0.0)    # No initial bias

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

            # Apply spatial reduction to each modality
            reduced_features = [self.reduction[i](mod) for i, mod in enumerate(x_norm)]

            # Simple cross-modal attention
            fused_feature = self._simple_cross_attention(reduced_features)

            # Upsample back to original resolution
            fused_feature = self.upsample(fused_feature)

            # Project to RGB-like output
            fused = self.out_proj(fused_feature)

        elif self.fusion_type == 'corr_aware':
            # Extract features (same as before)
            x_downsampled = [self.downsample(mod) for mod in x_norm]
            # x_downsampled = x_norm #[self.reduction[i](mod) for i, mod in enumerate(x_norm)]
            shared_features = [self.shared_encoder(mod) for mod in x_downsampled]
            unique_features = [self.unique_encoder[i](mod) for i, mod in enumerate(x_downsampled)]
            
            # 1. Process Shared-Shared Correlations
            shared_corr_pairs = []
            for i in range(self.num_modalities):
                for j in range(i+1, self.num_modalities):
                    corr_ij = shared_features[i] * shared_features[j]
                    shared_corr_pairs.append(corr_ij)
            
            if shared_corr_pairs:
                shared_corr = torch.cat(shared_corr_pairs, dim=1)
                shared_corr_features = self.shared_corr_processor(shared_corr)
            else:
                # Fallback for single modality
                shared_corr_features = torch.zeros_like(shared_features[0][:, :embed_dim//4])
            
            # # 2. Process Unique-Shared Attention
            # attention_features = torch.zeros_like(shared_corr_features)
            
            # # For each modality, unique features attend to all shared features
            # for i in range(self.num_modalities):
            #     query = self.query_proj[i](unique_features[i])  # [B, C/4, H, W]
                
            #     # Compute attention with all shared features
            #     attention_map = torch.zeros_like(query)
            #     for j in range(self.num_modalities):
            #         key = self.key_proj(shared_features[j])    # [B, C/4, H, W]
            #         value = self.value_proj(shared_features[j])  # [B, C/4, H, W]
                    
            #         # Compute spatial attention
            #         # Reshape for matrix multiplication
            #         b, c, h, w = query.shape
            #         q_flat = query.flatten(2)  # [B, C, HW]
            #         k_flat = key.flatten(2)    # [B, C, HW]
            #         v_flat = value.flatten(2)  # [B, C, HW]
                    
            #         # Compute attention scores
            #         attn = torch.matmul(q_flat.transpose(1, 2), k_flat) / math.sqrt(c)  # [B, HW, HW]
            #         attn = F.softmax(attn, dim=-1)
                    
            #         # Apply attention
            #         attended = torch.matmul(attn, v_flat.transpose(1, 2)).transpose(1, 2)  # [B, C, HW]
            #         attended = attended.reshape(b, c, h, w)
                    
            #         # Accumulate attention results from all shared features
            #         attention_map = attention_map + attended / self.num_modalities
                
            #     # Accumulate attention features from all modalities
            #     attention_features = attention_features + attention_map / self.num_modalities
            # Replace the problematic attention computation in your forward method:

            # 2. Process Unique-Shared Attention with windowing
            attention_features = torch.zeros_like(shared_corr_features)
            window_size = 16  # Adjust this based on your GPU memory

            # For each modality, unique features attend to all shared features
            for i in range(self.num_modalities):
                query = self.query_proj[i](unique_features[i])  # [B, C/4, H, W]
                
                # Accumulate attention results across modalities
                attention_map = torch.zeros_like(query)
                
                for j in range(self.num_modalities):
                    key = self.key_proj(shared_features[j])    # [B, C/4, H, W]
                    value = self.value_proj(shared_features[j])  # [B, C/4, H, W]
                    
                    # Get dimensions
                    b, c, h, w = query.shape
                    
                    # Reshape to windows
                    # First pad if needed
                    pad_h = (window_size - h % window_size) % window_size
                    pad_w = (window_size - w % window_size) % window_size
                    
                    if pad_h > 0 or pad_w > 0:
                        query_padded = F.pad(query, (0, pad_w, 0, pad_h))
                        key_padded = F.pad(key, (0, pad_w, 0, pad_h))
                        value_padded = F.pad(value, (0, pad_w, 0, pad_h))
                    else:
                        query_padded = query
                        key_padded = key
                        value_padded = value
                    
                    # New dimensions after padding
                    _, _, hp, wp = query_padded.shape
                    
                    # Reshape to [B, C, H//window_size, window_size, W//window_size, window_size]
                    query_windows = query_padded.view(b, c, hp // window_size, window_size, wp // window_size, window_size)
                    key_windows = key_padded.view(b, c, hp // window_size, window_size, wp // window_size, window_size)
                    value_windows = value_padded.view(b, c, hp // window_size, window_size, wp // window_size, window_size)
                    
                    # Permute and reshape to [B*(H//window_size)*(W//window_size), window_size*window_size, C]
                    query_windows = query_windows.permute(0, 2, 4, 1, 3, 5).contiguous().view(
                        b * (hp // window_size) * (wp // window_size), window_size * window_size, c
                    )
                    key_windows = key_windows.permute(0, 2, 4, 1, 3, 5).contiguous().view(
                        b * (hp // window_size) * (wp // window_size), window_size * window_size, c
                    )
                    value_windows = value_windows.permute(0, 2, 4, 1, 3, 5).contiguous().view(
                        b * (hp // window_size) * (wp // window_size), window_size * window_size, c
                    )
                    
                    # Compute attention scores
                    attn = torch.bmm(query_windows, key_windows.transpose(1, 2)) * (1.0 / math.sqrt(c))
                    attn = F.softmax(attn, dim=-1)
                    
                    # Apply attention
                    out = torch.bmm(attn, value_windows)  # [B*(H//ws)*(W//ws), window_size*window_size, C]
                    
                    # Reshape back to [B, C, H, W]
                    out = out.view(b, hp // window_size, wp // window_size, window_size, window_size, c)
                    out = out.permute(0, 5, 1, 3, 2, 4).contiguous().view(b, c, hp, wp)
                    
                    # Remove padding if needed
                    if pad_h > 0 or pad_w > 0:
                        out = out[:, :, :h, :w]
                    
                    # Accumulate attention results
                    attention_map = attention_map + out / self.num_modalities
                
                # Accumulate attention features from all modalities
                attention_features = attention_features + attention_map / self.num_modalities
            
            # 3. Combine correlation and attention information for weight generation
            combined_features = torch.cat([shared_corr_features, attention_features], dim=1)
            weights = self.weight_generator(combined_features)
            
            # Split weights into alpha and beta
            alpha_beta = weights.chunk(2, dim=1)
            alpha = alpha_beta[0].chunk(self.num_modalities, dim=1)  # For shared features
            beta = alpha_beta[1].chunk(self.num_modalities, dim=1)   # For unique features
            
            # Apply weights and combine features
            weighted_shared = [feat * w for feat, w in zip(shared_features, alpha)]
            weighted_unique = [feat * w for feat, w in zip(unique_features, beta)]
            
            # Combine features for each modality then sum
            combined_features = []
            for s, u in zip(weighted_shared, weighted_unique):
                combined_features.append(torch.cat([s, u], dim=1))
            
            # Sum across modalities
            fused = sum(combined_features)
            # Upsample back to original resolution
            if hasattr(self, 'upsample'):
                fused = self.upsample(fused)
            else:
                # Fallback if upsample is not defined
                fused = F.interpolate(fused, size=(1024, 1024), mode='bilinear', align_corners=False)

            # Project to RGB-like output
            if hasattr(self, 'out_proj'):
                fused = self.out_proj(fused)

        # Apply batch norm (differentiable)
        fused = self.out_norm(fused)

        # Apply sigmoid to bound to [0, 1]
        fused = torch.sigmoid(fused)
        # Scale to [0, 255]
        fused = fused * 255.0
        # # Apply ImageNet normalization
        # fused = (fused - self.mean) / self.std

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


