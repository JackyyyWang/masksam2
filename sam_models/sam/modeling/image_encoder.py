# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ...common import LayerNorm2d
from ...ImageEncoder import AdapterBlock, Block, LoraBlock, AdaloraBlock
# from ...ImageEncoder.vit.multi_lora_block import MultiModalFusion
from ...ImageEncoder.vit.multi_channel_block import MultiModalFusion

from typing import Optional, Tuple, Type, List, Union

# Modified ImageEncoderViT class to support multimodal input
class ImageEncoderViT(nn.Module):
    def __init__(
            self,
            args,
            img_size: int = 1024,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.args = args

        # Add multimodal fusion module if specified
        self.multimodal_fusion = None
        if hasattr(args, 'mod') and args.mod == 'multi_lora':
            num_modalities = getattr(args, 'num_modalities', 3)
            fusion_type = getattr(args, 'fusion_type', 'cross_attention')
            fusion_embed_dim = getattr(args, 'fusion_embed_dim', 64)
            lora_rank = getattr(args, 'lora_rank', 4)
            use_modality_dropout = getattr(args, 'use_modality_dropout', True)
            modality_dropout_rate = getattr(args, 'modality_dropout_rate', 0.1)

            # Use efficient window-based implementation
            window_size = getattr(args, 'window_size', 8)
            reduction_factor = getattr(args, 'reduction_factor', 4)

            self.multimodal_fusion = MultiModalFusion(
                args=args,
                num_modalities=num_modalities,
                out_channels=in_chans,  # Match encoder input channels
                fusion_type=fusion_type,
                embed_dim=fusion_embed_dim,
                lora_rank=lora_rank,
                use_modality_dropout=use_modality_dropout,
                modality_dropout_rate=modality_dropout_rate,
                window_size=window_size,
                reduction_factor=reduction_factor
            )

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, 1024 // patch_size, 1024 // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        if args.mod == 'sam_adpt':
            block_class = AdapterBlock
        elif args.mod == 'sam_lora':
            block_class = LoraBlock
        elif args.mod == 'sam_adalora':
            block_class = AdaloraBlock
        elif args.mod == 'multi_lora':
            # For multi_lora, we use LoraBlock for the transformer blocks
            block_class = LoraBlock
        else:
            block_class = Block

        for i in range(depth):
            block = block_class(
                args=self.args,
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle multimodal input if using multi_lora
        if hasattr(self, 'multimodal_fusion') and self.multimodal_fusion is not None and self.args.mod == 'multi_lora':
            # For your data format, x already contains all modalities in channels
            # We just need to ensure we have the correct number of modalities
            if x.size(1) >= self.multimodal_fusion.num_modalities:
                # Pass the multimodal input through the fusion module
                # This will output a 3-channel tensor compatible with SAM
                x = self.multimodal_fusion(x)

        x = self.patch_embed(x)
        if self.pos_embed is not None:
            # resize position embedding to match the input
            new_abs_pos = F.interpolate(
                self.pos_embed.permute(0, 3, 1, 2),
                size=(x.shape[1], x.shape[2]),
                mode="bicubic",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            x = x + new_abs_pos

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x

# # This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
# class ImageEncoderViT(nn.Module):
#     def __init__(
#         self,
#         args,
#         img_size: int = 1024,
#         patch_size: int = 16,
#         in_chans: int = 3,
#         embed_dim: int = 768,
#         depth: int = 12,
#         num_heads: int = 12,
#         mlp_ratio: float = 4.0,
#         out_chans: int = 256,
#         qkv_bias: bool = True,
#         norm_layer: Type[nn.Module] = nn.LayerNorm,
#         act_layer: Type[nn.Module] = nn.GELU,
#         use_abs_pos: bool = True,
#         use_rel_pos: bool = False,
#         rel_pos_zero_init: bool = True,
#         window_size: int = 0,
#         global_attn_indexes: Tuple[int, ...] = (),
#     ) -> None:
#         """
#         Args:
#             img_size (int): Input image size.
#             patch_size (int): Patch size.
#             in_chans (int): Number of input image channels.
#             embed_dim (int): Patch embedding dimension.
#             depth (int): Depth of
#              ViT.
#             num_heads (int): Number of attention heads in each ViT block.
#             mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#             qkv_bias (bool): If True, add a learnable bias to query, key, value.
#             norm_layer (nn.Module): Normalization layer.
#             act_layer (nn.Module): Activation layer.
#             use_abs_pos (bool): If True, use absolute positional embeddings.
#             use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
#             rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
#             window_size (int): Window size for window attention blocks.
#             global_attn_indexes (list): Indexes for blocks using global attention.
#         """
#         super().__init__()
#         self.img_size = img_size
#         self.args = args
#
#         self.patch_embed = PatchEmbed(
#             kernel_size=(patch_size, patch_size),
#             stride=(patch_size, patch_size),
#             in_chans=in_chans,
#             embed_dim=embed_dim,
#         )
#
#         self.pos_embed: Optional[nn.Parameter] = None
#         if use_abs_pos:
#             # Initialize absolute positional embedding with pretrain image size.
#             self.pos_embed = nn.Parameter(
#                 torch.zeros(1, 1024 // patch_size, 1024 // patch_size, embed_dim)
#             )
#
#         self.blocks = nn.ModuleList()
#         if args.mod == 'sam_adpt':
#             block_class = AdapterBlock
#         elif args.mod == 'sam_lora':
#             block_class = LoraBlock
#         elif args.mod == 'sam_adalora':
#             block_class = AdaloraBlock
#         else:
#             block_class = Block
#
#         for i in range(depth):
#             block = block_class(
#                 args=self.args,
#                 dim=embed_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 norm_layer=norm_layer,
#                 act_layer=act_layer,
#                 use_rel_pos=use_rel_pos,
#                 rel_pos_zero_init=rel_pos_zero_init,
#                 window_size=window_size if i not in global_attn_indexes else 0,
#                 input_size=(img_size // patch_size, img_size // patch_size),
#             )
#             self.blocks.append(block)
#
#         self.neck = nn.Sequential(
#             nn.Conv2d(
#                 embed_dim,
#                 out_chans,
#                 kernel_size=1,
#                 bias=False,
#             ),
#             LayerNorm2d(out_chans),
#             nn.Conv2d(
#                 out_chans,
#                 out_chans,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False,
#             ),
#             LayerNorm2d(out_chans),
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#
#         x = self.patch_embed(x)
#         if self.pos_embed is not None:
#             # resize position embedding to match the input
#             new_abs_pos = F.interpolate(
#                 self.pos_embed.permute(0, 3, 1, 2),
#                 size=(x.shape[1], x.shape[2]),
#                 mode="bicubic",
#                 align_corners=False,
#             ).permute(0, 2, 3, 1)
#             x = x + new_abs_pos
#
#         for blk in self.blocks:
#             x = blk(x)
#
#         x = self.neck(x.permute(0, 3, 1, 2))
#
#         return x

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x

