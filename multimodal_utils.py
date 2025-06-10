#!/usr/bin/env python3

"""
Utilities for handling multimodal 5-channel input in SAM2
"""

import torch
import numpy as np

from diffusion_utils import diffusion_sam_transform


def modify_model_for_multimodal(model, in_channels=3):  # Changed default from 5 to 3
    """
    Modify the first convolutional layer to accept multiple input channels

    Args:
        model: The SAM2 model
        in_channels: Number of input channels (default: 3)

    Returns:
        Modified model
    """
    # Get the current weights of the first conv layer
    first_conv = None

    # Find the first conv layer in the image encoder
    if hasattr(model, 'image_encoder') and hasattr(model.image_encoder, 'layers'):
        # For Hiera architecture in SAM2
        first_conv = model.image_encoder.layers[0].proj

    if first_conv is not None and isinstance(first_conv, torch.nn.Conv2d):
        # Get current weights
        current_weights = first_conv.weight

        # If current_weights channels match in_channels, no need to modify
        if current_weights.shape[1] == in_channels:
            print(f"First conv layer already has {in_channels} input channels - no changes made")
            return model

        # Initialize new conv with more input channels
        new_conv = torch.nn.Conv2d(
            in_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )

        # Initialize new weights
        with torch.no_grad():
            if in_channels > current_weights.shape[1]:
                # For the first original channels, copy the pre-trained weights
                new_weights = torch.zeros(current_weights.shape[0], in_channels,
                                          current_weights.shape[2], current_weights.shape[3],
                                          device=current_weights.device)
                new_weights[:, :current_weights.shape[1], :, :] = current_weights

                # Initialize additional channels with mean of existing weights
                channel_mean = current_weights.mean(dim=1, keepdim=True)
                for i in range(current_weights.shape[1], in_channels):
                    new_weights[:, i:i + 1, :, :] = channel_mean

                new_conv.weight.copy_(new_weights)
            else:
                # Just use the first N channels if in_channels < original channels
                new_conv.weight.copy_(current_weights[:, :in_channels, :, :])

            if first_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias)

        # Replace the first conv layer
        if hasattr(model, 'image_encoder') and hasattr(model.image_encoder, 'layers'):
            model.image_encoder.layers[0].proj = new_conv

        print(f"Modified first conv layer to accept {in_channels} input channels")

    return model

def normalize_multimodal_volume(volume_data, normalize_per_channel=True):
    """
    Normalize a multimodal volume (5 x Slice x H x W)

    Args:
        volume_data: Multimodal volume with shape (5 x Slice x H x W)
        normalize_per_channel: Whether to normalize each channel separately

    Returns:
        Normalized volume with the same shape
    """
    # Get dimensions
    n_channels = volume_data.shape[0]

    if normalize_per_channel:
        # Normalize each channel independently
        normalized_volume = np.zeros_like(volume_data, dtype=np.float32)

        for c in range(n_channels):
            channel_data = volume_data[c]
            channel_min = channel_data.min()
            channel_max = channel_data.max()

            if channel_max > channel_min:
                normalized_volume[c] = (channel_data - channel_min) / (channel_max - channel_min)
            else:
                normalized_volume[c] = channel_data

        return normalized_volume
    else:
        # Normalize based on global min and max
        data_min = volume_data.min()
        data_max = volume_data.max()

        if data_max > data_min:
            return (volume_data - data_min) / (data_max - data_min)
        else:
            return volume_data


def samify_volume_with_diffusion(volume_tensor, steps=10, model=None, ddim=False):
    """Convert a 3-channel volume to SAM-compatible representation using diffusion."""
    if not isinstance(volume_tensor, torch.Tensor):
        volume_tensor = torch.tensor(volume_tensor, dtype=torch.float32)
    if volume_tensor.dim() == 3:
        volume_tensor = volume_tensor.unsqueeze(0)
    return diffusion_sam_transform(volume_tensor, steps=steps, model=model, ddim=ddim)
