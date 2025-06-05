import torch
import torch.nn as nn

from diffusion_fusion import LightweightDiffusionFusion


class MultiChannelDiffusionBlock(nn.Module):
    """Replace naive fusion with diffusion-based approach."""

    def __init__(self, config):
        super().__init__()
        self.diffusion = LightweightDiffusionFusion(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input shape [B, C, H, W] with C=num_modalities
        return self.diffusion.fast_sample(x, num_steps=self.diffusion.timesteps)

