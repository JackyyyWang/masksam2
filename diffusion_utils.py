import torch
import torch.nn as nn
import torch.nn.functional as F


def beta_schedule(start=0.0001, end=0.02, steps=10):
    """Linear schedule for beta values."""
    return torch.linspace(start, end, steps)


def forward_diffusion(x0, betas):
    """Perform the forward diffusion process."""
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    noise = torch.randn_like(x0)
    t = len(betas) - 1
    return torch.sqrt(alphas_bar[t]) * x0 + torch.sqrt(1 - alphas_bar[t]) * noise


class SimpleUNet(nn.Module):
    """Lightweight U-Net used for denoising."""

    def __init__(self, in_channels=3, base_channels=16):
        super().__init__()
        self.enc1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.enc2 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dec1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.out_conv = nn.Conv2d(base_channels + base_channels, in_channels, 3, padding=1)

    def forward(self, x, t_embed=None):
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(self.pool(x1)))
        x3 = F.relu(self.dec1(x2))
        x3 = torch.cat([x3, x1], dim=1)
        return self.out_conv(x3)


def reverse_diffusion(x_t, model, betas, ddim=False):
    """Perform the reverse diffusion to obtain x0."""
    x = x_t
    for step in reversed(range(len(betas))):
        sigma = betas[step].sqrt() if not ddim else 0.0
        mu = model(x, torch.tensor([step], device=x.device))
        if step > 0:
            noise = torch.randn_like(x) if not ddim else 0.0
            x = mu + sigma * noise
        else:
            x = mu
    return x


def diffusion_sam_transform(x, steps=10, model=None, ddim=False):
    """Run diffusion-based transformation to produce SAM-compatible images."""
    betas = beta_schedule(steps=steps)
    x_t = forward_diffusion(x, betas)
    if model is None:
        model = SimpleUNet(in_channels=x.shape[1])
    model.eval()
    with torch.no_grad():
        x_recon = reverse_diffusion(x_t, model, betas, ddim=ddim)
    return x_recon
