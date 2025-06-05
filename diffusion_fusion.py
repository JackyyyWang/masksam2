import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightUNet(nn.Module):
    """Simplified U-Net used for diffusion denoising."""

    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: int = 128,
                 base_channels: int = 64, channel_multipliers=None):
        super().__init__()
        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4, 8]
        self.time_embed = nn.Embedding(10, time_embedding_dim)
        self.enc1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.enc2 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=1)
        self.enc3 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.out_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_embed(t)
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(self.pool(x1)))
        x3 = F.relu(self.enc3(self.pool(x2)))
        x = F.relu(self.up2(x3))
        x = F.relu(self.up1(x + x2))
        out = self.out_conv(x + x1)
        return out


def ms_ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Placeholder multi-scale SSIM (single scale for simplicity)."""
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = ((x - mu_x) ** 2).mean()
    sigma_y = ((y - mu_y) ** 2).mean()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
    return ssim.clamp(0, 1)


def sobel_edges(img: torch.Tensor) -> torch.Tensor:
    """Compute Sobel edges for a batch of images."""
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    edges_x = F.conv2d(img, sobel_x, padding=1)
    edges_y = F.conv2d(img, sobel_y, padding=1)
    return torch.sqrt(edges_x ** 2 + edges_y ** 2)


def mutual_information(x: torch.Tensor, y: torch.Tensor, bins: int = 32) -> torch.Tensor:
    """Approximate mutual information between two images."""
    x_hist = torch.histc(x, bins=bins, min=0.0, max=1.0)
    y_hist = torch.histc(y, bins=bins, min=0.0, max=1.0)
    joint_hist = torch.histc(torch.stack([x, y], dim=-1).view(-1, 2).sum(1), bins=bins, min=0.0, max=2.0)
    p_x = x_hist / x_hist.sum()
    p_y = y_hist / y_hist.sum()
    p_xy = joint_hist / joint_hist.sum()
    mi = (p_xy * (torch.log(p_xy + 1e-8) - torch.log(p_x.unsqueeze(1) + 1e-8) - torch.log(p_y.unsqueeze(0) + 1e-8))).sum()
    return mi


class StructurePreservationNet(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class PerceptualLoss(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(x, y)


class LightweightDiffusionFusion(nn.Module):
    """Fast-DDPM fusion with structure preservation"""

    def __init__(self, config):
        super().__init__()
        self.timesteps = 10
        self.channels = config.num_modalities
        self.register_buffer('betas', torch.linspace(0.0001, 0.02, self.timesteps))
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.denoiser = LightweightUNet(self.channels, 3, time_embedding_dim=128, base_channels=64)
        self.structure_encoder = StructurePreservationNet()
        self.perceptual_loss = PerceptualLoss()

    def add_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x)
        a_bar = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return torch.sqrt(a_bar) * x + torch.sqrt(1 - a_bar) * noise

    def denoise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.denoiser(x, t)

    def structure_preservation_loss(self, x_orig: torch.Tensor, x_fused: torch.Tensor):
        L_perceptual = self.perceptual_loss(x_fused, x_orig)
        L_ssim = 1 - ms_ssim(x_fused, x_orig)
        edges_orig = sobel_edges(x_orig)
        edges_fused = sobel_edges(x_fused)
        L_edge = F.mse_loss(edges_fused, edges_orig)
        L_mi = -mutual_information(x_fused, x_orig)
        return {
            'perceptual': L_perceptual,
            'ssim': L_ssim,
            'edge': L_edge,
            'mutual_info': L_mi,
            'total': L_perceptual + L_ssim + L_edge + L_mi,
        }

    def fast_sample(self, x_condition: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        x = torch.randn_like(x_condition)
        for i in reversed(range(num_steps)):
            t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
            eps_pred = self.denoiser(x, t)
            a_t = self.alphas_cumprod[i]
            a_prev = self.alphas_cumprod[i - 1] if i > 0 else torch.tensor(1.0, device=x.device)
            x_pred = (x - torch.sqrt(1 - a_t) * eps_pred) / torch.sqrt(a_t)
            if i > 0:
                x = torch.sqrt(a_prev) * x_pred + torch.sqrt(1 - a_prev) * eps_pred
            else:
                x = x_pred
        return self.normalize_to_rgb(x)

    @staticmethod
    def normalize_to_rgb(x: torch.Tensor) -> torch.Tensor:
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        return (x * 255.0).clamp(0, 255)

