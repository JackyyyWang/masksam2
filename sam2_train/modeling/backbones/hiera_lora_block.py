import torch
import torch.nn as nn
import torch.nn.functional as F
from sam_models.common.loralib import layers as lora


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)
    return x


class LoraAttention(nn.Module):
    def __init__(self, dim: int, dim_out: int, num_heads: int, q_pool: nn.Module = None, r: int = 4, lora_alpha: int = 1) -> None:
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim ** -0.5
        self.q_pool = q_pool
        self.qkv = lora.MergedLinear(
            dim,
            dim_out * 3,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            enable_lora=[True, True, True],
            fan_in_fan_out=True,
            merge_weights=False,
            bias=True,
        )
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        q, k, v = torch.unbind(qkv, 2)
        if self.q_pool is not None:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]
            q = q.reshape(B, H * W, self.num_heads, -1)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        x = attn.transpose(1, 2).reshape(B, H, W, -1)
        x = self.proj(x)
        return x


class LoraMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, act_layer=nn.GELU, drop: float = 0.0, r: int = 4, lora_alpha: int = 1) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = lora.Linear(in_features, hidden_features, r=r, lora_alpha=lora_alpha)
        self.act = act_layer()
        self.fc2 = lora.Linear(hidden_features, out_features, r=r, lora_alpha=lora_alpha)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
