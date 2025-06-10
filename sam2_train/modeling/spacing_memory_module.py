# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Spacing-aware memory module.

This module modifies attention scores based on physical distance between slices.
It can be used with any attention mechanism by calling ``apply_spacing_decay``
on the pre-softmax attention scores. It also provides utilities for computing
memory weights and pruning distant memories.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class SpacingAwareMemoryModule(nn.Module):
    """Standalone module for spacing-aware memory decay."""

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__()
        config = config or {}
        self.config = {
            "max_distance": config.get("max_distance", 30.0),
            "num_heads": config.get("num_heads", 8),
            "decay_type": config.get("decay_type", "exponential"),
            "temperature": config.get("temperature", 1.0),
        }

        # Learnable decay parameters
        self.decay_rate = nn.Parameter(torch.tensor(0.1))
        self.decay_alpha = nn.Parameter(torch.tensor(1.0))
        self.decay_beta = nn.Parameter(torch.tensor(5.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply_spacing_decay(
        self,
        attention_scores: torch.Tensor,
        spacings: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        """Apply spacing-based decay to attention scores.

        Args:
            attention_scores: ``[..., seq_len, seq_len]`` attention logits.
            spacings: ``[seq_len-1]`` or ``[batch, seq_len-1]`` physical
                spacings in millimeters.
            dim: dimension of ``attention_scores`` that represents the key
                sequence length. Defaults to ``-1``.

        Returns:
            Attention scores modified with distance-based decay.
        """
        distances = self._compute_distance_matrix(spacings)
        decay_weights = self._compute_decay_weights(distances)

        # Align dimensions for broadcasting
        while decay_weights.dim() < attention_scores.dim():
            decay_weights = decay_weights.unsqueeze(1)

        scaled_scores = attention_scores + torch.log(decay_weights + 1e-8)
        return scaled_scores

    def compute_memory_weights(
        self,
        frame_positions: torch.Tensor,
        spacings: torch.Tensor,
        current_idx: int,
    ) -> torch.Tensor:
        """Compute normalized memory weights for frames.

        Args:
            frame_positions: ``[num_frames]`` frame indices stored in the memory
                bank.
            spacings: ``[seq_len-1]`` spacing values in millimeters.
            current_idx: index of the current frame.
        Returns:
            ``[num_frames]`` tensor of weights summing to 1.
        """
        distances = self._compute_distances_from_frame(
            frame_positions, spacings, current_idx
        )
        weights = self._compute_decay_weights(distances)
        weights = weights / (weights.sum() + 1e-8)
        return weights

    def get_pruning_mask(
        self,
        frame_positions: torch.Tensor,
        spacings: torch.Tensor,
        current_idx: int,
        threshold: float = 0.1,
    ) -> torch.Tensor:
        """Return a mask indicating which memories to keep.

        Memories with weight smaller than ``threshold`` will be pruned.
        """
        weights = self.compute_memory_weights(frame_positions, spacings, current_idx)
        return weights > threshold

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------
    def _compute_distance_matrix(self, spacings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise physical distances between frames."""
        if spacings.dim() == 1:
            spacings = spacings.unsqueeze(0)
        batch_size, seq_len_minus1 = spacings.shape
        seq_len = seq_len_minus1 + 1

        cum_dist = torch.zeros(batch_size, seq_len, device=spacings.device)
        cum_dist[:, 1:] = torch.cumsum(spacings, dim=1)

        dist_matrix = torch.abs(cum_dist[:, :, None] - cum_dist[:, None, :])
        return dist_matrix

    def _compute_distances_from_frame(
        self, frame_positions: torch.Tensor, spacings: torch.Tensor, idx: int
    ) -> torch.Tensor:
        """Compute distance of each frame from ``idx``."""
        if spacings.dim() == 1:
            spacings = spacings.unsqueeze(0)
        batch_size = spacings.shape[0]
        assert batch_size == 1, "Batch spacings only supported for batch size 1"
        dist_matrix = self._compute_distance_matrix(spacings)[0]
        distances = dist_matrix[idx, frame_positions]
        return distances

    def _compute_decay_weights(self, distances: torch.Tensor) -> torch.Tensor:
        """Compute decay weights based on configured decay type."""
        if self.config["decay_type"] == "exponential":
            weights = torch.exp(-self.decay_rate * distances)
        elif self.config["decay_type"] == "sigmoid":
            weights = 1.0 / (
                1.0 + torch.exp(self.decay_alpha * (distances - self.decay_beta))
            )
        elif self.config["decay_type"] == "gaussian":
            sigma = self.decay_beta.clamp(min=1e-6)
            weights = torch.exp(-0.5 * (distances / sigma) ** 2)
        else:
            raise ValueError(f"Unknown decay type {self.config['decay_type']}")

        # Ensure weights go to zero as distance approaches infinity
        weights = weights.clamp(min=1e-8)
        return weights

