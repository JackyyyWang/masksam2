# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Spacing-aware memory module for medical imaging applications.

This module modifies attention scores based on physical distance between consecutive slices
in 3D medical volumes. When using SAM2 for medical imaging, each 2D slice is treated as a
"video frame", and this module applies distance-based attention decay to prevent spurious
segmentation propagation across large inter-slice gaps.

Key features:
- Physical distance-based attention decay (in millimeters)
- Support for uniform slice spacing (typical in medical imaging)
- Medical imaging validation (reasonable spacing ranges)
- Robust error handling for missing or invalid spacing data
- Multiple decay functions: exponential, sigmoid, gaussian
"""

from __future__ import annotations
from typing import Optional
import torch
from torch import nn
import numpy as np


class SpacingAwareMemoryModule(nn.Module):
    """
    Spacing-aware memory module for medical imaging applications.

    This module modifies attention scores based on the physical distance between
    consecutive slices in a 3D medical volume. It provides distance-based decay
    to attention weights, helping prevent spurious segmentation propagation
    across large anatomical gaps.

    Args:
        decay_type: Type of decay function ('exponential', 'sigmoid', 'gaussian')
        num_heads: Number of attention heads (for dimension alignment)
        max_distance: Maximum expected distance in mm (for normalization)
        min_decay: Minimum decay value to prevent complete attention loss
        medical_validation: Whether to apply medical imaging specific validation
    """

    def __init__(
            self,
            decay_type: str = "exponential",
            num_heads: int = 8,
            max_distance: float = 100.0,
            min_decay: float = 0.01,
            medical_validation: bool = True,
    ) -> None:
        super().__init__()
        self.decay_type = decay_type
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.min_decay = min_decay
        self.medical_validation = medical_validation

        # Learnable decay parameters for each function
        if decay_type == "exponential":
            # f(d) = exp(-rate * d)
            self.decay_rate = nn.Parameter(torch.tensor(0.1))
        elif decay_type == "sigmoid":
            # f(d) = sigmoid(-alpha * (d - beta))
            self.sigmoid_alpha = nn.Parameter(torch.tensor(1.0))
            self.sigmoid_beta = nn.Parameter(torch.tensor(10.0))  # Inflection point in mm
        elif decay_type == "gaussian":
            # f(d) = exp(-0.5 * (d / sigma)^2)
            self.gaussian_sigma = nn.Parameter(torch.tensor(15.0))  # Standard deviation in mm
        else:
            raise ValueError(f"Unknown decay type: {decay_type}. Choose from ['exponential', 'sigmoid', 'gaussian']")

    def forward(
            self,
            attention_scores: torch.Tensor,
            spacings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply spacing-based decay to attention scores. Standard nn.Module forward method.
        This is a wrapper for apply_spacing_decay for conventional PyTorch usage.

        Args:
            attention_scores: [..., seq_len, seq_len] attention logits for frame tokens
            spacings: [seq_len-1] or [batch, seq_len-1] physical spacings in millimeters

        Returns:
            Modified attention scores with distance-based decay
        """
        return self.apply_spacing_decay(attention_scores, spacings)

    def apply_spacing_decay(
            self,
            attention_scores: torch.Tensor,
            spacings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply spacing-based decay to attention scores for medical slice sequences.

        Args:
            attention_scores: [..., seq_len, seq_len] attention logits for frame tokens only
                             (object pointer tokens should be excluded before calling this)
            spacings: [seq_len-1] physical spacings in millimeters between consecutive slices
                     For uniform spacing, this can be created from a single spacing value

        Returns:
            Attention scores modified with distance-based decay applied in log space
        """
        if spacings is None or spacings.numel() == 0:
            return attention_scores

        # Ensure spacings is on the correct device
        spacings = spacings.to(attention_scores.device)

        # Validate and prepare spacings tensor
        spacings = self._validate_and_prepare_spacings(spacings, attention_scores)
        if spacings is None:
            return attention_scores  # Fallback to original attention if validation fails

        # Compute distance matrix between all slice pairs
        distances = self._compute_distance_matrix(spacings)

        # Compute decay weights based on distances
        decay_weights = self._compute_decay_weights(distances)

        # Ensure minimum decay to prevent complete attention loss
        decay_weights = torch.clamp(decay_weights, min=self.min_decay)

        # Align dimensions for broadcasting over attention heads
        while decay_weights.dim() < attention_scores.dim():
            decay_weights = decay_weights.unsqueeze(-3)  # Insert before last two dimensions

        # Apply decay in log space for numerical stability before softmax
        log_decay = torch.log(decay_weights + 1e-9)
        scaled_scores = attention_scores + log_decay

        return scaled_scores

    def _validate_and_prepare_spacings(
            self,
            spacings: torch.Tensor,
            attention_scores: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Validate and prepare spacing tensor for medical imaging context.

        Args:
            spacings: Input spacing tensor
            attention_scores: Attention tensor to match dimensions with

        Returns:
            Validated and properly shaped spacing tensor, or None if validation fails
        """
        # Handle scalar spacing (convert to tensor)
        if spacings.dim() == 0:
            spacings = spacings.unsqueeze(0)

        # Remove extra batch dimensions if present
        if spacings.dim() > 1:
            spacings = spacings.squeeze()
            if spacings.dim() > 1:
                # If still multi-dimensional, take the first batch
                spacings = spacings[0]

        # Validate sequence length compatibility
        seq_len = attention_scores.size(-1)
        expected_spacing_len = seq_len - 1
        actual_spacing_len = spacings.size(0) if spacings.dim() > 0 else 0

        # Handle dimension mismatch
        if actual_spacing_len != expected_spacing_len:
            if actual_spacing_len == 1:
                # Single spacing value - expand to uniform spacing
                uniform_spacing = spacings[0]
                spacings = torch.full((expected_spacing_len,), uniform_spacing,
                                      device=spacings.device, dtype=spacings.dtype)
            elif actual_spacing_len > expected_spacing_len:
                # Truncate spacings
                spacings = spacings[:expected_spacing_len]
            elif actual_spacing_len > 0:
                # Pad with mean of existing spacings
                mean_spacing = spacings.mean()
                pad_length = expected_spacing_len - actual_spacing_len
                pad = torch.full((pad_length,), mean_spacing,
                                 device=spacings.device, dtype=spacings.dtype)
                spacings = torch.cat([spacings, pad], dim=0)
            else:
                # No valid spacings provided
                if expected_spacing_len > 0:
                    # Create default uniform spacing of 1.0mm
                    spacings = torch.ones(expected_spacing_len,
                                          device=attention_scores.device,
                                          dtype=attention_scores.dtype)
                else:
                    return None

        # Medical imaging validation
        if self.medical_validation:
            if not self._validate_medical_spacing(spacings):
                return None

        return spacings

    def _validate_medical_spacing(self, spacings: torch.Tensor) -> bool:
        """
        Validate spacing values for medical imaging context.

        Args:
            spacings: Spacing tensor to validate

        Returns:
            True if spacing values are reasonable for medical imaging
        """
        if spacings.numel() == 0:
            return False

        spacing_values = spacings.detach().cpu().numpy()

        # Basic validation: all positive values
        if np.any(spacing_values <= 0):
            print(f"Warning: Non-positive spacing values detected: {spacing_values}")
            return False

        # Medical imaging range validation
        min_spacing = np.min(spacing_values)
        max_spacing = np.max(spacing_values)
        mean_spacing = np.mean(spacing_values)

        # Typical medical imaging ranges
        if min_spacing < 0.1:  # Less than 0.1mm is unusual
            print(f"Warning: Very small spacing detected (min: {min_spacing:.3f}mm)")

        if max_spacing > 50:  # Greater than 50mm is very unusual
            print(f"Warning: Very large spacing detected (max: {max_spacing:.1f}mm)")
            return False  # This is likely an error

        if mean_spacing > 25:  # Mean > 25mm is unusual for most applications
            print(f"Warning: Large mean spacing ({mean_spacing:.1f}mm) - may affect attention patterns")

        return True

    def _compute_distance_matrix(self, spacings: torch.Tensor) -> torch.Tensor:
        """
        Compute cumulative distances between all pairs of slices.

        Args:
            spacings: [seq_len-1] spacing between consecutive slices in mm

        Returns:
            [seq_len, seq_len] matrix of pairwise distances in mm
        """
        seq_len = spacings.size(0) + 1

        # Compute cumulative positions (distances from the first slice)
        # Position 0 is at distance 0, position 1 is at spacings[0], etc.
        cum_positions = torch.zeros(seq_len, device=spacings.device, dtype=spacings.dtype)
        cum_positions[1:] = torch.cumsum(spacings, dim=0)

        # Compute pairwise distances: |pos_i - pos_j|
        positions_i = cum_positions[:, None]  # [seq_len, 1]
        positions_j = cum_positions[None, :]  # [1, seq_len]
        distances = torch.abs(positions_i - positions_j)

        return distances

    def _compute_decay_weights(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Compute decay weights based on distances between slices.

        Args:
            distances: [seq_len, seq_len] matrix of pairwise distances in mm

        Returns:
            [seq_len, seq_len] matrix of decay weights (0-1 range)
        """
        if self.decay_type == "exponential":
            # Exponential decay: exp(-rate * distance)
            rate = torch.clamp(self.decay_rate, min=0.001, max=1.0)  # Prevent extreme values
            weights = torch.exp(-rate * distances)

        elif self.decay_type == "sigmoid":
            # Sigmoid decay: sigmoid(-alpha * (distance - beta))
            # Beta is the inflection point, alpha controls steepness
            alpha = torch.clamp(self.sigmoid_alpha, min=0.1, max=10.0)
            beta = torch.clamp(self.sigmoid_beta, min=0.0, max=50.0)
            weights = torch.sigmoid(-alpha * (distances - beta))

        elif self.decay_type == "gaussian":
            # Gaussian decay: exp(-0.5 * (distance / sigma)^2)
            # Sigma controls the width of the Gaussian
            sigma = torch.clamp(self.gaussian_sigma, min=1.0, max=100.0)
            weights = torch.exp(-0.5 * (distances / sigma) ** 2)

        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")

        return weights

    def compute_memory_weights(
            self,
            frame_positions: torch.Tensor,
            spacings: torch.Tensor,
            current_idx: int,
    ) -> torch.Tensor:
        """
        Compute normalized memory weights for specific frames relative to current frame.

        Args:
            frame_positions: [num_frames] indices of frames in the memory bank
            spacings: [seq_len-1] spacing values in millimeters
            current_idx: Index of the current frame

        Returns:
            [num_frames] tensor of normalized weights (sum to 1)
        """
        spacings = self._validate_and_prepare_spacings(spacings,
                                                       torch.zeros(len(spacings) + 1, len(spacings) + 1))
        if spacings is None:
            # Fallback to uniform weights
            return torch.ones(len(frame_positions)) / len(frame_positions)

        distances = self._compute_distances_from_frame(frame_positions, spacings, current_idx)
        weights = self._compute_decay_weights(distances.unsqueeze(0)).squeeze(0)

        # Normalize weights
        return weights / (weights.sum() + 1e-8)

    def _compute_distances_from_frame(
            self,
            frame_positions: torch.Tensor,
            spacings: torch.Tensor,
            current_idx: int
    ) -> torch.Tensor:
        """
        Compute distances of specific frames from a target frame.

        Args:
            frame_positions: [num_frames] frame indices
            spacings: [seq_len-1] spacing between consecutive slices
            current_idx: Index of the current frame

        Returns:
            [num_frames] distances in mm
        """
        # Compute cumulative positions
        seq_len = spacings.size(0) + 1
        cum_positions = torch.zeros(seq_len, device=spacings.device, dtype=spacings.dtype)
        cum_positions[1:] = torch.cumsum(spacings, dim=0)

        # Get current frame position
        current_pos = cum_positions[current_idx]

        # Get positions of requested frames
        frame_positions = torch.clamp(frame_positions, 0, seq_len - 1)  # Ensure valid indices
        frame_pos = cum_positions[frame_positions]

        # Compute distances
        distances = torch.abs(frame_pos - current_pos)

        return distances

    def get_pruning_mask(
            self,
            frame_positions: torch.Tensor,
            spacings: torch.Tensor,
            current_idx: int,
            threshold: float = 0.1,
    ) -> torch.Tensor:
        """
        Return a mask indicating which memories to keep based on distance.

        Args:
            frame_positions: [num_frames] frame indices in the memory bank
            spacings: [seq_len-1] spacing values in millimeters  
            current_idx: Index of the current frame
            threshold: Minimum weight threshold for keeping a memory

        Returns:
            [num_frames] boolean mask (True = keep, False = prune)
        """
        weights = self.compute_memory_weights(frame_positions, spacings, current_idx)
        return weights > threshold

    def get_effective_attention_range(self, spacings: torch.Tensor, threshold: float = 0.1) -> float:
        """
        Compute the effective attention range given current decay parameters.

        Args:
            spacings: [seq_len-1] spacing values in millimeters
            threshold: Minimum attention weight threshold

        Returns:
            Effective range in millimeters where attention > threshold
        """
        if spacings is None or spacings.numel() == 0:
            return float('inf')

        # Create a test distance range
        max_distance = spacings.sum() * 2  # Test up to 2x total volume extent
        test_distances = torch.linspace(0, max_distance, 1000, device=spacings.device)

        # Compute weights for test distances
        if self.decay_type == "exponential":
            rate = torch.clamp(self.decay_rate, min=0.001, max=1.0)
            weights = torch.exp(-rate * test_distances)
        elif self.decay_type == "sigmoid":
            alpha = torch.clamp(self.sigmoid_alpha, min=0.1, max=10.0)
            beta = torch.clamp(self.sigmoid_beta, min=0.0, max=50.0)
            weights = torch.sigmoid(-alpha * (test_distances - beta))
        elif self.decay_type == "gaussian":
            sigma = torch.clamp(self.gaussian_sigma, min=1.0, max=100.0)
            weights = torch.exp(-0.5 * (test_distances / sigma) ** 2)

        # Find the distance where weight drops below threshold
        above_threshold = weights > threshold
        if not torch.any(above_threshold):
            return 0.0

        # Find the last distance above threshold
        last_valid_idx = torch.where(above_threshold)[0][-1]
        effective_range = test_distances[last_valid_idx].item()

        return effective_range

    def create_uniform_spacing_tensor(
            self,
            num_slices: int,
            spacing_mm: float,
            device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Helper function to create uniform spacing tensor from a single spacing value.
        This is useful when your dataset provides a single spacing number per volume.

        Args:
            num_slices: Number of slices in the volume
            spacing_mm: Uniform spacing between consecutive slices in mm
            device: Device to place the tensor on

        Returns:
            [num_slices-1] tensor of uniform spacings
        """
        if num_slices <= 1:
            return torch.empty(0, dtype=torch.float32, device=device)

        if device is None:
            device = torch.device('cpu')

        return torch.full((num_slices - 1,), spacing_mm, dtype=torch.float32, device=device)

    def __repr__(self) -> str:
        return (f"SpacingAwareMemoryModule("
                f"decay_type='{self.decay_type}', "
                f"max_distance={self.max_distance}, "
                f"min_decay={self.min_decay}, "
                f"medical_validation={self.medical_validation})")


# Utility functions for medical imaging applications

def create_spacing_from_metadata(dicom_metadata: dict = None, nifti_header=None) -> float:
    """
    Extract spacing information from medical image metadata.

    Args:
        dicom_metadata: DICOM metadata dictionary  
        nifti_header: NIfTI header object

    Returns:
        Spacing in millimeters, or 1.0 as default
    """
    if dicom_metadata is not None:
        # Try different DICOM tags for spacing
        for tag in ['SliceThickness', 'SpacingBetweenSlices', 'PixelSpacing']:
            if tag in dicom_metadata:
                spacing = dicom_metadata[tag]
                if isinstance(spacing, (list, tuple)):
                    spacing = spacing[0] if len(spacing) > 0 else 1.0
                return float(spacing)

    if nifti_header is not None:
        # Extract from NIfTI header
        if hasattr(nifti_header, 'get_zooms'):
            zooms = nifti_header.get_zooms()
            if len(zooms) >= 3:
                return float(zooms[2])  # Z-dimension spacing

    # Default fallback
    print("Warning: Could not extract spacing from metadata, using 1.0mm default")
    return 1.0


def validate_medical_volume_spacing(volume_shape: tuple, spacing_mm: float) -> bool:
    """
    Validate that volume and spacing combination is reasonable for medical imaging.

    Args:
        volume_shape: Shape of the 3D volume (depth, height, width)
        spacing_mm: Spacing between slices in mm

    Returns:
        True if combination seems reasonable
    """
    if len(volume_shape) < 3:
        return False

    depth = volume_shape[0]
    total_thickness = (depth - 1) * spacing_mm

    # Reasonable total thickness ranges for different anatomies
    if total_thickness < 10:  # Less than 1cm
        print(f"Warning: Very thin volume ({total_thickness:.1f}mm total)")
        return total_thickness > 1  # At least 1mm total

    if total_thickness > 1000:  # More than 1 meter
        print(f"Warning: Very thick volume ({total_thickness:.1f}mm total)")
        return False

    return True