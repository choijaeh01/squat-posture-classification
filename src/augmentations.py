"""
Lightweight data augmentation utilities for IMU window tensors.

Each transform consumes a tensor shaped `(channels, timesteps)` and returns a
tensor with the same shape. The functions can be composed with the helper
`compose_transforms` to build simple augmentation pipelines.

Included transforms:
    * noise injection and scaling
    * time shifts, stretch, warping, cropping
    * magnitude warping and channel dropout
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch

TensorTransform = Callable[[torch.Tensor], torch.Tensor]


def compose_transforms(transforms: Sequence[TensorTransform]) -> TensorTransform:
    """Compose multiple tensor transforms into a single callable."""

    def _inner(tensor: torch.Tensor) -> torch.Tensor:
        for transform in transforms:
            tensor = transform(tensor)
        return tensor

    return _inner


def add_gaussian_noise(std: float = 0.01) -> TensorTransform:
    """Return a transform that injects element-wise Gaussian noise."""

    def _inner(tensor: torch.Tensor) -> torch.Tensor:
        if std <= 0:
            return tensor
        return tensor + torch.randn_like(tensor) * std

    return _inner


def random_time_shift(max_shift: int = 5) -> TensorTransform:
    """Circularly shift the sequence along the time dimension."""

    def _inner(tensor: torch.Tensor) -> torch.Tensor:
        if max_shift <= 0:
            return tensor
        shift = torch.randint(-max_shift, max_shift + 1, size=()).item()
        if shift == 0:
            return tensor
        return torch.roll(tensor, shifts=shift, dims=-1)

    return _inner


def random_scaling(min_scale: float = 0.9, max_scale: float = 1.1) -> TensorTransform:
    """Apply channel-wise scaling sampled from a uniform distribution."""

    if min_scale <= 0 or max_scale <= 0:
        raise ValueError("Scaling factors must be positive.")
    if min_scale > max_scale:
        raise ValueError("`min_scale` cannot be greater than `max_scale`.")

    def _inner(tensor: torch.Tensor) -> torch.Tensor:
        if torch.isclose(torch.tensor(min_scale), torch.tensor(max_scale)):
            scale = min_scale
        else:
            scale = torch.empty(tensor.size(0)).uniform_(min_scale, max_scale)
        return tensor * scale.unsqueeze(-1)

    return _inner


def normalize(mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> TensorTransform:
    """Normalize the tensor using per-channel statistics."""

    if mean.ndim != 1 or std.ndim != 1:
        raise ValueError("Mean and std tensors must be 1D (per-channel statistics).")
    if mean.shape != std.shape:
        raise ValueError("Mean and std tensors must share the same shape.")

    def _inner(tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - mean.unsqueeze(-1)) / (std.unsqueeze(-1) + eps)

    return _inner


def random_time_stretch(min_scale: float = 0.8, max_scale: float = 1.25) -> TensorTransform:
    """
    Uniformly stretch or compress the timeline and resample back to the original length.

    Args:
        min_scale: Lower bound for the stretch factor (values < 1 speed up the motion).
        max_scale: Upper bound for the stretch factor (values > 1 slow down the motion).
    """

    if min_scale <= 0 or max_scale <= 0:
        raise ValueError("Scale factors must be positive.")
    if min_scale > max_scale:
        raise ValueError("`min_scale` cannot be greater than `max_scale`.")

    def _inner(tensor: torch.Tensor) -> torch.Tensor:
        if torch.isclose(torch.tensor(min_scale), torch.tensor(max_scale)):
            scale = float(min_scale)
        else:
            scale = float(torch.empty(1).uniform_(min_scale, max_scale).item())

        if np.isclose(scale, 1.0):
            return tensor

        length = tensor.size(-1)
        base_positions = np.linspace(0, length - 1, length, dtype=np.float32)
        warped_positions = np.clip(base_positions / scale, 0, length - 1)
        return _resample_to_positions(tensor, warped_positions)

    return _inner


def random_window_crop(min_ratio: float = 0.8) -> TensorTransform:
    """
    Crop a random sub-window and resample it to the original length.

    Smaller `min_ratio` values simulate quicker repetitions by keeping only a
    fraction of the window, while larger values apply gentler changes.
    """

    if not 0 < min_ratio <= 1.0:
        raise ValueError("`min_ratio` must be in the (0, 1] range.")

    def _inner(tensor: torch.Tensor) -> torch.Tensor:
        length = tensor.size(-1)
        if min_ratio == 1.0:
            return tensor

        rng = np.random.default_rng()
        ratio = float(rng.uniform(min_ratio, 1.0))
        crop_len = max(2, int(round(length * ratio)))
        crop_len = min(crop_len, length)

        max_start = length - crop_len
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        end = start + crop_len - 1

        warped_positions = np.linspace(start, end, length, dtype=np.float32)
        return _resample_to_positions(tensor, warped_positions)

    return _inner


def random_time_warp(max_warp: float = 0.2) -> TensorTransform:
    """
    Apply a smooth, non-linear warp along the time axis.

    Args:
        max_warp: Controls the magnitude of local speed changes. Values around
                  0.1-0.3 usually work well; near-zero values leave the window
                  almost unchanged.
    """

    if max_warp < 0:
        raise ValueError("`max_warp` must be non-negative.")

    def _inner(tensor: torch.Tensor) -> torch.Tensor:
        if max_warp == 0:
            return tensor

        length = tensor.size(-1)
        base = np.linspace(0, 1, length, dtype=np.float32)
        noise = np.random.normal(loc=1.0, scale=max_warp, size=length).astype(np.float32)
        noise = np.maximum(noise, 1e-3)
        cumulative = np.cumsum(noise)
        cumulative = (cumulative - cumulative.min()) / (cumulative.max() - cumulative.min() + 1e-6)
        warped_positions = np.interp(base, cumulative, base) * (length - 1)
        return _resample_to_positions(tensor, warped_positions.astype(np.float32))

    return _inner


def magnitude_warp(sigma: float = 0.2, knots: int = 4) -> TensorTransform:
    """
    Multiply the signal by a smooth curve sampled from Gaussian control points.

    Args:
        sigma: Standard deviation of the Gaussian control points. Larger values
               increase the amplitude variation.
        knots: Number of control points (>= 2).
    """

    if knots < 2:
        raise ValueError("`knots` must be at least 2.")
    if sigma < 0:
        raise ValueError("`sigma` must be non-negative.")

    def _inner(tensor: torch.Tensor) -> torch.Tensor:
        if sigma == 0:
            return tensor

        length = tensor.size(-1)
        rng = np.random.default_rng()
        knot_x = np.linspace(0, length - 1, knots)
        knot_y = rng.normal(loc=1.0, scale=sigma, size=knots).astype(np.float32)
        warp_curve = np.interp(np.arange(length), knot_x, knot_y).astype(np.float32)

        warp_tensor = torch.from_numpy(warp_curve).to(device=tensor.device, dtype=tensor.dtype)
        return tensor * warp_tensor.unsqueeze(0)

    return _inner


def random_sensor_dropout(drop_prob: float = 0.1, fill_value: float = 0.0) -> TensorTransform:
    """
    Randomly zero-out entire sensor channels to simulate dropout or occlusion.

    Args:
        drop_prob: Probability of dropping each channel. Zero disables dropout.
        fill_value: Replacement value for dropped channels (default zero).
    """

    if not 0 <= drop_prob <= 1:
        raise ValueError("`drop_prob` must be between 0 and 1.")

    def _inner(tensor: torch.Tensor) -> torch.Tensor:
        if drop_prob == 0:
            return tensor

        mask = torch.rand(tensor.size(0), device=tensor.device) < drop_prob
        if not torch.any(mask):
            return tensor

        output = tensor.clone()
        output[mask] = fill_value
        return output

    return _inner


def _resample_to_positions(tensor: torch.Tensor, positions: np.ndarray) -> torch.Tensor:
    """
    Helper: linearly resample `tensor` (channels, timesteps) at given positions.

    `positions`는 [0, length-1] 범위의 모노토닉 배열이어야 합니다.
    """

    length = tensor.size(-1)
    xp = np.arange(length, dtype=np.float32)
    clipped = np.clip(positions, 0.0, float(length - 1))

    array = tensor.detach().cpu().numpy()
    resampled = np.empty_like(array)
    for idx in range(array.shape[0]):
        resampled[idx] = np.interp(clipped, xp, array[idx])

    return torch.from_numpy(resampled).to(device=tensor.device, dtype=tensor.dtype)
