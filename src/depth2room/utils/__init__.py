"""Shared utilities for depth2room."""

import torch


def validate_depth_tensor(
    tensor: torch.Tensor,
    *,
    expected_frames: int | None = None,
    expected_height: int | None = None,
    expected_width: int | None = None,
    label: str = "depth tensor",
) -> None:
    """Validate a depth tensor meets the expected format.

    Checks:
      - 4D shape [3, T, H, W]
      - float dtype
      - values in [-1, 1] (with small epsilon)
      - no NaN/Inf values
      - optional frame count and spatial dimension checks

    Args:
        tensor: The depth tensor to validate.
        expected_frames: If set, assert T == expected_frames.
        expected_height: If set, assert H == expected_height.
        expected_width: If set, assert W == expected_width.
        label: Human-readable label for error messages.

    Raises:
        AssertionError: If any check fails.
    """
    assert isinstance(tensor, torch.Tensor), (
        f"{label}: expected torch.Tensor, got {type(tensor)}"
    )
    assert tensor.ndim == 4, (
        f"{label}: expected 4D [3, T, H, W], got shape {tensor.shape}"
    )
    assert tensor.shape[0] == 3, (
        f"{label}: channel dim must be 3, got {tensor.shape[0]}"
    )
    assert tensor.dtype in (torch.float32, torch.float16, torch.bfloat16), (
        f"{label}: expected float dtype, got {tensor.dtype}"
    )
    eps = 1e-3
    assert tensor.min() >= -1.0 - eps and tensor.max() <= 1.0 + eps, (
        f"{label}: values must be in [-1, 1], "
        f"got min={tensor.min():.4f} max={tensor.max():.4f}"
    )
    assert torch.isfinite(tensor).all(), (
        f"{label}: contains NaN/Inf values"
    )
    if expected_frames is not None:
        assert tensor.shape[1] == expected_frames, (
            f"{label}: expected {expected_frames} frames, got {tensor.shape[1]}"
        )
    if expected_height is not None:
        assert tensor.shape[2] == expected_height, (
            f"{label}: expected height {expected_height}, got {tensor.shape[2]}"
        )
    if expected_width is not None:
        assert tensor.shape[3] == expected_width, (
            f"{label}: expected width {expected_width}, got {tensor.shape[3]}"
        )
