"""Shared utilities for depth2room."""

import numpy as np
import torch


def center_crop_resize(img: torch.Tensor, target_h: int,
                       target_w: int) -> torch.Tensor:
    """Center-crop and resize a [C, H, W] or [H, W] tensor to target size.

    Scales up (preserving aspect ratio) so the smaller dimension matches target,
    then center-crops to exactly (target_h, target_w). This avoids aspect-ratio
    warping that direct resize would cause.
    """
    if img.ndim == 2:
        h, w = img.shape
        unsqueeze = True
        img = img.unsqueeze(0)
    else:
        assert img.ndim == 3
        _, h, w = img.shape
        unsqueeze = False

    scale = max(target_h / h, target_w / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    assert new_h >= target_h and new_w >= target_w

    img_float = img.unsqueeze(0).float()
    img_resized = torch.nn.functional.interpolate(
        img_float, size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    img_resized = img_resized.squeeze(0)

    top = (new_h - target_h) // 2
    left = (new_w - target_w) // 2
    img_cropped = img_resized[:, top:top + target_h, left:left + target_w]

    assert img_cropped.shape[-2:] == (target_h, target_w)

    if unsqueeze:
        img_cropped = img_cropped.squeeze(0)

    return img_cropped


def normalize_depth_clip(
    raw_depth_frames: list[torch.Tensor],
    z_far: float,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
    eps: float = 1e-8,
) -> tuple[list[torch.Tensor], float, float]:
    """Convert raw linear depth frames to normalized disparity in [-1, 1].

    Uses per-clip global normalization with percentile clipping: computes the
    disparity range across ALL frames in the clip, clips to percentile bounds,
    then linearly scales to [-1, 1]. This preserves cross-frame depth
    consistency (following LongVie/LongVie2 convention).

    Args:
        raw_depth_frames: List of [H, W] float32 linear depth tensors.
        z_far: Far plane value for validity thresholding.
        percentile_low: Lower percentile for outlier clipping (default 1.0).
        percentile_high: Upper percentile for outlier clipping (default 99.0).
        eps: Epsilon for numerical stability.

    Returns:
        Tuple of:
          - List of [3, H, W] float32 tensors in [-1, 1], 3-channel grayscale
          - d_min: float, global disparity lower bound (percentile-clipped)
          - d_max: float, global disparity upper bound (percentile-clipped)
    """
    # Step 1: Convert all frames to disparity, collect valid values
    disparities = []
    valid_masks = []
    all_valid_disp = []
    for depth in raw_depth_frames:
        valid = (depth > 0) & (depth < z_far)
        assert valid.any(), "No valid depth pixels in frame"
        disp = torch.zeros_like(depth)
        disp[valid] = 1.0 / depth[valid]
        disparities.append(disp)
        valid_masks.append(valid)
        all_valid_disp.append(disp[valid])

    # Step 2: Compute global percentile range across all frames
    # Use numpy.percentile instead of torch.quantile — the latter has an
    # internal size limit (~2^24 elements) that clips of 81 frames exceed.
    all_valid_np = torch.cat(all_valid_disp).cpu().numpy()
    d_min = float(np.percentile(all_valid_np, percentile_low))
    d_max = float(np.percentile(all_valid_np, percentile_high))
    assert d_max > d_min + eps, (
        f"Degenerate disparity range: d_min={d_min}, d_max={d_max}"
    )

    # Step 3: Normalize each frame with the global range
    normalized_frames = []
    for disp, valid in zip(disparities, valid_masks):
        disp_clipped = torch.clamp(disp, d_min, d_max)
        disp_01 = (disp_clipped - d_min) / (d_max - d_min + eps)
        disp_01[~valid] = 0.0
        disp_normalized = 2.0 * disp_01 - 1.0
        disp_normalized[~valid] = 0.0

        depth_rgb = disp_normalized.unsqueeze(0).expand(3, -1, -1).contiguous()
        normalized_frames.append(depth_rgb)

    return normalized_frames, d_min, d_max


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
