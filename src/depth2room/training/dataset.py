"""
Custom dataset for VACE depth-conditioned training.

Extends UnifiedDataset to handle pre-computed depth tensors stored as .pt files.
Standard columns (video, vace_reference_image, prompt) are loaded via default operators,
while vace_video (depth) is loaded as a raw float32 tensor [3, num_frames, H, W]
already normalized to [-1, 1].

The returned dict contains:
  - "prompt": str
  - "video": list[PIL.Image]  (target RGB video frames)
  - "vace_video_tensor": torch.Tensor  (depth in [-1,1], shape [3,T,H,W])
  - "vace_reference_image": list[PIL.Image] or None
"""

import math
import os
import random

import torch
from diffsynth.core.data.unified_dataset import UnifiedDataset
from depth2room.utils import validate_depth_tensor
from diffsynth.core.data.operators import (
    ToAbsolutePath,
    LoadImage,
    ImageCropAndResize,
    LoadVideo,
    DataProcessingOperator,
    RouteByType,
    RouteByExtensionName,
    ToList,
)


class LoadDepthTensor(DataProcessingOperator):
    """Load a .pt depth tensor file and validate its contents."""

    def __init__(self, base_path=""):
        self.base_path = base_path

    def __call__(self, data):
        path = data
        if not os.path.isabs(path):
            path = os.path.join(self.base_path, path)

        tensor = torch.load(path, map_location="cpu", weights_only=True)
        validate_depth_tensor(tensor, label=f"depth file {path}")
        return tensor


class LoadValidityMask(DataProcessingOperator):
    """Load a .pt validity mask file [1, T, H, W] with values 0.0/1.0."""

    def __init__(self, base_path=""):
        self.base_path = base_path

    def __call__(self, data):
        path = data
        if not os.path.isabs(path):
            path = os.path.join(self.base_path, path)

        tensor = torch.load(path, map_location="cpu", weights_only=True)
        assert tensor.ndim == 4 and tensor.shape[0] == 1, (
            f"Validity mask must be [1, T, H, W], got {tensor.shape}"
        )
        return tensor


class VACEDepthDataset(torch.utils.data.Dataset):
    """
    Dataset for VACE depth-conditioned video generation training.

    Wraps a UnifiedDataset for standard fields (video, vace_reference_image, prompt)
    and adds custom loading for depth tensor files via the 'vace_video' column.

    The depth tensor is returned under the key 'vace_video_tensor' to signal
    the custom training unit that it should skip preprocess_video().

    Args:
        base_path: Root directory containing all data files.
        metadata_path: Path to the metadata CSV.
        num_frames: Number of video frames to load.
        height: Target frame height (or None for dynamic).
        width: Target frame width (or None for dynamic).
        max_pixels: Max pixels per frame for dynamic resolution.
        repeat: Number of times to repeat the dataset per epoch.
    """

    def __init__(
        self,
        base_path,
        metadata_path,
        num_frames=81,
        height=480,
        width=832,
        max_pixels=1920 * 1080,
        repeat=1,
        ref_drop_prob=0.5,
    ):
        self.base_path = base_path
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.ref_drop_prob = ref_drop_prob
        self.depth_loader = LoadDepthTensor(base_path=base_path)
        self.validity_loader = LoadValidityMask(base_path=base_path)

        # Build the underlying UnifiedDataset for standard columns.
        # 'vace_video' is handled via special_operator_map to keep it as raw path,
        # then we intercept it in __getitem__.
        # 'vace_reference_image' loads as a list of PIL images via standard image ops.
        self.unified_dataset = UnifiedDataset(
            base_path=base_path,
            metadata_path=metadata_path,
            repeat=repeat,
            data_file_keys=["video", "vace_reference_image"],
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=base_path,
                max_pixels=max_pixels,
                height=height,
                width=width,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=num_frames,
                time_division_factor=4,
                time_division_remainder=1,
            ),
            special_operator_map={
                "vace_reference_image": (
                    ToAbsolutePath(base_path)
                    >> RouteByExtensionName(operator_map=[
                        (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, 16, 16) >> ToList()),
                    ])
                ),
            },
        )

        # Required by runner.py (launch_training_task checks dataset.load_from_cache)
        self.load_from_cache = False

        # Store metadata for direct depth loading
        # Fix NaN/empty values from pandas CSV reader (empty cells become float NaN).
        # Remove keys with empty/NaN values so operators are not called on them.
        for row in self.unified_dataset.data:
            keys_to_remove = []
            for key, val in row.items():
                if isinstance(val, float) and math.isnan(val):
                    keys_to_remove.append(key)
                elif isinstance(val, str) and val.strip() == "":
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del row[key]
        self.data = self.unified_dataset.data

    def save_debug_sample(self, idx, output_dir):
        """Save a single sample's components as image files for visual inspection.

        Saves: first/last RGB frames, first/last depth frames, reference image.
        Useful for verifying data loading before training.
        """
        from PIL import Image as PILImage

        os.makedirs(output_dir, exist_ok=True)
        data = self[idx]

        # Save first and last RGB frames
        for i, label in [(0, "first"), (len(data["video"]) - 1, "last")]:
            data["video"][i].save(os.path.join(output_dir, f"rgb_{label}.jpg"))

        # Save depth frames as grayscale images
        depth = data.get("vace_video_tensor")
        if depth is not None:
            for i, label in [(0, "first"), (depth.shape[1] - 1, "last")]:
                frame = ((depth[0, i] + 1) / 2 * 255).clamp(0, 255).byte().numpy()
                PILImage.fromarray(frame, mode="L").save(
                    os.path.join(output_dir, f"depth_{label}.jpg")
                )

        # Save reference image
        ref = data.get("vace_reference_image")
        if ref is not None and isinstance(ref, list) and len(ref) > 0:
            ref[0].save(os.path.join(output_dir, "reference.jpg"))

        # Save metadata
        import json
        meta = {
            "idx": idx,
            "prompt": data.get("prompt", ""),
            "num_video_frames": len(data["video"]),
            "depth_shape": list(depth.shape) if depth is not None else None,
            "has_reference": ref is not None,
        }
        with open(os.path.join(output_dir, "debug_info.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def __len__(self):
        return len(self.unified_dataset)

    def __getitem__(self, idx):
        # Get the standard data from UnifiedDataset
        data = self.unified_dataset[idx]

        # Load depth tensor from the vace_video column (it's a .pt path in the CSV)
        raw_row = self.data[idx % len(self.data)]
        vace_video_path = raw_row.get("vace_video", "")

        if vace_video_path:
            depth_tensor = self.depth_loader(vace_video_path)
            # Validate depth tensor dimensions match video frames
            num_video_frames = len(data["video"])
            assert depth_tensor.shape[1] == num_video_frames, (
                f"Depth tensor has {depth_tensor.shape[1]} frames but video has "
                f"{num_video_frames} frames for {vace_video_path}"
            )
            frame_h, frame_w = data["video"][0].size[1], data["video"][0].size[0]
            assert depth_tensor.shape[2] == frame_h and depth_tensor.shape[3] == frame_w, (
                f"Depth tensor spatial size {depth_tensor.shape[2]}x{depth_tensor.shape[3]} "
                f"doesn't match video frame size {frame_h}x{frame_w} for {vace_video_path}"
            )
            data["vace_video_tensor"] = depth_tensor
        else:
            data["vace_video_tensor"] = None

        # Load validity mask from the vace_validity_mask column
        validity_path = raw_row.get("vace_validity_mask", "")
        if validity_path:
            validity_mask = self.validity_loader(validity_path)
            if data["vace_video_tensor"] is not None:
                assert validity_mask.shape[1] == data["vace_video_tensor"].shape[1], (
                    f"Validity mask has {validity_mask.shape[1]} frames but depth has "
                    f"{data['vace_video_tensor'].shape[1]} frames"
                )
            data["vace_validity_mask"] = validity_mask
        else:
            data["vace_validity_mask"] = None

        # Handle empty vace_reference_image (CSV has empty string for no-ref rows)
        if "vace_reference_image" not in data or data.get("vace_reference_image") in ("", None):
            data["vace_reference_image"] = None
        # Dynamic reference image dropout (re-randomized per sample per epoch)
        elif self.ref_drop_prob > 0 and random.random() < self.ref_drop_prob:
            data["vace_reference_image"] = None

        # Remove the raw vace_video string path if still present
        if "vace_video" in data and isinstance(data["vace_video"], str):
            del data["vace_video"]

        # Assertions on returned data
        assert "prompt" in data, "Dataset item must contain 'prompt'"
        assert "video" in data, "Dataset item must contain 'video'"
        assert isinstance(data["video"], list), "video must be a list of PIL Images"
        assert len(data["video"]) > 0, "video must have at least one frame"
        if data["vace_video_tensor"] is not None:
            assert isinstance(data["vace_video_tensor"], torch.Tensor), (
                "vace_video_tensor must be a torch.Tensor"
            )

        return data
