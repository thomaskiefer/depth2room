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

import os
import torch
from diffsynth.core.data.unified_dataset import UnifiedDataset
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

        # Assertions: validate the depth tensor
        assert isinstance(tensor, torch.Tensor), (
            f"Depth file must contain a torch.Tensor, got {type(tensor)}"
        )
        assert tensor.ndim == 4, (
            f"Depth tensor must be 4D [C, T, H, W], got shape {tensor.shape}"
        )
        assert tensor.shape[0] == 3, (
            f"Depth tensor channel dim must be 3, got {tensor.shape[0]}"
        )
        assert tensor.dtype == torch.float32, (
            f"Depth tensor must be float32, got {tensor.dtype}"
        )
        assert tensor.min() >= -1.0 - 1e-3 and tensor.max() <= 1.0 + 1e-3, (
            f"Depth tensor values must be in [-1, 1], got min={tensor.min():.4f} max={tensor.max():.4f}"
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
    ):
        self.base_path = base_path
        self.depth_loader = LoadDepthTensor(base_path=base_path)

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
        import math
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
            data["vace_video_tensor"] = depth_tensor
        else:
            data["vace_video_tensor"] = None

        # Handle empty vace_reference_image (CSV has empty string for no-ref rows)
        if "vace_reference_image" not in data or data.get("vace_reference_image") in ("", None):
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
