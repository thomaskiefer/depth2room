"""
Custom VACE training unit and training module for depth-conditioned video generation.

Contains:
  - WanVideoUnit_VACE_Depth: Subclass of WanVideoUnit_VACE that accepts pre-computed
    float tensors as vace_video input (skipping preprocess_video).
  - replace_vace_unit: Utility to swap the standard unit in a pipeline.
"""

import torch
import torch.nn.functional as F
from einops import rearrange

from diffsynth.pipelines.wan_video import WanVideoPipeline, WanVideoUnit_VACE
from depth2room.utils import validate_depth_tensor


class WanVideoUnit_VACE_Depth(WanVideoUnit_VACE):
    """
    Subclassed VACE unit that handles float tensor depth input + validity mask.

    When vace_video is already a float tensor in [-1, 1] range, it skips
    preprocess_video() and uses the tensor directly. An all-ones mask is
    applied as default (standard VACE inpainting behavior: treat the entire
    depth video as the "active" region).

    The 160-channel encoding:
      - inactive = vace_video * (1 - mask)  [all zeros since mask is all ones]
      - reactive = vace_video * mask         [the depth video itself]
      - Both are VAE-encoded → 16ch each = 32ch
      - Edit mask is patchified (8x8) → 64ch
      - Validity mask is patchified (8x8) → 64ch
      - vace_context = concat(video_latents, mask_latents, validity_latents) = 160ch
      - If reference image is provided, its latents are prepended temporally
    """

    def __init__(self):
        super().__init__()
        # Extend input_params to include vace_validity_mask so the
        # PipelineUnitRunner forwards it from inputs_shared to process().
        self.input_params = (
            "vace_video", "vace_video_mask", "vace_reference_image", "vace_scale",
            "height", "width", "num_frames", "tiled", "tile_size", "tile_stride",
            "vace_validity_mask",
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        vace_video, vace_video_mask, vace_reference_image, vace_scale,
        height, width, num_frames,
        tiled, tile_size, tile_stride,
        vace_validity_mask=None,
    ):
        if vace_video is not None or vace_video_mask is not None or vace_reference_image is not None:
            pipe.load_models_to_device(["vae"])

            if vace_video is None:
                vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=pipe.torch_dtype, device=pipe.device)
            elif isinstance(vace_video, torch.Tensor) and vace_video.ndim == 4:
                # Float tensor depth input: [C, T, H, W] -> [1, C, T, H, W]
                # Already in [-1, 1] range, skip preprocess_video()
                validate_depth_tensor(vace_video, label="vace_video input")
                vace_video = vace_video.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            else:
                # Standard PIL Image list path (fallback)
                vace_video = pipe.preprocess_video(vace_video)

            # All-ones mask: treat entire depth video as active region (standard VACE behavior)
            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            else:
                vace_video_mask = pipe.preprocess_video(vace_video_mask, min_value=0, max_value=1)

            # Assertion: verify mask is all ones when using depth tensors (no inpainting regions)
            assert vace_video_mask.shape == vace_video.shape, (
                f"Mask shape {vace_video_mask.shape} must match video shape {vace_video.shape}"
            )

            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = pipe.vae.encode(inactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            reactive = pipe.vae.encode(reactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            assert torch.isfinite(inactive).all(), "NaN/Inf in inactive latents after VAE encode"
            assert torch.isfinite(reactive).all(), "NaN/Inf in reactive latents after VAE encode"
            vace_video_latents = torch.concat((inactive, reactive), dim=1)

            mask_h, mask_w = vace_video_mask.shape[3], vace_video_mask.shape[4]
            assert mask_h % 8 == 0 and mask_w % 8 == 0, (
                f"Mask spatial dims must be divisible by 8 for patchification, "
                f"got {mask_h}x{mask_w}"
            )
            vace_mask_latents = rearrange(vace_video_mask[0, 0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(
                vace_mask_latents,
                size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]),
                mode='nearest-exact',
            )

            # Patchify validity mask (same spatial decomposition as edit mask)
            if vace_validity_mask is not None and isinstance(vace_validity_mask, torch.Tensor) and vace_validity_mask.ndim == 4:
                vm = vace_validity_mask.to(dtype=pipe.torch_dtype, device=pipe.device)
                # [1, T, H, W] -> patchify 8x8 -> [1, 64, T, H/8, W/8]
                validity_latents = rearrange(vm[0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
                validity_latents = F.interpolate(
                    validity_latents,
                    size=vace_mask_latents.shape[2:],
                    mode='nearest-exact',
                )
            else:
                # No validity mask: default to all-valid (ones).
                # Before training this doesn't matter (zero-init weights × anything = 0).
                # After training, ones = "all pixels valid" which is the safe default.
                validity_latents = torch.ones_like(vace_mask_latents)

            if vace_reference_image is None:
                pass
            else:
                if not isinstance(vace_reference_image, list):
                    vace_reference_image = [vace_reference_image]

                vace_reference_image = pipe.preprocess_video(vace_reference_image)

                bs, c, f, h, w = vace_reference_image.shape
                new_vace_ref_images = []
                for j in range(f):
                    new_vace_ref_images.append(vace_reference_image[0, :, j:j + 1])
                vace_reference_image = new_vace_ref_images

                vace_reference_latents = pipe.vae.encode(
                    vace_reference_image, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
                ).to(dtype=pipe.torch_dtype, device=pipe.device)
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
                vace_reference_latents = [u.unsqueeze(0) for u in vace_reference_latents]

                vace_video_latents = torch.concat((*vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :f]), vace_mask_latents), dim=2)
                # Reference frames get zero validity (no depth info for reference image)
                validity_latents = torch.concat((torch.zeros_like(validity_latents[:, :, :f]), validity_latents), dim=2)

            vace_context = torch.concat((vace_video_latents, vace_mask_latents, validity_latents), dim=1)

            # Assertion: VACE context must have 160 channels
            # inactive(16ch) + reactive(16ch) + edit_mask(64ch) + validity_mask(64ch) = 160
            assert vace_context.shape[1] == 160, (
                f"VACE context must have 160 channels, got {vace_context.shape[1]}. "
                f"Shape: {vace_context.shape}"
            )

            return {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return {"vace_context": None, "vace_scale": vace_scale}


def patch_pipeline_for_validity_mask(pipe: WanVideoPipeline):
    """Patch the pipeline's __call__ to accept and forward vace_validity_mask.

    WanVideoPipeline.__call__() has a fixed signature that doesn't include
    vace_validity_mask. We create a dynamic subclass and reassign pipe.__class__
    so that pipe(...) resolves to our patched __call__ via normal class MRO.

    Note: instance-level __call__ assignment (types.MethodType) does NOT work
    because Python's implicit special-method lookup for pipe(...) uses
    type(pipe).__call__, bypassing instance __dict__.
    """
    original_cls = pipe.__class__
    original_call = original_cls.__call__

    class _PatchedPipeline(original_cls):
        def __call__(self, *args, vace_validity_mask=None, **kwargs):
            if vace_validity_mask is not None:
                real_runner = self.unit_runner

                class _PatchedRunner:
                    def __init__(self, real, mask):
                        self.real = real
                        self.mask = mask

                    def __call__(self, unit, pipe_arg, inputs_shared, *rest):
                        if isinstance(unit, WanVideoUnit_VACE_Depth):
                            inputs_shared["vace_validity_mask"] = self.mask
                        return self.real(unit, pipe_arg, inputs_shared, *rest)

                self.unit_runner = _PatchedRunner(real_runner, vace_validity_mask)
                try:
                    result = original_call(self, *args, **kwargs)
                finally:
                    self.unit_runner = real_runner
                return result
            else:
                return original_call(self, *args, **kwargs)

    pipe.__class__ = _PatchedPipeline
    return pipe


def replace_vace_unit(pipe: WanVideoPipeline):
    """
    Replace the standard WanVideoUnit_VACE in the pipeline's unit list
    with our depth-aware WanVideoUnit_VACE_Depth.
    """
    new_units = []
    replaced = False
    for unit in pipe.units:
        if isinstance(unit, WanVideoUnit_VACE):
            new_units.append(WanVideoUnit_VACE_Depth())
            replaced = True
        else:
            new_units.append(unit)
    assert replaced, "No WanVideoUnit_VACE found in pipeline units to replace!"
    pipe.units = new_units
    return pipe
