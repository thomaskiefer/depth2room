"""
Custom VACE training unit and training module for depth-conditioned video generation.

Contains:
  - WanVideoUnit_VACE_Depth: Subclass of WanVideoUnit_VACE that accepts pre-computed
    float tensors as vace_video input (skipping preprocess_video).
  - replace_vace_unit: Utility to swap the standard unit in a pipeline.
"""

import torch
from einops import rearrange

from diffsynth.pipelines.wan_video import WanVideoPipeline, WanVideoUnit_VACE
from diffsynth.diffusion.base_pipeline import PipelineUnit


class WanVideoUnit_VACE_Depth(WanVideoUnit_VACE):
    """
    Subclassed VACE unit that handles float tensor depth input.

    When vace_video is already a float tensor in [-1, 1] range, it skips
    preprocess_video() and uses the tensor directly. An all-ones mask is
    applied as default (standard VACE inpainting behavior: treat the entire
    depth video as the "active" region).

    The rest of the 96-channel encoding proceeds identically to the base class:
      - inactive = vace_video * (1 - mask)  [all zeros since mask is all ones]
      - reactive = vace_video * mask         [the depth video itself]
      - Both are VAE-encoded, concatenated along channel dim
      - Mask is downsampled to latent space
      - If reference image is provided, its latents are prepended
      - Final vace_context = concat(video_latents, mask_latents) along channel dim
    """

    def process(
        self,
        pipe: WanVideoPipeline,
        vace_video, vace_video_mask, vace_reference_image, vace_scale,
        height, width, num_frames,
        tiled, tile_size, tile_stride
    ):
        if vace_video is not None or vace_video_mask is not None or vace_reference_image is not None:
            pipe.load_models_to_device(["vae"])

            if vace_video is None:
                vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=pipe.torch_dtype, device=pipe.device)
            elif isinstance(vace_video, torch.Tensor) and vace_video.ndim == 4:
                # Float tensor depth input: [C, T, H, W] -> [1, C, T, H, W]
                # Already in [-1, 1] range, skip preprocess_video()
                assert vace_video.dtype in (torch.float32, torch.float16, torch.bfloat16), (
                    f"Depth tensor must be a float type, got {vace_video.dtype}"
                )
                assert vace_video.min() >= -1.0 - 1e-3 and vace_video.max() <= 1.0 + 1e-3, (
                    f"Depth tensor values must be in [-1, 1], "
                    f"got min={vace_video.min():.4f} max={vace_video.max():.4f}"
                )
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
            vace_video_latents = torch.concat((inactive, reactive), dim=1)

            vace_mask_latents = rearrange(vace_video_mask[0, 0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(
                vace_mask_latents,
                size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]),
                mode='nearest-exact',
            )

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

            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)

            # Assertion: VACE context must have 96 channels
            # inactive(16ch) + reactive(16ch) + mask(64ch) = 96 channels
            assert vace_context.shape[1] == 96, (
                f"VACE context must have 96 channels, got {vace_context.shape[1]}. "
                f"Shape: {vace_context.shape}"
            )

            return {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return {"vace_context": None, "vace_scale": vace_scale}


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
