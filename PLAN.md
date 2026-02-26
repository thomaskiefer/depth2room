# CAD-Estate VACE Training Fix Plan

## Summary

Fix the depth-conditioned VACE training pipeline for CAD-Estate to resolve
incorrect training results. Three independent issues identified, plus a
captioning improvement.

---

## 1. Fix depth rendering clear_color bug (CRITICAL)

**Problem**: `render_scene()` uses `clear_color=(0,0,0,1)` by default. Background
pixels (no geometry) get alpha=1.0, which is read as depth=1.0m. The validity
check `(depth > 0) & (depth < z_far)` passes these as valid. 94.9% of frames
have d_max=exactly 1.0 from this bug — false "close surfaces" everywhere there's
empty space, and real geometry gets compressed into half the disparity range.

**Fix**: Pass `clear_color=(0,0,0,0)` in `prepare_cad_estate_vace_data.py:366`.

**Status**: Already applied.

---

## 2. Add validity mask via Context Embedder expansion

**Problem**: Invalid depth pixels (holes where no room structure exists) are
encoded as 0.0 in [-1,1], which is ambiguous with valid mid-range disparity.
The model can't distinguish "no geometry, hallucinate freely" from "real
structure at medium distance."

**Approach**: Add a separate binary validity mask that bypasses the VAE and
enters through the Context Embedder (same path as the existing edit mask M).
This follows the VACE paper's architecture pattern exactly (Section 3.3.1).
Full fine-tuning of VACE weights ensures the model can learn the new signal.

**Changes required**:

### 2a. Data prep (`prepare_cad_estate_vace_data.py`)
- Save `validity_mask.pt` per clip: shape `[1, T, H, W]`, float32, values 0/1
- Derived from existing `valid = (depth > 0) & (depth < z_far)` (line 183)
- This mask is already computed but currently discarded

### 2b. Dataset (`vace_depth_dataset.py`)
- Add `LoadValidityMask` operator
- Return as `data["vace_validity_mask"]`

### 2c. VACE unit (`vace_depth_training_unit.py`)
- Accept validity mask as additional input
- Patchify identically to edit mask:
  `rearrange(mask, "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)` + interpolate
- Concatenate to vace_context: 32 (video latents) + 64 (edit mask) + 64 (validity mask) = 160 channels
- Update assertion from 96 → 160

### 2d. Training script (`train_vace_depth.py`)
- Route `vace_validity_mask` through `parse_extra_inputs`
- Weight surgery after model load:
  ```python
  old_conv = pipe.vace.vace_patch_embedding  # Conv3d(96, dim, (1,2,2))
  new_conv = Conv3d(160, dim, kernel_size=(1,2,2), stride=(1,2,2))
  new_conv.weight.data.zero_()
  new_conv.weight.data[:, :96] = old_conv.weight.data
  new_conv.bias.data = old_conv.bias.data.clone()
  pipe.vace.vace_patch_embedding = new_conv
  ```

### 2e. Shell script (`train_vace_full.sh`)
- Add `vace_validity_mask` to `--extra_inputs`

---

## 3. Fix depth normalization consistency

**Problem**: Per-frame min-max normalization (line 189-196) means the same
physical depth maps to different values across frames. Inference code
(`infer_from_exr.py`) defaults to global normalization — train/infer mismatch.

**Fix**: Switch to per-clip global normalization in `prepare_cad_estate_vace_data.py`:
- Compute d_min/d_max across ALL valid pixels in ALL frames of the clip
- Apply the same normalization to every frame
- Update `infer_from_exr.py` to use matching per-clip normalization
- Save the global d_min/d_max in depth_meta.json for inference use

---

## 4. Regenerate captions

**Problem**: Current pipeline uses Qwen2.5-VL (20-40 word caption) → Qwen2.5
LM extension with WAN system prompt. The WAN prompt examples include "CG game
concept digital art", "Japanese-style film photography", "Anime thick-coated
illustration" — biasing captions toward stylized/artistic language instead of
photorealistic real-estate descriptions.

**New approach** (VideoX-Fun style, two-stage):

### Stage 1: Video captioning with InternVL3.5-38B
- Model: `OpenGVLab/InternVL3_5-38B`
- Sample 8 frames uniformly from each RGB video
- Feed as multi-image with frame labels
- Prompt: `"Describe this video in detail. Don't repeat."`
- Params: temperature=0, max_tokens=512

### Stage 2: Caption rewriting with GPT-OSS-120B
- Model: `openai/gpt-oss-120b`
- Adapt VideoX-Fun's rewrite.txt prompt:
  - Strip "The video shows..." prefixes
  - Remove subjective content
  - Remove non-visual descriptions
  - Keep camera movement/style if present in original
  - Add: "This is real photorealistic interior footage" style anchor
- Params: temperature=0.7, max_tokens=1024
- Output: JSON `{"rewritten description": "..."}`

### Infrastructure
- InternVL3.5-38B: ~2x H100 via vLLM
- GPT-OSS-120B: 1x H100 via vLLM (MoE, 5B active params)

---

## Execution order

1. Fix clear_color (done) + normalization → regenerate depth dataset
2. Save validity masks during regeneration (same pass)
3. Implement Context Embedder expansion (2b-2e)
4. Regenerate captions (can run in parallel with above)
5. Train with all fixes applied
6. Evaluate structure adherence
7. Only if needed: add loss weighting on structural pixels

---

## What we decided NOT to do

- **Sentinel value encoding** (-1.0 for invalid): Works but less clean than
  a separate mask channel. Considered but rejected in favor of the architectural
  approach since we're doing full fine-tuning anyway.
- **Fill holes with monocular depth**: Wrong — holes ARE the signal. The task is
  "given room structure, hallucinate furnishing." Filling defeats the purpose.
- **Repurpose vace_video_mask as validity mask**: Semantic mismatch with VACE
  training (mask means edit/keep, not valid/invalid).
- **Loss weighting**: Premature. Try architectural fixes first, evaluate, then
  add if model still ignores structure in valid regions.
