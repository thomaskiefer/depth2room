#!/usr/bin/env python3
"""Generate captions for CAD-Estate VACE training scenes.

Two-phase pipeline following the EasyAnimate/VideoX-Fun recipe:

  Phase 1 (VLM): CapRL-Qwen3VL-4B captions video frames → vlm_captions.json
  Phase 2 (LLM): gpt-oss-120b rewrites captions in EasyAnimate style → captions.json

Both models are loaded sequentially (each needs most of the GPU memory).

Fallback mode (--fallback_only): Generates template captions from room structure
labels in the annotation .npz files, without any model inference.

Output: a captions.json mapping scene_id -> caption string.
"""

import argparse
import gc
import json
import os
import re
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Structural element types from cad_estate.room_structure
STRUCTURAL_ELEMENT_TYPES = ["<ignore>", "wall", "floor", "ceiling", "slanted"]

# Number of frames to sample from each video (EasyAnimate default)
NUM_SAMPLED_FRAMES = 8

# --- VLM captioning prompts (CapRL recipe) ---

VLM_SYSTEM_PROMPT = "You are a helpful assistant."
VLM_CAPTION_PROMPT_TEXT = "Describe this video in detail. Don't repeat."

# --- LLM rewrite prompt (adapted from EasyAnimate rewrite.txt) ---

REWRITE_SYSTEM_PROMPT = "You are a helpful assistant."

REWRITE_USER_PROMPT = """\
Please rewrite the video description to be useful for AI to re-generate the video, according to the following requirements
1. Do not start with something similar to 'The video/scene/frame shows' or "In this video/scene/frame".
2. Remove the subjective content deviates from describing the visual content of the video. For instance, a sentence like "It gives a feeling of ease and tranquility and makes people feel comfortable" is considered subjective.
3. Remove the non-existent description that does not in the visual content of the video, For instance, a sentence like "There is no visible detail that could be used to identify the individual beyond what is shown." is considered as the non-existent description.
4. The rewritten description should include the main subject (person, object, animal, or none) actions and their attributes or status sequence, the background (the objects, location, weather, and time).
5. If the original description includes the view shot, camera movement and the video style, the rewritten description should also include them. If not, there is no need to invent them on your own.
6. Here are some examples of good descriptions: 1) A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about. 2) A large orange octopus is seen resting on the bottom of the ocean floor, blending in with the sandy and rocky terrain. Its tentacles are spread out around its body, and its eyes are closed. The octopus is unaware of a king crab that is crawling towards it from behind a rock, its claws raised and ready to attack. The crab is brown and spiny, with long legs and antennae. The scene is captured from a wide angle, showing the vastness and depth of the ocean. The water is clear and blue, with rays of sunlight filtering through. The shot is sharp and crisp, with a high dynamic range. The octopus and the crab are in focus, while the background is slightly blurred, creating a depth of field effect.
7. Output with the following json format:
{"rewritten description": "your rewritten description here"}

Here is the video description:
"""


# --- Video frame extraction (following EasyAnimate recipe) ---

@contextmanager
def _video_reader(path: str):
    """Context manager for decord VideoReader to avoid memory leaks."""
    from decord import VideoReader
    vr = VideoReader(path, num_threads=2)
    try:
        yield vr
    finally:
        del vr
        gc.collect()


def extract_frames(video_path: str, num_frames: int = NUM_SAMPLED_FRAMES) -> list[Image.Image]:
    """Uniformly sample frames from a video (EasyAnimate recipe).

    Uses np.linspace(0, total, num_frames, endpoint=False) — same formula
    as EasyAnimate/VideoX-Fun's extract_frames() with method="uniform".
    """
    with _video_reader(video_path) as vr:
        indices = np.linspace(0, len(vr), num_frames, endpoint=False, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(f) for f in frames]


# --- Shared helpers ---

def find_scene_videos(dataset_dir: str) -> dict[str, str]:
    """Find all <scene_id>_rgb.mp4 files in the dataset directory.

    Returns a dict mapping scene_id -> path to the first video found.
    Each scene may have multiple videos; we pick the first one sorted
    alphabetically (consistent with how ref images were chosen).
    """
    videos: dict[str, str] = {}
    dataset_path = Path(dataset_dir)
    for mp4 in sorted(dataset_path.glob("*/*_rgb.mp4")):
        scene_id = mp4.stem.removesuffix("_rgb")
        if scene_id not in videos:
            videos[scene_id] = str(mp4)
    for mp4 in sorted(dataset_path.glob("*_rgb.mp4")):
        scene_id = mp4.stem.removesuffix("_rgb")
        if scene_id not in videos:
            videos[scene_id] = str(mp4)
    return videos


def build_fallback_caption(scene_id: str, annotations_dir: str) -> str:
    """Build a template caption from room structure labels."""
    npz_path = os.path.join(annotations_dir, scene_id, "room_structure.npz")
    if not os.path.exists(npz_path):
        return (
            "An indoor scene showing a room interior with walls, floor, and ceiling. "
            "The camera slowly pans across the space, revealing architectural details "
            "and spatial layout. Natural lighting illuminates the room."
        )

    data = np.load(npz_path, allow_pickle=True)
    labels = data["layout_labels"]

    counts = Counter()
    for label_idx in labels:
        label_idx = int(label_idx)
        if 0 <= label_idx < len(STRUCTURAL_ELEMENT_TYPES):
            label_name = STRUCTURAL_ELEMENT_TYPES[label_idx]
            if label_name != "<ignore>":
                counts[label_name] += 1

    if not counts:
        return (
            "An indoor scene showing a room interior. The camera slowly pans across "
            "the space, revealing the room layout and architectural features. "
            "Soft natural lighting fills the environment."
        )

    parts = []
    elem_plurals = {
        "wall": "walls", "floor": "floors",
        "ceiling": "ceilings", "slanted": "slanted surfaces",
    }
    for elem_type in ["wall", "floor", "ceiling", "slanted"]:
        count = counts.get(elem_type, 0)
        if count > 0:
            if count == 1:
                parts.append(
                    f"{count} {elem_type}" if elem_type != "slanted"
                    else f"{count} slanted surface"
                )
            else:
                parts.append(f"{count} {elem_plurals[elem_type]}")

    if len(parts) == 1:
        elements_str = parts[0]
    elif len(parts) == 2:
        elements_str = f"{parts[0]} and {parts[1]}"
    else:
        elements_str = ", ".join(parts[:-1]) + f", and {parts[-1]}"

    return (
        f"A real-estate interior walkthrough video of a room with {elements_str}. "
        f"The camera slowly moves through the space, capturing the architectural "
        f"structure and spatial layout. Natural lighting illuminates the room surfaces, "
        f"revealing textures and depth. The scene is captured in real-life footage."
    )


def generate_fallback_captions(
    ref_images: dict[str, str],
    annotations_dir: str,
) -> dict[str, str]:
    """Generate template captions from room structure labels only."""
    captions = {}
    scene_ids = sorted(ref_images.keys())
    total = len(scene_ids)
    print(f"Generating fallback captions for {total} scenes...")

    for i, scene_id in enumerate(scene_ids):
        captions[scene_id] = build_fallback_caption(scene_id, annotations_dir)
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] processed")

    return captions


# --- Phase 1: VLM captioning with CapRL-Qwen3VL-4B ---

def load_vlm_model(model_path: str | None):
    """Load CapRL-Qwen3VL-4B for video/image captioning.

    4B params (~8GB bf16), fits easily on a single GPU.
    RL-trained specifically for dense image captioning (ICLR 2026).
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if model_path is None:
        model_path = "internlm/CapRL-Qwen3VL-4B"

    print(f"Loading VLM model from: {model_path}")

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    # Limit visual tokens per frame to ~256 (equivalent to EasyAnimate's
    # max_dynamic_patch=1).  Qwen3-VL uses patch_size=16, merge_size=2,
    # so each visual token = 32×32 = 1024 pixels.  256 tokens = 262144 px.
    # Without this the processor uses full resolution (~540 tokens/frame).
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=256 * 32 * 32,
        max_pixels=256 * 32 * 32,
    )

    print("VLM model loaded.")
    return model, processor


def caption_video_vlm(video_path: str, model, processor) -> str:
    """Caption a scene by sampling 8 frames from its video.

    Follows the EasyAnimate/VideoX-Fun multi-frame captioning approach:
    uniformly sample NUM_SAMPLED_FRAMES frames, pass each as a separate
    image in the conversation, then generate a description.
    """
    frames = extract_frames(video_path, NUM_SAMPLED_FRAMES)

    # Build multi-image message (Qwen3-VL format).
    # Don't label frames individually — we want a coherent video description,
    # not per-frame breakdowns.
    content = [{"type": "image", "image": f} for f in frames]
    content.append({"type": "text", "text": VLM_CAPTION_PROMPT_TEXT})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": VLM_SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated_ids, skip_special_tokens=True).strip()


def run_vlm_captioning(
    scene_videos: dict[str, str],
    annotations_dir: str,
    vlm_model_path: str | None,
    vlm_output_path: str,
    existing_vlm_captions: dict[str, str] | None = None,
) -> dict[str, str]:
    """Phase 1: Caption all scenes with CapRL-Qwen3VL-4B.

    Samples 8 frames from each scene's video and generates a caption.
    Saves intermediate VLM captions to vlm_output_path incrementally.
    Supports resume via existing_vlm_captions.
    """
    captions = dict(existing_vlm_captions) if existing_vlm_captions else {}

    scene_ids = sorted(scene_videos.keys())
    todo = [s for s in scene_ids if s not in captions]
    skipped = len(scene_ids) - len(todo)
    total = len(scene_ids)

    if skipped > 0:
        print(f"VLM phase: {skipped} already done, {len(todo)} remaining out of {total}")
    if len(todo) == 0:
        print("VLM phase: all scenes already captioned.")
        return captions

    model, processor = load_vlm_model(vlm_model_path)

    print(f"VLM phase: captioning {len(todo)} scenes ({NUM_SAMPLED_FRAMES} frames each)...")

    for i, scene_id in enumerate(todo):
        video_path = scene_videos[scene_id]

        try:
            caption = caption_video_vlm(video_path, model, processor)
            captions[scene_id] = caption
        except Exception as e:
            print(f"  [{skipped+i+1}/{total}] {scene_id}: VLM failed ({e}), using fallback")
            captions[scene_id] = build_fallback_caption(scene_id, annotations_dir)

        with open(vlm_output_path, "w") as f:
            json.dump(captions, f, indent=2, ensure_ascii=False)

        print(f"  [{skipped+i+1}/{total}] {scene_id}: OK ({len(captions[scene_id])} chars)")

    del model, processor
    torch.cuda.empty_cache()
    print("VLM phase complete, model unloaded.")

    return captions


# --- Phase 2: LLM rewriting with gpt-oss-120b ---

def load_llm_model(model_path: str | None):
    """Load gpt-oss-120b for caption rewriting.

    MoE model (120B total, 5.1B active). MXFP4 quantization is baked into
    the model config -- transformers handles it automatically.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_path is None:
        model_path = "openai/gpt-oss-120b"

    print(f"Loading LLM model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    ).eval()

    print("LLM model loaded.")
    return model, tokenizer


def _extract_rewritten_description(text: str) -> str | None:
    """Extract rewritten description from JSON output.

    Follows the EasyAnimate extract_output() pattern:
    parse {"rewritten description": "..."} from the LLM response.
    """
    match = re.search(r"\{(.+?)\}", text, re.DOTALL)
    if not match:
        return None

    inner = match.group(1).strip()
    prefix = '"rewritten description":'
    lower_inner = inner.lower()
    if lower_inner.startswith('"rewritten description":'):
        value = inner[len(prefix):].strip()
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1]

    # Fallback: try json.loads
    try:
        parsed = json.loads("{" + match.group(1) + "}")
        return parsed.get("rewritten description")
    except json.JSONDecodeError:
        return None


def rewrite_caption_llm(
    caption: str,
    model,
    tokenizer,
    max_retries: int = 3,
) -> str | None:
    """Rewrite a VLM caption using gpt-oss-120b.

    Returns the rewritten description, or None if parsing fails after retries.
    """
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": REWRITE_USER_PROMPT + caption},
    ]

    for attempt in range(max_retries):
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        rewritten = _extract_rewritten_description(response)
        if rewritten is not None and len(rewritten) > 50:
            return rewritten

        print(f"    Retry {attempt+1}/{max_retries}: failed to parse JSON from LLM output")

    return None


def run_llm_rewriting(
    vlm_captions: dict[str, str],
    annotations_dir: str,
    llm_model_path: str | None,
    output_path: str,
    existing_captions: dict[str, str] | None = None,
) -> dict[str, str]:
    """Phase 2: Rewrite all VLM captions with gpt-oss-120b.

    Saves final captions to output_path incrementally.
    Supports resume via existing_captions.
    """
    captions = dict(existing_captions) if existing_captions else {}

    scene_ids = sorted(vlm_captions.keys())
    todo = [s for s in scene_ids if s not in captions]
    skipped = len(scene_ids) - len(todo)
    total = len(scene_ids)

    if skipped > 0:
        print(f"LLM phase: {skipped} already done, {len(todo)} remaining out of {total}")
    if len(todo) == 0:
        print("LLM phase: all scenes already rewritten.")
        return captions

    model, tokenizer = load_llm_model(llm_model_path)

    print(f"LLM phase: rewriting {len(todo)} captions...")

    for i, scene_id in enumerate(todo):
        vlm_caption = vlm_captions[scene_id]

        try:
            rewritten = rewrite_caption_llm(vlm_caption, model, tokenizer)
            if rewritten is not None:
                captions[scene_id] = rewritten
            else:
                print(f"  [{skipped+i+1}/{total}] {scene_id}: JSON parse failed, keeping VLM caption")
                captions[scene_id] = vlm_caption
        except Exception as e:
            print(f"  [{skipped+i+1}/{total}] {scene_id}: LLM failed ({e}), keeping VLM caption")
            captions[scene_id] = vlm_caption

        with open(output_path, "w") as f:
            json.dump(captions, f, indent=2, ensure_ascii=False)

        print(f"  [{skipped+i+1}/{total}] {scene_id}: OK ({len(captions[scene_id])} chars)")

    del model, tokenizer
    torch.cuda.empty_cache()
    print("LLM phase complete, model unloaded.")

    return captions


# --- Main ---

def main():
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="Generate captions for CAD-Estate VACE training scenes."
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True,
        help="Path to vace_training_dataset directory containing *_rgb.mp4 videos.",
    )
    parser.add_argument(
        "--annotations_dir", type=str, required=True,
        help="Path to cad_estate annotations directory (contains scene_id/room_structure.npz).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to output captions.json. Defaults to <dataset_dir>/captions.json.",
    )
    parser.add_argument(
        "--fallback_only", action="store_true",
        help="Skip model captioning; generate template captions from room structure labels only.",
    )
    parser.add_argument(
        "--vlm_model_path", type=str, default=None,
        help="Path to CapRL-Qwen3VL-4B model. Default: internlm/CapRL-Qwen3VL-4B.",
    )
    parser.add_argument(
        "--llm_model_path", type=str, default=None,
        help="Path to gpt-oss-120b model. Default: openai/gpt-oss-120b.",
    )
    parser.add_argument(
        "--vlm_only", action="store_true",
        help="Only run Phase 1 (VLM captioning). Saves vlm_captions.json.",
    )
    parser.add_argument(
        "--llm_only", action="store_true",
        help="Only run Phase 2 (LLM rewriting). Requires existing vlm_captions.json.",
    )
    parser.add_argument(
        "--shard_id", type=int, default=None,
        help="Shard index for parallel processing (0-indexed). Use with --num_shards.",
    )
    parser.add_argument(
        "--num_shards", type=int, default=None,
        help="Total number of shards for parallel processing.",
    )
    args = parser.parse_args()

    # Resolve output paths
    output_path = args.output or os.path.join(args.dataset_dir, "captions.json")
    base, ext = os.path.splitext(output_path)
    vlm_output_path = f"{base}_vlm{ext}"

    # Find scene videos
    scene_videos = find_scene_videos(args.dataset_dir)
    print(f"Found {len(scene_videos)} scene videos in {args.dataset_dir}")
    if len(scene_videos) == 0:
        print("WARNING: No *_rgb.mp4 videos found. Checking annotation directories...")
        ann_path = Path(args.annotations_dir)
        if ann_path.is_dir():
            for scene_dir in sorted(ann_path.iterdir()):
                if scene_dir.is_dir():
                    scene_videos[scene_dir.name] = ""
            print(f"Found {len(scene_videos)} scenes from annotations directory")

    # Shard: select subset and apply per-shard output file suffixes
    if args.shard_id is not None and args.num_shards is not None:
        all_keys = sorted(scene_videos.keys())
        shard_keys = [k for i, k in enumerate(all_keys) if i % args.num_shards == args.shard_id]
        scene_videos = {k: scene_videos[k] for k in shard_keys}

        base_vlm, ext_vlm = os.path.splitext(vlm_output_path)
        vlm_output_path = f"{base_vlm}_shard{args.shard_id}{ext_vlm}"
        base_out, ext_out = os.path.splitext(output_path)
        output_path = f"{base_out}_shard{args.shard_id}{ext_out}"
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(scene_videos)} scenes -> {output_path}")

    # Resume: load existing intermediate and final captions
    existing_vlm_captions = {}
    if os.path.exists(vlm_output_path):
        with open(vlm_output_path, "r") as f:
            existing_vlm_captions = json.load(f)
        print(f"Loaded {len(existing_vlm_captions)} existing VLM captions from {vlm_output_path} (resume)")

    existing_captions = {}
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            existing_captions = json.load(f)
        print(f"Loaded {len(existing_captions)} existing final captions from {output_path} (resume)")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(vlm_output_path)), exist_ok=True)

    if args.fallback_only:
        captions = generate_fallback_captions(scene_videos, args.annotations_dir)
    else:
        vl_scenes = {k: v for k, v in scene_videos.items() if v}

        if not vl_scenes:
            print("No scene videos available for VLM captioning. Using fallback for all.")
            captions = generate_fallback_captions(scene_videos, args.annotations_dir)
        else:
            # Phase 1: VLM captioning (8 frames per video)
            if not args.llm_only:
                vlm_captions = run_vlm_captioning(
                    vl_scenes, args.annotations_dir,
                    args.vlm_model_path, vlm_output_path,
                    existing_vlm_captions,
                )
            else:
                vlm_captions = existing_vlm_captions
                assert len(vlm_captions) > 0, (
                    f"--llm_only requires existing VLM captions at {vlm_output_path}"
                )

            # Phase 2: LLM rewriting
            if not args.vlm_only:
                captions = run_llm_rewriting(
                    vlm_captions, args.annotations_dir,
                    args.llm_model_path, output_path,
                    existing_captions,
                )
            else:
                captions = vlm_captions

            # Add fallback captions for scenes without videos
            for scene_id in scene_videos:
                if scene_id not in captions:
                    captions[scene_id] = build_fallback_caption(
                        scene_id, args.annotations_dir,
                    )

    # Validate captions
    assert len(captions) > 0, "No captions generated"
    for scene_id, caption in captions.items():
        assert isinstance(caption, str) and len(caption) > 10, \
            f"Invalid caption for {scene_id}: {caption!r}"

    # Final save
    with open(output_path, "w") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(captions)} captions to {output_path}")


if __name__ == "__main__":
    main()
