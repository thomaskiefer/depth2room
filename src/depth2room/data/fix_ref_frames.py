#!/usr/bin/env python3
"""Fix reference images in the VACE training dataset.

Overwrites each scene's _ref.jpg with frame 0 from its _rgb.mp4 video
and updates ref_frame_idx to 0 in metadata.json.

Supports sharding for parallel execution across multiple jobs:
    # Run 4 shards in parallel:
    python -m depth2room.data.fix_ref_frames --num_shards 4 --shard_id 0
    python -m depth2room.data.fix_ref_frames --num_shards 4 --shard_id 1
    python -m depth2room.data.fix_ref_frames --num_shards 4 --shard_id 2
    python -m depth2room.data.fix_ref_frames --num_shards 4 --shard_id 3

After all shards complete, run once without sharding to consolidate metadata:
    python -m depth2room.data.fix_ref_frames
"""

import argparse
import json
import logging
import os

import av
from PIL import Image

DATA_DIR = "/iopsstor/scratch/cscs/thomaskiefer/cad_estate/data/vace_training_dataset"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def extract_frame0(video_path: str) -> Image.Image:
    """Extract the first frame from an mp4 file and return it as a PIL Image."""
    container = av.open(video_path)
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")
        container.close()
        return Image.fromarray(img)
    container.close()
    raise RuntimeError(f"No frames found in {video_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fix reference images: extract frame 0 from _rgb.mp4 and overwrite _ref.jpg"
    )
    parser.add_argument(
        "--data_dir", type=str, default=DATA_DIR,
        help="Root directory of the VACE training dataset",
    )
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    data_dir = args.data_dir
    metadata_path = os.path.join(data_dir, "metadata.json")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    total = len(metadata)
    log.info("Loaded metadata with %d scenes", total)

    # Determine this shard's slice
    shard_indices = list(range(args.shard_id, total, args.num_shards))
    log.info("Shard %d/%d: processing %d scenes", args.shard_id, args.num_shards, len(shard_indices))

    fixed = 0
    skipped = 0
    failed = 0

    for count, idx in enumerate(shard_indices):
        entry = metadata[idx]
        clip_name = entry["clip_name"]

        # Resume-safe: skip if already fixed
        if entry.get("ref_frame_idx") == 0:
            skipped += 1
            continue

        rgb_path = os.path.join(data_dir, entry["rgb_path"])
        ref_path = os.path.join(data_dir, entry["ref_path"])

        if not os.path.exists(rgb_path):
            log.warning("RGB video not found for %s: %s", clip_name, rgb_path)
            failed += 1
            continue

        try:
            img = extract_frame0(rgb_path)
            img.save(ref_path, quality=95)
            entry["ref_frame_idx"] = 0
            fixed += 1
        except Exception as e:
            log.error("Failed to fix %s: %s", clip_name, e)
            failed += 1
            continue

        if (count + 1) % 100 == 0:
            log.info("Progress: %d/%d (fixed=%d, skipped=%d, failed=%d)",
                     count + 1, len(shard_indices), fixed, skipped, failed)

    log.info("Done. fixed=%d, skipped=%d, failed=%d", fixed, skipped, failed)

    # Also update the per-scene _depth_meta.json files
    for idx in shard_indices:
        entry = metadata[idx]
        clip_name = entry["clip_name"]
        video_id = entry["video_id"]
        meta_path = os.path.join(data_dir, video_id, f"{clip_name}_depth_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    depth_meta = json.load(f)
                depth_meta["ref_frame_idx"] = 0
                with open(meta_path, "w") as f:
                    json.dump(depth_meta, f)
            except Exception as e:
                log.warning("Could not update depth_meta for %s: %s", clip_name, e)

    # Save updated metadata.json
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Saved updated metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
