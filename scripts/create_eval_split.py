#!/usr/bin/env python3
"""
Create train/eval split for VACE depth training.

Generates:
  - metadata.csv       (training only, excludes held-out video_ids)
  - metadata_eval.csv  (eval only, held-out video_ids)
  - eval_metadata.json (eval entries from metadata.json)
  - eval_captions.json (eval captions)

Usage:
    python scripts/create_eval_split.py
"""

import csv
import json
import os
import random

DATA_DIR = "/iopsstor/scratch/cscs/thomaskiefer/cad_estate/data/vace_training_dataset"

EVAL_VIDEO_IDS = {
    "--dubp2RBuc",   # 4 clips, hard (83.6% validity, P8)
    "3OnUUkgGe7Y",   # 6 clips, typical (93.2% validity, P63)
    "43tBrOV1-sY",   # 5 clips, clean + 1080p (95.4% validity, P83)
}

SEED = 42


def main():
    random.seed(SEED)

    with open(os.path.join(DATA_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    with open(os.path.join(DATA_DIR, "captions.json")) as f:
        captions = json.load(f)

    print(f"Total metadata entries: {len(metadata)}")
    print(f"Total captions: {len(captions)}")

    # Split metadata.json into train/eval
    train_meta, eval_meta = [], []
    for entry in metadata:
        video_id = entry.get("video_id", entry["clip_name"].rsplit("_", 1)[0])
        if video_id in EVAL_VIDEO_IDS:
            eval_meta.append(entry)
        else:
            train_meta.append(entry)

    # Split captions
    eval_captions = {}
    for entry in eval_meta:
        cn = entry["clip_name"]
        if cn in captions:
            eval_captions[cn] = captions[cn]

    print(f"\nTrain entries: {len(train_meta)}")
    print(f"Eval entries:  {len(eval_meta)} ({len(eval_captions)} with captions)")
    for vid in sorted(EVAL_VIDEO_IDS):
        count = sum(1 for e in eval_meta if e.get("video_id") == vid)
        print(f"  {vid}: {count} clips")

    # Write eval metadata.json and captions.json
    eval_meta_path = os.path.join(DATA_DIR, "eval_metadata.json")
    eval_caps_path = os.path.join(DATA_DIR, "eval_captions.json")
    with open(eval_meta_path, "w") as f:
        json.dump(eval_meta, f, indent=2)
    with open(eval_caps_path, "w") as f:
        json.dump(eval_captions, f, indent=2)
    print(f"\nWrote {eval_meta_path}")
    print(f"Wrote {eval_caps_path}")

    # Generate eval CSV (same format as training metadata.csv)
    eval_csv_rows = []
    for entry in eval_meta:
        clip_name = entry["clip_name"]
        if clip_name not in eval_captions:
            continue
        rgb = entry["rgb_path"]
        depth = entry["depth_path"]
        ref = entry.get("ref_path", "")
        validity = entry.get("validity_path", "")
        if not os.path.exists(os.path.join(DATA_DIR, rgb)):
            continue
        if not os.path.exists(os.path.join(DATA_DIR, depth)):
            continue
        eval_csv_rows.append({
            "video": rgb,
            "vace_video": depth,
            "vace_validity_mask": validity,
            "vace_reference_image": ref,
            "prompt": eval_captions[clip_name],
        })

    eval_csv_path = os.path.join(DATA_DIR, "metadata_eval.csv")
    with open(eval_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["video", "vace_video", "vace_validity_mask", "vace_reference_image", "prompt"]
        )
        writer.writeheader()
        writer.writerows(eval_csv_rows)
    print(f"Wrote {len(eval_csv_rows)} eval rows to {eval_csv_path}")

    # Regenerate training CSV (exclude eval video_ids)
    train_csv_rows = []
    for entry in train_meta:
        clip_name = entry["clip_name"]
        if clip_name not in captions:
            continue
        rgb = entry["rgb_path"]
        depth = entry["depth_path"]
        ref = entry.get("ref_path", "")
        validity = entry.get("validity_path", "")
        if not os.path.exists(os.path.join(DATA_DIR, rgb)):
            continue
        if not os.path.exists(os.path.join(DATA_DIR, depth)):
            continue
        train_csv_rows.append({
            "video": rgb,
            "vace_video": depth,
            "vace_validity_mask": validity,
            "vace_reference_image": ref,
            "prompt": captions[clip_name],
        })

    random.shuffle(train_csv_rows)
    train_csv_path = os.path.join(DATA_DIR, "metadata.csv")
    with open(train_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["video", "vace_video", "vace_validity_mask", "vace_reference_image", "prompt"]
        )
        writer.writeheader()
        writer.writerows(train_csv_rows)
    print(f"Wrote {len(train_csv_rows)} training rows to {train_csv_path}")


if __name__ == "__main__":
    main()
