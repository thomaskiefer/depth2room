#!/usr/bin/env python3
"""
Create metadata CSV for VACE depth-conditioned training.

Reads captions.json and metadata.json from the prepared dataset directory to
build a CSV with columns:
  video, vace_video, vace_reference_image, prompt

Usage:
    python -m depth2room.data.create_metadata \
        --data_dir /path/to/vace_training_dataset
"""

import argparse
import csv
import json
import os
import random


def main():
    parser = argparse.ArgumentParser(description="Create metadata CSV for VACE depth training.")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing prepared training data.",
    )
    parser.add_argument(
        "--captions_json", type=str, default=None,
        help="Path to captions.json. Defaults to <data_dir>/captions.json.",
    )
    parser.add_argument(
        "--metadata_json", type=str, default=None,
        help="Path to metadata.json. Defaults to <data_dir>/metadata.json.",
    )
    parser.add_argument(
        "--output_csv", type=str, default=None,
        help="Output CSV path. Defaults to <data_dir>/metadata.csv.",
    )
    parser.add_argument(
        "--no_ref_fraction", type=float, default=0.15,
        help="Fraction of samples that omit the reference image (default: 0.15).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if args.captions_json is None:
        args.captions_json = os.path.join(args.data_dir, "captions.json")
    if args.metadata_json is None:
        args.metadata_json = os.path.join(args.data_dir, "metadata.json")
    if args.output_csv is None:
        args.output_csv = os.path.join(args.data_dir, "metadata.csv")

    with open(args.captions_json, "r") as f:
        captions = json.load(f)
    print(f"Loaded {len(captions)} captions from {args.captions_json}")

    with open(args.metadata_json, "r") as f:
        metadata_list = json.load(f)
    print(f"Loaded {len(metadata_list)} scene entries from {args.metadata_json}")

    rows = []
    missing_caption = 0
    missing_files = 0

    for entry in metadata_list:
        clip_name = entry["clip_name"]
        depth_path = entry["depth_path"]
        rgb_path = entry["rgb_path"]
        ref_path = entry.get("ref_path")

        if clip_name not in captions:
            missing_caption += 1
            continue

        prompt = captions[clip_name]

        if not os.path.exists(os.path.join(args.data_dir, rgb_path)):
            missing_files += 1
            continue
        if not os.path.exists(os.path.join(args.data_dir, depth_path)):
            missing_files += 1
            continue

        # ref_path is always set in metadata.json; the no-ref dropout
        # happens here only (not doubled with prepare_data.py).
        include_ref = ref_path and (random.random() >= args.no_ref_fraction)
        ref_rel = ref_path if include_ref else ""

        validity_path = entry.get("validity_path", "")
        if validity_path and not os.path.exists(os.path.join(args.data_dir, validity_path)):
            missing_files += 1
            continue

        rows.append({
            "video": rgb_path,
            "vace_video": depth_path,
            "vace_validity_mask": validity_path,
            "vace_reference_image": ref_rel,
            "prompt": prompt,
        })

    if missing_caption > 0:
        print(f"Warning: {missing_caption} scenes had no caption in captions.json")
    if missing_files > 0:
        print(f"Warning: {missing_files} scenes had missing files on disk")

    assert len(rows) > 0, (
        f"No valid rows produced from {len(metadata_list)} metadata entries and "
        f"{len(captions)} captions ({missing_caption} missing captions, {missing_files} missing files)"
    )

    random.shuffle(rows)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "vace_video", "vace_validity_mask", "vace_reference_image", "prompt"])
        writer.writeheader()
        writer.writerows(rows)

    ref_count = sum(1 for r in rows if r["vace_reference_image"])
    no_ref_count = len(rows) - ref_count
    print(f"Written {len(rows)} rows to {args.output_csv}")
    print(f"  With reference image: {ref_count} ({100*ref_count/max(len(rows),1):.1f}%)")
    print(f"  Without reference image: {no_ref_count} ({100*no_ref_count/max(len(rows),1):.1f}%)")


if __name__ == "__main__":
    main()
