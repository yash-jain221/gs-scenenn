#!/usr/bin/env python3
"""
Compute per-scene overexposure thresholds by averaging only overexposed frames.
Overexposure score = fraction of pixels with intensity >= sat threshold.
"""

import argparse
import csv
import os

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Overexposure threshold per scene")
    p.add_argument("--raw-root", default="data/scenenn/raw",
                   help="Root folder containing SceneNN raw scenes")
    p.add_argument("--out-csv", default="analysis/overexp_thresholds.csv",
                   help="Output CSV path")
    p.add_argument("--scene-list", default=None,
                   help="Path to txt file with scene ids (one per line)")
    p.add_argument("--sat-thresh", type=int, default=250,
                   help="Grayscale saturation threshold (0-255)")

    overexp_group = p.add_mutually_exclusive_group(required=True)
    overexp_group.add_argument("--overexp-fixed", type=float, default=None,
                               metavar="FRACTION",
                               help="Mark frames overexposed if score >= this (e.g. 0.10)")
    overexp_group.add_argument("--overexp-percentile", type=float, default=None,
                               metavar="PCT",
                               help="Mark top PCT%% frames as overexposed per scene (e.g. 10)")
    return p.parse_args()


def find_scenes(root_dir):
    if not os.path.isdir(root_dir):
        return []
    return sorted(
        d for d in os.listdir(root_dir)
        if d.isdigit() and os.path.isdir(os.path.join(root_dir, d))
    )


def load_scene_list(path):
    with open(path) as f:
        lines = [line.strip() for line in f.readlines()]
    scenes = [line for line in lines if line and not line.startswith("#")]
    return scenes


def overexp_score(img_path, sat_thresh, bad_files=None):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        if bad_files is not None:
            bad_files.append(img_path)
        return None
    return float(np.mean(img >= sat_thresh))


def compute_scene_scores(scene_dir, sat_thresh, bad_files=None):
    images_dir = os.path.join(scene_dir, "image")
    if not os.path.isdir(images_dir):
        return []
    fnames = sorted(f for f in os.listdir(images_dir) if f.endswith(".png"))
    scores = []
    for fname in fnames:
        score = overexp_score(os.path.join(images_dir, fname), sat_thresh, bad_files)
        if score is not None:
            scores.append(score)
    return scores


def select_overexposed(scores, overexp_fixed=None, overexp_percentile=None):
    if not scores:
        return [], None
    arr = np.array(scores, dtype=np.float64)
    if overexp_percentile is not None:
        cutoff = float(np.percentile(arr, 100 - overexp_percentile))
    else:
        cutoff = float(overexp_fixed)
    over = arr[arr >= cutoff]
    return over.tolist(), cutoff


def main():
    args = parse_args()
    if args.scene_list:
        scenes = load_scene_list(args.scene_list)
    else:
        scenes = find_scenes(args.raw_root)
    if not scenes:
        raise SystemExit(f"No numeric scenes found in: {args.raw_root}")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    bad_files = []
    total_frames = 0
    total_overexp = 0
    overexp_sum = 0.0
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "scene_id",
            "n_frames",
            "n_overexp",
            "overexp_cutoff",
            "overexp_mean",
        ]
        writer.writerow(header)

        for scene_id in scenes:
            scene_dir = os.path.join(args.raw_root, scene_id)
            scores = compute_scene_scores(scene_dir, args.sat_thresh, bad_files)
            if not scores:
                writer.writerow([scene_id, 0, 0, "", ""])
                continue

            over, cutoff = select_overexposed(
                scores,
                overexp_fixed=args.overexp_fixed,
                overexp_percentile=args.overexp_percentile,
            )
            if over:
                over_mean = float(np.mean(over))
                overexp_sum += float(np.sum(over))
                total_overexp += len(over)
            else:
                over_mean = ""

            total_frames += len(scores)

            writer.writerow([
                scene_id,
                len(scores),
                len(over),
                f"{cutoff:.6f}",
                f"{over_mean:.6f}" if over_mean != "" else "",
            ])

        if total_overexp > 0:
            global_mean = overexp_sum / total_overexp
            writer.writerow([
                "ALL",
                total_frames,
                total_overexp,
                "",
                f"{global_mean:.6f}",
            ])

    print(f"Wrote: {args.out_csv}")
    if bad_files:
        bad_path = os.path.join(os.path.dirname(args.out_csv), "bad_images.txt")
        with open(bad_path, "w", newline="") as f:
            for path in bad_files:
                f.write(f"{path}\n")
        print(f"Bad images logged to: {bad_path}")


if __name__ == "__main__":
    main()
