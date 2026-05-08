#!/usr/bin/env python3
"""
SceneNN Indoor Scene Quality Analyser  v2
==========================================
Analyses indoor scenes for the 4 issues that matter most for 3DGS quality
(based on GaussianRoom, IndoorGS CVPR2025, and Indoor 3DGS without camera poses):

  1. BLUR           — Laplacian variance (low = blurry, noisy photometric signal)
  2. OVEREXPOSURE   — Saturated pixel fraction (high = photometric signal destroyed)
  3. TEXTURELESSNESS— Mean Sobel gradient magnitude per frame (low = flat walls/floors)
  4. POSE DRIFT     — Per-frame translation delta from trajectory (high = SLAM instability)

Everything else (optical flow, coverage, brightness delta) is removed.
These 4 directly map to the 3 known indoor 3DGS failure modes:
  - Bad photometric signal  → blur + overexposure
  - Underconstrained geometry → texturelessness
  - Wrong camera poses        → pose drift

Usage:
    python analyse_scenes_v2.py --scene-list scenes.txt
    python analyse_scenes_v2.py --scene-list scenes.txt --data-root data/scenenn/raw --out-dir analysis_v2

scenes.txt: one scene ID per line (e.g. 021)
"""

import os, sys, math, csv, argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", palette="muted")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-list",    required=True)
    p.add_argument("--data-root",     default="data/scenenn/raw")
    p.add_argument("--out-dir",       default="analysis_v2")
    p.add_argument("--max-frames",    type=int,   default=500,
                   help="Max frames sampled per scene (default 500)")
    # Thresholds for summary flagging
    p.add_argument("--blur-thresh",   type=float, default=80.0)
    p.add_argument("--overexp-thresh",type=float, default=0.05,
                   help="Fraction of pixels > 250 (default 0.05 = 5%%)")
    p.add_argument("--texture-thresh",type=float, default=8.0,
                   help="Mean Sobel gradient below this = textureless (default 8.0)")
    p.add_argument("--trans-thresh",  type=float, default=0.3,
                   help="Translation jump in metres (default 0.3)")
    return p.parse_args()


# ── TRAJECTORY ────────────────────────────────────────────────────────────────
def read_trajectory(path):
    poses = {}
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i < len(lines):
        fid = int(lines[i].split()[0])
        mat = np.array([[float(v) for v in lines[i+k].split()] for k in range(1,5)])
        poses[fid] = mat
        i += 5
    return poses

def parse_fid(fname):
    return int(''.join(filter(str.isdigit, Path(fname).stem)))

def trans_delta(pa, pb):
    return float(np.linalg.norm(pa[:3,3] - pb[:3,3]))


# ── PER-FRAME METRICS ────────────────────────────────────────────────────────
def blur_score(gray):
    """Laplacian variance — higher = sharper. Research: GS-Blur NeurIPS 2024."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def overexp_score(gray, sat=250):
    """Fraction of pixels >= sat. Research: LO-Gaussian Eurographics 2024."""
    return float(np.mean(gray >= sat))

def texture_score(gray):
    """
    Mean Sobel gradient magnitude.
    Low = textureless (flat walls, floors).
    Research: GaussianRoom (arxiv 2405.19671), IndoorGS CVPR 2025.
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.sqrt(gx**2 + gy**2).mean())


# ── PER-SCENE ────────────────────────────────────────────────────────────────
def analyse_scene(sid, data_root, args):
    img_dir   = os.path.join(data_root, sid, "image")
    traj_path = os.path.join(data_root, sid, "trajectory.log")

    if not os.path.isdir(img_dir) or not os.path.isfile(traj_path):
        print(f"  [SKIP] Missing data for scene {sid}")
        return None, None

    poses    = read_trajectory(traj_path)
    allfiles = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
    step     = max(1, math.ceil(len(allfiles) / args.max_frames))
    files    = allfiles[::step]
    print(f"  {len(files)}/{len(allfiles)} frames (step={step})")

    rows      = []
    prev_pose = None

    for fname in files:
        fid  = parse_fid(fname)
        gray = cv2.imread(os.path.join(img_dir, fname), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        bl = blur_score(gray)
        oe = overexp_score(gray)
        tx = texture_score(gray)

        td = 0.0
        if fid in poses and prev_pose is not None:
            td = trans_delta(prev_pose, poses[fid])
        if fid in poses:
            prev_pose = poses[fid]

        rows.append({
            "frame":        fname,
            "fid":          fid,
            "blur":         round(bl, 2),
            "overexp":      round(oe, 4),
            "texture":      round(tx, 3),
            "trans_delta_m":round(td, 4),
        })

    n = len(rows)
    summary = {
        "scene_id":         sid,
        "frames_analysed":  n,
        # means
        "avg_blur":         round(np.mean([r["blur"]    for r in rows]), 2),
        "avg_overexp":      round(np.mean([r["overexp"] for r in rows]), 4),
        "avg_texture":      round(np.mean([r["texture"] for r in rows]), 3),
        "avg_trans_delta":  round(np.mean([r["trans_delta_m"] for r in rows]), 4),
        # flagged frame counts
        "blurry_frames":    sum(1 for r in rows if r["blur"]         < args.blur_thresh),
        "overexp_frames":   sum(1 for r in rows if r["overexp"]      > args.overexp_thresh),
        "textureless_frames":sum(1 for r in rows if r["texture"]     < args.texture_thresh),
        "pose_jumps":       sum(1 for r in rows if r["trans_delta_m"]> args.trans_thresh),
    }
    # add percentages
    for k in ["blurry_frames","overexp_frames","textureless_frames","pose_jumps"]:
        summary[k.replace("_frames","_pct").replace("_jumps","_jump_pct")] = \
            round(summary[k] / n * 100, 1) if n > 0 else 0.0

    return rows, summary


# ── CSV OUTPUT ────────────────────────────────────────────────────────────────
def save_frames_csv(sid, rows, out_dir):
    if not rows: return
    path = os.path.join(out_dir, f"{sid}_frames.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

def save_summary_csv(summaries, out_dir):
    if not summaries: return
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summaries[0].keys())
        w.writeheader(); w.writerows(summaries)
    print(f"Summary CSV -> {path}")


# ── CHARTS ───────────────────────────────────────────────────────────────────
COLORS = ["#7B68EE", "#FF8C69", "#20B2AA", "#FFD700"]

def chart_issue_breakdown(summaries, out_dir):
    """Stacked bar: % of flagged frames per issue, sorted by total severity."""
    scenes = [s["scene_id"] for s in summaries]
    pct_keys   = ["blurry_pct","overexp_pct","textureless_pct","pose_jump_pct"]
    pct_labels = ["Blurry %","Overexposed %","Textureless %","Pose Jumps %"]

    # fix key name mismatch
    for s in summaries:
        s["blurry_pct"]      = round(s["blurry_frames"]      / s["frames_analysed"] * 100, 1)
        s["overexp_pct"]     = round(s["overexp_frames"]      / s["frames_analysed"] * 100, 1)
        s["textureless_pct"] = round(s["textureless_frames"]  / s["frames_analysed"] * 100, 1)
        s["pose_jump_pct"]   = round(s["pose_jumps"]          / s["frames_analysed"] * 100, 1)

    totals = [sum(s[k] for k in pct_keys) for s in summaries]
    order  = sorted(range(len(scenes)), key=lambda i: totals[i], reverse=True)
    scenes = [scenes[i] for i in order]
    sums   = [summaries[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(10, len(scenes)*1.3), 6))
    bottom  = np.zeros(len(scenes))
    for key, label, color in zip(pct_keys, pct_labels, COLORS):
        vals = np.array([s[key] for s in sums])
        ax.bar(scenes, vals, bottom=bottom, label=label, color=color)
        bottom += vals

    ax.set_title("Indoor Scene Quality Issues — sorted by severity\n"
                 "(GaussianRoom / IndoorGS CVPR 2025 failure modes)", fontsize=13)
    ax.set_xlabel("Scene ID")
    ax.set_ylabel("% of frames flagged")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "issue_breakdown.png"), dpi=150)
    plt.close(fig)
    print("  Saved: issue_breakdown.png")


def chart_per_scene_timeseries(sid, rows, out_dir):
    """4-panel per-frame timeseries for one scene."""
    fids    = [r["fid"]          for r in rows]
    blur    = [r["blur"]         for r in rows]
    overexp = [r["overexp"]*100  for r in rows]   # convert to %
    texture = [r["texture"]      for r in rows]
    trans   = [r["trans_delta_m"]for r in rows]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Scene {sid} — Frame-level quality metrics", fontsize=14)

    data = [
        (blur,    "Blur score (Laplacian var)",  "#7B68EE", 80,    "< 80 = blurry"),
        (overexp, "Overexposure (%pixels>250)",   "#FF8C69", 5,     "> 5% = overexposed"),
        (texture, "Texture score (Sobel mean)",   "#20B2AA", 8,     "< 8 = textureless"),
        (trans,   "Pose trans delta (m)",         "#FFD700", 0.3,   "> 0.3m = pose jump"),
    ]

    for ax, (vals, ylabel, color, thresh, note) in zip(axes, data):
        ax.plot(fids, vals, color=color, linewidth=0.8, alpha=0.9)
        ax.axhline(thresh, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.text(0.01, 0.92, note, transform=ax.transAxes,
                fontsize=8, color="white", alpha=0.7,
                bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a2e", alpha=0.5))
        ax.fill_between(fids, vals, alpha=0.15, color=color)

    axes[-1].set_xlabel("Frame ID")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{sid}_timeseries.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {sid}_timeseries.png")


def chart_metric_distributions(summaries, out_dir):
    """Grouped bar of avg metric values across scenes for direct comparison."""
    scenes = [s["scene_id"] for s in summaries]
    metrics = [
        ("avg_blur",       "Avg Blur Score"),
        ("avg_overexp",    "Avg Overexp (fraction)"),
        ("avg_texture",    "Avg Texture Score"),
        ("avg_trans_delta","Avg Pose Delta (m)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Average Quality Metric per Scene", fontsize=14)

    for ax, (key, title), color in zip(axes.ravel(), metrics, COLORS):
        vals = [s[key] for s in summaries]
        order = sorted(range(len(scenes)), key=lambda i: vals[i], reverse=True)
        sorted_scenes = [scenes[i] for i in order]
        sorted_vals   = [vals[i]   for i in order]
        bars = ax.bar(sorted_scenes, sorted_vals, color=color)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Scene ID")
        for bar, val in zip(bars, sorted_vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "metric_comparison.png"), dpi=150)
    plt.close(fig)
    print("  Saved: metric_comparison.png")


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    with open(args.scene_list) as f:
        scene_ids = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Analysing {len(scene_ids)} scene(s): {', '.join(scene_ids)}\n")

    summaries = []
    for sid in scene_ids:
        print(f"=== Scene {sid} ===")
        rows, summary = analyse_scene(sid, args.data_root, args)
        if summary is None:
            continue
        save_frames_csv(sid, rows, args.out_dir)
        summaries.append(summary)
        print(f"  blur={summary['avg_blur']:.1f}  overexp={summary['avg_overexp']:.3f}  "
              f"texture={summary['avg_texture']:.2f}  trans={summary['avg_trans_delta']:.3f}")
        print(f"  flagged: blurry={summary['blurry_frames']}  "
              f"overexp={summary['overexp_frames']}  "
              f"textureless={summary['textureless_frames']}  "
              f"pose_jumps={summary['pose_jumps']}\n")
        chart_per_scene_timeseries(sid, rows, args.out_dir)

    save_summary_csv(summaries, args.out_dir)

    if len(summaries) > 1:
        chart_issue_breakdown(summaries, args.out_dir)
        chart_metric_distributions(summaries, args.out_dir)
    elif len(summaries) == 1:
        print("(Only 1 scene — skipping multi-scene comparison charts)")

    print(f"\nDone. All outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
