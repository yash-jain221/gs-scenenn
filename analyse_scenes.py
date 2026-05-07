"""
SceneNN Frame & Trajectory Analysis Tool
=========================================
Analyses each scene in a list for the following issues:

  1. Motion  — optical-flow magnitude, camera-shake (RPE), camera-translation delta
  2. Lighting — brightness delta, overexposure ratio, auto-exposure events
  3. Texture  — textureless-flat-surface ratio (Laplacian + gradient variance)
  4. Coverage — per-voxel view count from frustum projection (regions seen <= 2 views)

Usage:
    python analyse_scenes.py --scene-list scenes.txt [--data-root data/scenenn/raw] [--out-dir analysis]

scenes.txt format: one scene id per line, e.g.:
    021
    025
    016
"""

import os
import sys
import argparse
import math
import csv
from pathlib import Path

import cv2
import numpy as np


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SceneNN Frame & Trajectory Analyser")
    p.add_argument("--scene-list",  required=True, help="Path to .txt file with one scene ID per line")
    p.add_argument("--data-root",   default="data/scenenn/raw", help="Root folder containing raw scene dirs")
    p.add_argument("--out-dir",     default="analysis",  help="Output directory for CSV + summary reports")
    p.add_argument("--max-frames",  type=int, default=500,
                   help="Max frames to process per scene (uniformly sampled, default 500)")
    # Thresholds
    p.add_argument("--blur-thresh",        type=float, default=80.0,  help="Laplacian variance threshold for blur")
    p.add_argument("--flow-thresh",        type=float, default=15.0,  help="Optical flow (px) spike threshold")
    p.add_argument("--brightness-thresh",  type=float, default=20.0,  help="Brightness delta spike threshold")
    p.add_argument("--overexpose-thresh",  type=float, default=5.0,   help="%% of pixels > 250 to flag overexposure")
    p.add_argument("--texture-thresh",     type=float, default=100.0, help="Gradient variance below = textureless")
    p.add_argument("--trans-thresh",       type=float, default=0.3,   help="Translation jump (m) to flag pose discontinuity")
    p.add_argument("--coverage-voxel-m",   type=float, default=0.1,   help="Voxel size (m) for coverage grid")
    p.add_argument("--coverage-max-views", type=int,   default=2,     help="Flag voxels seen <= this many times")
    return p.parse_args()


# ── TRAJECTORY ────────────────────────────────────────────────────────────────
def read_trajectory(path):
    """Returns dict {frame_id: 4x4 c2w numpy array}."""
    poses = {}
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        fid = int(parts[0])
        mat = np.array([
            [float(v) for v in lines[i+1].split()],
            [float(v) for v in lines[i+2].split()],
            [float(v) for v in lines[i+3].split()],
            [float(v) for v in lines[i+4].split()],
        ])
        poses[fid] = mat
        i += 5
    return poses


def parse_fid(fname):
    digits = ''.join(filter(str.isdigit, Path(fname).stem))
    return int(digits)


# ── METRIC HELPERS ────────────────────────────────────────────────────────────
def blur_score(gray):
    """Laplacian variance — higher = sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def texture_score(gray):
    """
    Mean gradient magnitude across the image.
    Low value = textureless/flat.
    Uses Sobel gradients — research basis: Gradient Response Maps (Hinterstoisser et al., PAMI 2012).
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    return float(mag.mean())


def textureless_ratio(gray, patch_size=32, thresh=10.0):
    """
    Fraction of non-overlapping patches where mean gradient magnitude < thresh.
    Returns 0.0 – 1.0 (1.0 = fully textureless).
    """
    h, w = gray.shape
    total = bad = 0
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = gray[y:y+patch_size, x:x+patch_size]
            gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
            if np.sqrt(gx**2 + gy**2).mean() < thresh:
                bad += 1
            total += 1
    return bad / total if total > 0 else 0.0


def optical_flow_magnitude(prev_gray, curr_gray):
    """
    Dense Farneback optical flow, returns mean and 95th-percentile magnitude (pixels).
    Research basis: MotionGS (NeurIPS 2024) uses dense optical flow to detect motion priors.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return float(mag.mean()), float(np.percentile(mag, 95))


def translation_delta(pose_a, pose_b):
    """
    Euclidean distance between camera centres in world space.
    Research basis: Relative Pose Error (RPE) used in SLAM benchmarking
    (Sturm et al. TUM RGB-D benchmark, 2012).
    """
    return float(np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3]))


def rotation_delta_deg(pose_a, pose_b):
    """
    Angular difference between two rotation matrices (degrees).
    dR = Ra^T * Rb; angle = arccos((trace(dR) - 1) / 2)
    Research basis: RPE rotational component (Sturm et al., 2012).
    """
    Ra = pose_a[:3, :3]
    Rb = pose_b[:3, :3]
    dR = Ra.T @ Rb
    cos_angle = (np.trace(dR) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(math.degrees(math.acos(cos_angle)))


# ── COVERAGE ANALYSIS ─────────────────────────────────────────────────────────
def analyse_coverage(poses_ordered, voxel_size=0.1, max_views=2,
                     fx=544.47, fy=544.47, cx=320.0, cy=240.0,
                     W=640, H=480, depth_range=(0.3, 5.0)):
    """
    Project a depth grid into each camera frustum and count how many cameras
    observe each voxel. Returns fraction of voxels seen <= max_views times.

    Research basis: SceneSplat (CVPR 2025) identifies under-observed regions
    as a key cause of quality degradation in indoor 3DGS.
    """
    positions = np.array([p[:3, 3] for p in poses_ordered])
    scene_min = positions.min(axis=0) - 1.0
    scene_max = positions.max(axis=0) + 1.0

    xs = np.arange(scene_min[0], scene_max[0], voxel_size)
    ys = np.arange(scene_min[1], scene_max[1], voxel_size)
    zs = np.arange(scene_min[2], scene_max[2], voxel_size)

    # Sample a subset of grid points to keep it fast
    MAX_VOXELS = 50000
    gx, gy, gz = np.meshgrid(xs[::max(1, len(xs)//20)],
                               ys[::max(1, len(ys)//20)],
                               zs[::max(1, len(zs)//20)])
    pts_world = np.stack([gx.ravel(), gy.ravel(), gz.ravel(), 
                          np.ones(gx.size)], axis=1)  # (N, 4)
    if len(pts_world) > MAX_VOXELS:
        idx = np.random.choice(len(pts_world), MAX_VOXELS, replace=False)
        pts_world = pts_world[idx]

    view_counts = np.zeros(len(pts_world), dtype=np.int32)

    for c2w in poses_ordered:
        w2c = np.linalg.inv(c2w)
        pts_cam = (w2c @ pts_world.T).T  # (N, 4)
        Z = pts_cam[:, 2]
        valid_depth = (Z > depth_range[0]) & (Z < depth_range[1])

        u = fx * pts_cam[:, 0] / np.where(Z != 0, Z, 1e-6) + cx
        v = fy * pts_cam[:, 1] / np.where(Z != 0, Z, 1e-6) + cy

        in_frustum = valid_depth & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        view_counts[in_frustum] += 1

    low_coverage = (view_counts > 0) & (view_counts <= max_views)
    observed = view_counts > 0
    frac = low_coverage.sum() / observed.sum() if observed.sum() > 0 else 0.0
    return float(frac), int(low_coverage.sum()), int(observed.sum())


# ── PER-SCENE ANALYSIS ────────────────────────────────────────────────────────
def analyse_scene(scene_id, data_root, args):
    scene_dir  = os.path.join(data_root, scene_id)
    traj_path  = os.path.join(scene_dir, "trajectory.log")
    images_dir = os.path.join(scene_dir, "image")

    if not os.path.isdir(images_dir) or not os.path.isfile(traj_path):
        print(f"  [SKIP] Missing image folder or trajectory.log for scene {scene_id}")
        return None, None

    print(f"  Reading trajectory...")
    poses = read_trajectory(traj_path)

    print(f"  Listing images...")
    all_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
    if not all_files:
        print(f"  [SKIP] No PNG images found for scene {scene_id}")
        return None, None

    # Subsample if needed
    step = max(1, math.ceil(len(all_files) / args.max_frames))
    files = all_files[::step]
    print(f"  Using {len(files)}/{len(all_files)} frames (step={step})")

    frame_rows = []
    prev_gray  = None
    prev_mean  = None
    prev_pose  = None

    for i, fname in enumerate(files):
        fid  = parse_fid(fname)
        path = os.path.join(images_dir, fname)
        img  = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # --- 1. Motion metrics ---
        flow_mean = flow_p95 = 0.0
        if prev_gray is not None:
            flow_mean, flow_p95 = optical_flow_magnitude(prev_gray, gray)

        # --- 2. Lighting metrics ---
        mean_brightness  = float(gray.mean())
        brightness_delta = abs(mean_brightness - prev_mean) if prev_mean is not None else 0.0
        overexpose_pct   = float((gray > 250).sum() / gray.size * 100)
        # Saturation spike = auto-exposure event (Kinect AGC behaviour)
        mean_saturation  = float(hsv[:, :, 1].mean())

        # --- 3. Texture metrics ---
        lap_score = blur_score(gray)
        tex_score = texture_score(gray)
        tl_ratio  = textureless_ratio(gray, patch_size=32, thresh=10.0)

        # --- 4. Pose metrics (trajectory) ---
        trans_delta = rot_delta = 0.0
        if fid in poses and prev_pose is not None:
            trans_delta = translation_delta(prev_pose, poses[fid])
            rot_delta   = rotation_delta_deg(prev_pose, poses[fid])
        pose_valid = fid in poses

        frame_rows.append({
            "frame":           fname,
            "fid":             fid,
            "pose_valid":      pose_valid,
            # Motion
            "flow_mean_px":    round(flow_mean, 3),
            "flow_p95_px":     round(flow_p95, 3),
            # Pose
            "trans_delta_m":   round(trans_delta, 4),
            "rot_delta_deg":   round(rot_delta, 3),
            # Lighting
            "brightness":      round(mean_brightness, 2),
            "brightness_delta":round(brightness_delta, 2),
            "overexpose_pct":  round(overexpose_pct, 2),
            "saturation":      round(mean_saturation, 2),
            # Texture
            "blur_score":      round(lap_score, 2),
            "texture_score":   round(tex_score, 3),
            "textureless_ratio": round(tl_ratio, 3),
        })

        prev_gray = gray
        prev_mean = mean_brightness
        if fid in poses:
            prev_pose = poses[fid]

        if (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{len(files)} frames...")

    # --- Coverage analysis ---
    print("  Running coverage analysis (this may take ~30s)...")
    ordered_poses = [poses[parse_fid(f)] for f in files if parse_fid(f) in poses]
    low_cov_frac, low_cov_voxels, total_observed = analyse_coverage(
        ordered_poses,
        voxel_size=args.coverage_voxel_m,
        max_views=args.coverage_max_views
    )

    # --- Scene-level summary ---
    rows     = frame_rows
    n        = len(rows)
    summary  = {
        "scene_id":               scene_id,
        "total_frames":           len(all_files),
        "analysed_frames":        n,
        # Motion
        "flow_mean_avg":          round(np.mean([r["flow_mean_px"] for r in rows]), 2),
        "flow_spikes":            sum(1 for r in rows if r["flow_p95_px"] > args.flow_thresh),
        "pose_trans_spikes":      sum(1 for r in rows if r["trans_delta_m"] > args.trans_thresh),
        "pose_rot_spikes_deg":    sum(1 for r in rows if r["rot_delta_deg"] > 30),
        # Lighting
        "brightness_spikes":      sum(1 for r in rows if r["brightness_delta"] > args.brightness_thresh),
        "overexposed_frames":     sum(1 for r in rows if r["overexpose_pct"] > args.overexpose_thresh),
        "auto_expose_events":     sum(1 for r in rows if r["saturation"] < 10 and r["brightness_delta"] > args.brightness_thresh),
        # Texture
        "blurry_frames":          sum(1 for r in rows if r["blur_score"] < args.blur_thresh),
        "textureless_frames":     sum(1 for r in rows if r["textureless_ratio"] > 0.5),
        "avg_texture_score":      round(np.mean([r["texture_score"] for r in rows]), 2),
        # Coverage
        "low_coverage_frac":      round(low_cov_frac, 4),
        "low_coverage_voxels":    low_cov_voxels,
        "total_observed_voxels":  total_observed,
    }

    return frame_rows, summary


# ── REPORT WRITING ────────────────────────────────────────────────────────────
def write_frame_csv(scene_id, frame_rows, out_dir):
    path = os.path.join(out_dir, f"{scene_id}_frames.csv")
    if not frame_rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=frame_rows[0].keys())
        writer.writeheader()
        writer.writerows(frame_rows)
    print(f"  Wrote frame CSV -> {path}")


def write_summary_txt(summaries, out_dir, args):
    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w") as f:
        f.write("SceneNN Frame & Trajectory Analysis — Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write("Thresholds used:\n")
        f.write(f"  Blur           < {args.blur_thresh}  (Laplacian variance)\n")
        f.write(f"  Optical flow   > {args.flow_thresh} px  (95th-pct, Farneback dense flow)\n")
        f.write(f"  Brightness Δ   > {args.brightness_thresh}   (auto-exposure proxy)\n")
        f.write(f"  Overexposure   > {args.overexpose_thresh}%  pixels > 250\n")
        f.write(f"  Texture        > 50% patches with gradient mag < 10\n")
        f.write(f"  Pose trans     > {args.trans_thresh} m  per frame (RPE, Sturm 2012)\n")
        f.write(f"  Coverage       voxels seen <= {args.coverage_max_views} views ({args.coverage_voxel_m}m voxels)\n\n")

        header = (f"{'Scene':<8} {'Frames':>7} {'FlowSpikes':>10} {'PoseJumps':>9} "
                  f"{'BrtSpikes':>9} {'Overexp':>7} {'Blurry':>7} "
                  f"{'Textureless':>11} {'LowCovFrac':>10}")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for s in summaries:
            f.write(
                f"{s['scene_id']:<8} "
                f"{s['analysed_frames']:>7} "
                f"{s['flow_spikes']:>10} "
                f"{s['pose_trans_spikes']:>9} "
                f"{s['brightness_spikes']:>9} "
                f"{s['overexposed_frames']:>7} "
                f"{s['blurry_frames']:>7} "
                f"{s['textureless_frames']:>11} "
                f"{s['low_coverage_frac']:>10.3f}\n"
            )
        f.write("\n")

    print(f"\nSummary written -> {path}")


def write_summary_csv(summaries, out_dir):
    path = os.path.join(out_dir, "summary.csv")
    if not summaries:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Summary CSV written -> {path}")


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    with open(args.scene_list) as f:
        scene_ids = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    if not scene_ids:
        print("No scene IDs found in scene list file.")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Analysing {len(scene_ids)} scene(s): {', '.join(scene_ids)}\n")

    summaries = []
    for sid in scene_ids:
        print(f"\n{'='*50}")
        print(f"Scene: {sid}")
        print(f"{'='*50}")
        frame_rows, summary = analyse_scene(sid, args.data_root, args)
        if summary is None:
            continue
        write_frame_csv(sid, frame_rows, args.out_dir)
        summaries.append(summary)

        # Quick per-scene console summary
        print(f"\n  === {sid} Summary ===")
        print(f"  Motion  — Flow spikes: {summary['flow_spikes']}  |  Pose jumps: {summary['pose_trans_spikes']}")
        print(f"  Lighting— Brightness spikes: {summary['brightness_spikes']}  |  Overexposed: {summary['overexposed_frames']}")
        print(f"  Texture — Blurry: {summary['blurry_frames']}  |  Textureless: {summary['textureless_frames']}")
        print(f"  Coverage— Low-view fraction: {summary['low_coverage_frac']:.3f}  "
              f"({summary['low_coverage_voxels']} voxels / {summary['total_observed_voxels']} observed)")

    write_summary_txt(summaries, args.out_dir, args)
    write_summary_csv(summaries, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
