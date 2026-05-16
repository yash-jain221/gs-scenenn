#!/usr/bin/env python3
"""
scenenn_to_colmap.py
--------------------
All-in-one SceneNN → COLMAP converter with uniform-coverage frame selection.

Frame selection algorithm:
  - Splits all frames into N equal temporal windows
  - Inside each window, picks the frame most visually different
    from the previously selected frame (Laplacian variance of diff image)
  - Guarantees whole-scene coverage regardless of scene dynamics

Pipeline:
  1. Frame selection (uniform windows + visual diff)
  2. Copy selected RGB + depth → output/images/, output/depth/
  3. Write cameras.bin + images.bin from GT trajectory.log
  4. COLMAP feature_extractor → exhaustive_matcher → point_triangulator
  5. Final points3D.bin (triangulated sparse cloud)

Usage:
    python scenenn_to_colmap.py --scene_dir /path/to/scenenn/raw/005
    python scenenn_to_colmap.py --scene_dir /path/to/005 --target_frames 300
    python scenenn_to_colmap.py --scene_dir /path/to/005 --target_frames 200 --colmap /usr/local/bin/colmap

SceneNN camera intrinsics (Asus Xtion Pro, 640x480):
    fx = fy = 544.47,  cx = 320.0,  cy = 240.0
"""

import os, sys, glob, struct, shutil, subprocess, argparse
import numpy as np
import cv2
import json

# ── SceneNN intrinsics ──────────────────────────────────────────────────────
WIDTH, HEIGHT = 640, 480
FX, FY       = 544.47, 544.47
CX, CY       = 320.0,  240.0


# ── Utilities ────────────────────────────────────────────────────────────────
def parse_frame_id(filename):
    stem   = os.path.splitext(os.path.basename(filename))[0]
    digits = "".join(c for c in stem if c.isdigit())
    return int(digits)


def run_colmap(cmd, desc):
    print(f"\n  >> {desc}")
    print(f"     {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        sys.exit(f"ERROR: '{desc}' failed (exit {result.returncode})")


def visual_diff_score(img_a, img_b):
    """
    Returns a scalar difference score between two grayscale images.
    Uses mean absolute difference — fast and robust enough for frame selection.
    """
    diff = cv2.absdiff(img_a, img_b)
    return float(np.mean(diff))


# ── Step 1 — Frame selection ──────────────────────────────────────────────────
def select_frames(images_dir, target_frames, resize_w=160, resize_h=120):
    """
    Uniform-coverage frame selector:
      1. Sort all frames chronologically
      2. Split into `target_frames` equal temporal windows
      3. Inside each window, pick the frame with highest visual diff
         from the previously selected frame
      4. Returns sorted list of selected frame IDs

    Images are resized to (resize_w x resize_h) for fast comparison.
    """
    all_files = sorted(f for f in os.listdir(images_dir) if f.endswith(".png"))
    if not all_files:
        sys.exit(f"ERROR: No PNG files found in {images_dir}")

    total = len(all_files)
    n_windows = min(target_frames, total)
    window_size = total / n_windows

    print(f"\nStep 1: Frame selection")
    print(f"  Total frames    : {total}")
    print(f"  Target selected : {n_windows}")
    print(f"  Window size     : {window_size:.1f} frames")
    print(f"  Comparison res  : {resize_w}x{resize_h}")
    print(f"  Loading thumbnails ...", end="", flush=True)

    # Pre-load all frames as small grayscale thumbnails
    thumbs = []
    for fname in all_files:
        img = cv2.imread(os.path.join(images_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            thumbs.append(np.zeros((resize_h, resize_w), dtype=np.uint8))
        else:
            thumbs.append(cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_AREA))

    print(f" done ({total} thumbnails)")

    selected_fids = []
    prev_thumb    = None

    for w in range(n_windows):
        # Window boundaries
        start = int(w * window_size)
        end   = int((w + 1) * window_size)
        end   = min(end, total)
        if start >= end:
            continue

        if prev_thumb is None:
            # First window: pick middle frame (avoid edge artifacts)
            mid = (start + end) // 2
            best_idx = mid
        else:
            # Pick frame with highest diff from last selected
            best_score = -1
            best_idx   = start
            for i in range(start, end):
                score = visual_diff_score(prev_thumb, thumbs[i])
                if score > best_score:
                    best_score = score
                    print(f"  Window {w}: best frame {i} ({all_files[i]}) (best score={score:.2f})", end="\r", flush=True)
                    best_idx   = i

        selected_fids.append(parse_frame_id(all_files[best_idx]))
        prev_thumb = thumbs[best_idx]

        if (w + 1) % 50 == 0 or w == n_windows - 1:
            print(f"  Windows processed: {w+1}/{n_windows}", end="\r", flush=True)

    print(f"\n  Selected        : {len(selected_fids)} frames")
    return selected_fids


# ── Step 2 — trajectory.log ───────────────────────────────────────────────────
def read_trajectory(path):
    poses = {}
    with open(path) as f:
        lines = [l.rstrip() for l in f]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        parts = line.split()
        if len(parts) < 2:
            i += 1
            continue
        try:
            fid = int(parts[0])
            mat = np.array([
                list(map(float, lines[i+1].split())),
                list(map(float, lines[i+2].split())),
                list(map(float, lines[i+3].split())),
                list(map(float, lines[i+4].split())),
            ])
            poses[fid] = mat
        except (IndexError, ValueError):
            pass
        i += 5
    return poses


# ── Step 3 — copy RGB + depth ─────────────────────────────────────────────────
def copy_frames(scene_dir, matched_fids, out_img_dir, out_depth_dir):
    images_dir = os.path.join(scene_dir, "image")
    depth_dir  = os.path.join(scene_dir, "depth")
    has_depth  = os.path.isdir(depth_dir)

    os.makedirs(out_img_dir, exist_ok=True)
    if has_depth:
        os.makedirs(out_depth_dir, exist_ok=True)

    img_lookup   = {parse_frame_id(f): f
                    for f in os.listdir(images_dir) if f.endswith(".png")}
    depth_lookup = ({parse_frame_id(f): f
                     for f in os.listdir(depth_dir) if f.endswith(".png")}
                    if has_depth else {})

    copied_rgb = copied_depth = missing_depth = 0
    for fid in matched_fids:
        if fid in img_lookup:
            shutil.copy2(os.path.join(images_dir, img_lookup[fid]),
                         os.path.join(out_img_dir, f"{fid:06d}.png"))
            copied_rgb += 1
        if has_depth:
            if fid in depth_lookup:
                shutil.copy2(os.path.join(depth_dir, depth_lookup[fid]),
                             os.path.join(out_depth_dir, f"{fid:06d}.png"))
                copied_depth += 1
            else:
                missing_depth += 1

    print(f"  RGB copied    : {copied_rgb}")
    if has_depth:
        print(f"  Depth copied  : {copied_depth}")
        if missing_depth:
            print(f"  Depth missing : {missing_depth}")
    else:
        print("  Depth dir     : not found, skipping")
    return has_depth


# ── Step 4 — COLMAP binary writers ───────────────────────────────────────────
def write_cameras_bin(path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<i", 1))   # PINHOLE
        f.write(struct.pack("<Q", WIDTH))
        f.write(struct.pack("<Q", HEIGHT))
        for p in (FX, FY, CX, CY):
            f.write(struct.pack("<d", p))


def rotmat_to_quat(R):
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return 0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return (R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s


def write_images_bin(path, poses, fid_list):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(fid_list)))
        for img_id, fid in enumerate(fid_list, start=1):
            c2w   = poses[fid]
            R_w2c = c2w[:3, :3].T
            t_w2c = -R_w2c @ c2w[:3, 3]
            qw, qx, qy, qz = rotmat_to_quat(R_w2c)
            f.write(struct.pack("<I",    img_id))
            f.write(struct.pack("<dddd", qw, qx, qy, qz))
            f.write(struct.pack("<ddd",  *t_w2c))
            f.write(struct.pack("<I",    1))
            f.write((f"{fid:06d}.png\x00").encode("utf-8"))
            f.write(struct.pack("<Q",    0))

def export_poses(poses, matched_fids, out_dir):
    """Export poses in DA3-compatible format (N, 3, 4 numpy array)"""
    os.makedirs(out_dir, exist_ok=True)

    extrinsics = []
    for fid in matched_fids:
        c2w = poses[fid]          # [4, 4] camera-to-world (SceneNN format)
        extrinsics.append(c2w)

    extrinsics_np = np.stack(extrinsics)  # [N, 4, 4]
    np.save(os.path.join(out_dir, "extrinsics.npy"), extrinsics_np)

    # Also save intrinsics (same for all frames in SceneNN)
    intrinsic = np.array([
        [FX,  0,  CX],
        [0,  FY,  CY],
        [0,   0,   1]
    ], dtype=np.float64)
    # Tile it for all N frames
    intrinsics_np = np.tile(intrinsic[None], (len(matched_fids), 1, 1))  # [N, 3, 3]
    np.save(os.path.join(out_dir, "intrinsics.npy"), intrinsics_np)

    # Also save frame IDs so you know which GT image corresponds to which pose
    with open(os.path.join(out_dir, "frame_ids.json"), "w") as f:
        json.dump(matched_fids, f)

    print(f"\nStep 4.5: DA3 pose export")
    print(f"  Extrinsics : {extrinsics_np.shape}  → da3_poses/extrinsics.npy")
    print(f"  Intrinsics : {intrinsics_np.shape} → da3_poses/intrinsics.npy")
    print(f"  Frame IDs  : {len(matched_fids)}      → da3_poses/frame_ids.json")

def write_empty_points3d_bin(path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 0))

def batch_frames(fid_list, batch_size):
    """Split fid_list into batches of batch_size (no overlap)."""
    return [fid_list[i:i + batch_size] for i in range(0, len(fid_list), batch_size)]

# ── Steps 5–7 — COLMAP triangulation ─────────────────────────────────────────
def run_colmap_triangulation(colmap_exe, output_dir, sparse_dir, use_gpu=True):
    db_path     = os.path.join(output_dir, "database.db")
    print(f"\nStep 5–7: Running COLMAP triangulation with database {db_path} ...")
    out_img_dir = os.path.join(output_dir, "images")
    gpu_flag    = "1" if use_gpu else "0"

    if os.path.exists(db_path):
        os.remove(db_path)

    run_colmap([
        colmap_exe, "feature_extractor",
        "--database_path",                db_path,
        "--image_path",                   out_img_dir,
        "--ImageReader.single_camera",    "1",
        "--ImageReader.camera_model",     "PINHOLE",
        "--ImageReader.camera_params",    f"{FX},{FY},{CX},{CY}",
    ], "COLMAP feature_extractor")

    run_colmap([
        colmap_exe, "exhaustive_matcher",
        "--database_path",        db_path,
    ], "COLMAP exhaustive_matcher")

    tri_dir = os.path.join(output_dir, "sparse", "0_tri")
    if os.path.isdir(tri_dir):
        shutil.rmtree(tri_dir)
    os.makedirs(tri_dir, exist_ok=True)

    run_colmap([
        colmap_exe, "point_triangulator",
        "--database_path",                          db_path,
        "--image_path",                             out_img_dir,
        "--input_path",                             sparse_dir,
        "--output_path",                            tri_dir,
        "--Mapper.ba_refine_focal_length",          "0",
        "--Mapper.ba_refine_principal_point",       "0",
        "--Mapper.ba_refine_extra_params",          "0",
    ], "COLMAP point_triangulator")

    shutil.rmtree(sparse_dir)
    shutil.move(tri_dir, sparse_dir)
    print(f"  Sparse model written to {sparse_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="SceneNN → COLMAP: uniform frame selection + GT poses + triangulation"
    )
    ap.add_argument("--scene_dir",      required=True,
                    help="Scene root: image/, depth/, trajectory.log")
    ap.add_argument("--output_dir",     default=None,
                    help="Output root (default: <scene_dir>/../colmap_<name>)")
    ap.add_argument("--target_frames",  type=int, default=300,
                    help="Number of frames to select (default: 300)")
    ap.add_argument("--colmap",         default="colmap",
                    help="COLMAP executable (default: colmap)")
    ap.add_argument("--no_gpu",         action="store_true",
                    help="Disable GPU for COLMAP")
    ap.add_argument("--skip_colmap",    action="store_true",
                    help="Skip COLMAP triangulation (just prepare images + poses)")
    ap.add_argument("--batch_size", type=int, default=0,
                help="If >0, split selected frames into batches of this size")
    args = ap.parse_args()

    scene_dir  = os.path.abspath(args.scene_dir)
    scene_name = os.path.basename(scene_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else \
                 os.path.join(os.path.dirname(scene_dir), f"colmap_{scene_name}")
    sparse_dir = os.path.join(output_dir, "sparse", "0")

    print("=" * 60)
    print(f"Scene          : {scene_dir}")
    print(f"Output         : {output_dir}")
    print(f"Target frames  : {args.target_frames}")
    print(f"COLMAP         : {'SKIPPED' if args.skip_colmap else args.colmap}")
    print("=" * 60)

    images_dir = os.path.join(scene_dir, "image")
    traj_path  = os.path.join(scene_dir, "trajectory.log")
    if not os.path.isdir(images_dir):
        sys.exit(f"ERROR: image/ not found in {scene_dir}")
    if not os.path.exists(traj_path):
        sys.exit(f"ERROR: trajectory.log not found in {scene_dir}")

    # 1. Select frames
    selected_fids = select_frames(images_dir, args.target_frames)

    # 2. Load trajectory + match
    print("\nStep 2: Reading trajectory.log ...")
    poses = read_trajectory(traj_path)
    print(f"  Poses loaded  : {len(poses):,}")
    matched   = [fid for fid in selected_fids if fid in poses]
    unmatched = [fid for fid in selected_fids if fid not in poses]
    if unmatched:
        print(f"  WARNING       : {len(unmatched)} frames have no pose, skipping")
    if not matched:
        sys.exit("ERROR: No selected frames matched any pose.")
    print(f"  Final count   : {len(matched)} frames")

    # 3. Copy images + depth
    print("\nStep 3: Copying selected frames ...")
        # ── batch or single output ───────────────────────────────────────────────

    if args.batch_size > 0:
        batches = batch_frames(matched, args.batch_size)
        print(f"\nBatching: {len(matched)} frames → {len(batches)} batches of ≤{args.batch_size}")
    else:
        batches = [matched]   # single "batch" = original behaviour

    for bi, batch_fids in enumerate(batches):
        if args.batch_size > 0:
            batch_dir = os.path.join(output_dir, f"batch_{bi:02d}")
            print(f"\n{'='*40}\nBatch {bi:02d}: {len(batch_fids)} frames → {batch_dir}")
        else:
            batch_dir = output_dir

        out_img_dir   = os.path.join(batch_dir, "images")
        out_depth_dir = os.path.join(batch_dir, "depth")
        sparse_dir    = os.path.join(batch_dir, "sparse", "0")

        # 3. Copy images + depth
        print(f"\nStep 3: Copying frames ...")
        copy_frames(scene_dir, batch_fids, out_img_dir, out_depth_dir)

        # 4. Write COLMAP pose files
        print(f"\nStep 4: Writing COLMAP files ...")
        os.makedirs(sparse_dir, exist_ok=True)
        write_cameras_bin(os.path.join(sparse_dir, "cameras.bin"))
        print("  OK cameras.bin")
        write_images_bin(os.path.join(sparse_dir, "images.bin"), poses, batch_fids)
        print("  OK images.bin")
        write_empty_points3d_bin(os.path.join(sparse_dir, "points3D.bin"))
        print("  OK points3D.bin (placeholder)")

        # 4.5 Export DA3 poses for this batch
        export_poses(poses, batch_fids, os.path.join(batch_dir, "selected_poses"))

        # 5–7. COLMAP triangulation
        if not args.skip_colmap:
            print(f"\nSteps 5–7: COLMAP triangulation ...")
            run_colmap_triangulation(args.colmap, batch_dir, sparse_dir,
                                     use_gpu=not args.no_gpu)
        else:
            print("\nCOLMAP skipped (--skip_colmap).")

    print(f"\n{'='*60}\nDone! Output: {output_dir}\n{'='*60}")


if __name__ == "__main__":
    main()
