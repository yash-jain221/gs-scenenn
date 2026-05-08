import numpy as np
import struct
import os
import shutil
import math
import argparse
import cv2
import subprocess


# ── CLI ARGS ─────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Convert SceneNN to COLMAP format")
    parser.add_argument("--scene-id", type=str, default=None,
                        help="Process only this scene id (e.g. 021)")
    parser.add_argument("--output-name", type=str, default=None,
                        help="Override output folder name (e.g. 021_200)")
    parser.add_argument("--target-images", type=int, default=400,
                        help="Target number of images to subsample to (default: 400)")

    blur_group = parser.add_mutually_exclusive_group()
    blur_group.add_argument("--blur-fixed", type=float, default=None,
                            metavar="THRESHOLD",
                            help="Drop frames with Laplacian variance below this fixed threshold (e.g. 80)")
    blur_group.add_argument("--blur-percentile", type=float, default=None,
                            metavar="PCT",
                            help="Drop the bottom PCT%% of frames by blur score per scene (e.g. 10)")

    parser.add_argument("--blur-window", type=int, default=7,
                        help="Consecutive-segment window size N for blur removal (default: 7, from GS-Blur NeurIPS 2024)")
    parser.add_argument("--blur-vote-fraction", type=float, default=0.5,
                        help="Fraction of window that must be blurry to drop whole segment (default: 0.5, from Open-Sora 2.0)")

    overexp_group = parser.add_mutually_exclusive_group()
    overexp_group.add_argument("--overexp-fixed", type=float, default=None,
                               metavar="FRACTION",
                               help="Drop frames with saturated-pixel fraction above this threshold (e.g. 0.02)")
    overexp_group.add_argument("--overexp-percentile", type=float, default=None,
                               metavar="PCT",
                               help="Drop the top PCT%% of frames by overexposure score per scene (e.g. 10)")

    parser.add_argument("--overexp-window", type=int, default=7,
                        help="Consecutive-segment window size N for overexposure removal (default: 7)")
    parser.add_argument("--overexp-vote-fraction", type=float, default=0.5,
                        help="Fraction of window that must be overexposed to drop whole segment (default: 0.5)")
    parser.add_argument(
        "--use-colmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use COLMAP feature extraction/matching and point triangulation",
    )
    parser.add_argument(
        "--colmap-exe",
        type=str,
        default="colmap",
        help="COLMAP executable name or full path",
    )
    return parser.parse_args()


# ── SETTINGS ─────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_ROOT = os.path.join(SCRIPT_DIR, "data/scenenn/raw")
OUTPUT_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, "data", "scenenn", "colmap"))

FX, FY = 544.47, 544.47
CX, CY = 320.0, 240.0
W,  H  = 640, 480


# ── BLUR HELPERS ─────────────────────────────────────────────────────────────
def compute_blur_scores(images_dir, filenames):
    """Compute Laplacian variance blur score for every filename."""
    scores = {}
    for fname in filenames:
        img = cv2.imread(os.path.join(images_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            scores[fname] = 0.0
        else:
            scores[fname] = float(cv2.Laplacian(img, cv2.CV_64F).var())
    return scores


def remove_consecutive_blur_segments(filenames, scores, threshold, window=7, vote_fraction=0.5):
    """
    Mark frames as bad if they belong to a consecutive run where >= vote_fraction
    of a centred window of size `window` has score < threshold.

    Research basis:
      - Window N=7 : GS-Blur (NeurIPS 2024) synthesises blur by averaging 7 consecutive frames.
      - vote_fraction=0.5 : Open-Sora 2.0 majority-vote approach for clip-level blur detection.

    Returns (filtered_filenames, n_dropped).
    """
    n    = len(filenames)
    bad  = [False] * n
    half = window // 2

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_fnames     = filenames[lo:hi]
        blurry_in_window  = sum(1 for f in window_fnames if scores[f] < threshold)
        if blurry_in_window / len(window_fnames) >= vote_fraction:
            bad[i] = True

    kept    = [f for f, b in zip(filenames, bad) if not b]
    dropped = sum(bad)
    return kept, dropped


def apply_blur_filter(images_dir, filenames, args):
    """
    Compute blur scores and filter frames.
    Applies fixed OR percentile threshold (mutually exclusive flags), then
    removes consecutive blurry segments using sliding-window majority vote.
    Returns (filtered_filenames, stats_dict).
    """
    if args.blur_fixed is None and args.blur_percentile is None:
        return filenames, {}

    print("  Computing blur scores...")
    scores = compute_blur_scores(images_dir, filenames)

    if args.blur_percentile is not None:
        all_scores = [scores[f] for f in filenames]
        threshold  = float(np.percentile(all_scores, args.blur_percentile))
        print(f"  Relative threshold (bottom {args.blur_percentile}%): {threshold:.2f}")
    else:
        threshold = args.blur_fixed
        print(f"  Fixed threshold: {threshold:.2f}")

    avg_score            = float(np.mean([scores[f] for f in filenames]))
    below_before         = sum(1 for f in filenames if scores[f] < threshold)
    print(f"  Avg blur score: {avg_score:.2f}  |  Below threshold before seg-removal: {below_before}")

    # Step A: remove consecutive blurry segments
    after_seg, seg_dropped = remove_consecutive_blur_segments(
        filenames, scores, threshold,
        window=args.blur_window,
        vote_fraction=args.blur_vote_fraction
    )

    # Step B: drop any remaining isolated blurry frames that survived
    final            = [f for f in after_seg if scores[f] >= threshold]
    isolated_dropped = len(after_seg) - len(final)

    stats = {
        "avg_blur_score"  : avg_score,
        "threshold_used"  : threshold,
        "segment_dropped" : seg_dropped,
        "isolated_dropped": isolated_dropped,
        "total_dropped"   : seg_dropped + isolated_dropped,
        "remaining"       : len(final),
    }
    return final, stats


def compute_overexp_scores(images_dir, filenames, sat_thresh=250):
    """Compute overexposure score as fraction of near-white pixels per image."""
    scores = {}
    for fname in filenames:
        img = cv2.imread(os.path.join(images_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            scores[fname] = 0.0
        else:
            scores[fname] = float(np.mean(img >= sat_thresh))
    return scores


def remove_consecutive_overexp_segments(filenames, scores, threshold, window=7, vote_fraction=0.5):
    """
    Mark frames as bad if they belong to a consecutive run where >= vote_fraction
    of a centred window of size `window` has score > threshold.

    Returns (filtered_filenames, n_dropped).
    """
    n    = len(filenames)
    bad  = [False] * n
    half = window // 2

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_fnames     = filenames[lo:hi]
        overexp_in_window = sum(1 for f in window_fnames if scores[f] > threshold)
        if overexp_in_window / len(window_fnames) >= vote_fraction:
            bad[i] = True

    kept    = [f for f, b in zip(filenames, bad) if not b]
    dropped = sum(bad)
    return kept, dropped


def apply_overexp_filter(images_dir, filenames, args):
    """
    Compute overexposure scores and filter frames.
    Applies fixed OR percentile threshold (mutually exclusive flags), then
    removes consecutive overexposed segments using sliding-window majority vote.
    Returns (filtered_filenames, stats_dict).
    """
    if args.overexp_fixed is None and args.overexp_percentile is None:
        return filenames, {}

    print("  Computing overexposure scores...")
    scores = compute_overexp_scores(images_dir, filenames)

    if args.overexp_percentile is not None:
        all_scores = [scores[f] for f in filenames]
        threshold  = float(np.percentile(all_scores, 100 - args.overexp_percentile))
        print(f"  Relative threshold (top {args.overexp_percentile}%): {threshold:.4f}")
    else:
        threshold = args.overexp_fixed
        print(f"  Fixed threshold: {threshold:.4f}")

    avg_score            = float(np.mean([scores[f] for f in filenames]))
    above_before         = sum(1 for f in filenames if scores[f] > threshold)
    print(f"  Avg overexp score: {avg_score:.4f}  |  Above threshold before seg-removal: {above_before}")

    after_seg, seg_dropped = remove_consecutive_overexp_segments(
        filenames, scores, threshold,
        window=args.overexp_window,
        vote_fraction=args.overexp_vote_fraction
    )

    final            = [f for f in after_seg if scores[f] <= threshold]
    isolated_dropped = len(after_seg) - len(final)

    stats = {
        "avg_overexp_score" : avg_score,
        "threshold_used"    : threshold,
        "segment_dropped"   : seg_dropped,
        "isolated_dropped"  : isolated_dropped,
        "total_dropped"     : seg_dropped + isolated_dropped,
        "remaining"         : len(final),
    }
    return final, stats


# ── COLMAP I/O ────────────────────────────────────────────────────────────────
def read_trajectory(path):
    poses = {}
    with open(path) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        fid   = int(parts[0])
        mat   = np.array([
            [float(v) for v in lines[i+1].split()],
            [float(v) for v in lines[i+2].split()],
            [float(v) for v in lines[i+3].split()],
            [float(v) for v in lines[i+4].split()],
        ])
        poses[fid] = mat
        i += 5
    return poses


def rotmat_to_qvec(R):
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


def write_cameras_bin(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<i", 1))   # PINHOLE
        f.write(struct.pack("<Q", W))
        f.write(struct.pack("<Q", H))
        for p in [FX, FY, CX, CY]:
            f.write(struct.pack("<d", p))


def write_images_bin(path, poses, frame_ids):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(frame_ids)))
        for img_id, fid in enumerate(frame_ids, start=1):
            c2w      = poses[fid]
            w2c      = np.linalg.inv(c2w)
            R        = w2c[:3, :3]
            t        = w2c[:3, 3]
            qvec     = rotmat_to_qvec(R)
            img_name = f"{fid:06d}.png"
            f.write(struct.pack("<i", img_id))
            for q  in qvec: f.write(struct.pack("<d", q))
            for tv in t:    f.write(struct.pack("<d", tv))
            f.write(struct.pack("<i", 1))
            f.write(img_name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))


def write_points3d_bin(path, ply_path, n_points=50000):
    import open3d as o3d
    print("Loading mesh for point cloud...")
    mesh = o3d.io.read_triangle_mesh(ply_path)
    pcd  = mesh.sample_points_uniformly(number_of_points=n_points)
    pts  = np.asarray(pcd.points)
    cols = (np.asarray(pcd.colors) * 255).astype(np.uint8) if len(pcd.colors) > 0 \
           else np.full((len(pts), 3), 128, dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for i, (pt, col) in enumerate(zip(pts, cols)):
            f.write(struct.pack("<Q", i + 1))
            for v in pt: f.write(struct.pack("<d", v))
            f.write(struct.pack("<BBB", *col))
            f.write(struct.pack("<d", 0.0))
            f.write(struct.pack("<Q", 0))
    print(f"  Wrote {len(pts)} points")


def run_colmap(cmd_args):
    print("  COLMAP:", " ".join(cmd_args))
    subprocess.run(cmd_args, check=True)


def write_empty_points3d_bin(path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 0))


# ── UTILITIES ─────────────────────────────────────────────────────────────────
def parse_fid(fname):
    stem   = os.path.splitext(fname)[0]
    digits = ''.join(filter(str.isdigit, stem))
    return int(digits)


def find_datasets(root_dir):
    if not os.path.isdir(root_dir):
        return []
    return sorted(
        d for d in os.listdir(root_dir)
        if d.isdigit() and os.path.isdir(os.path.join(root_dir, d))
    )


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────
def process_dataset(raw_root, output_root, name, args):
    dataset_dir = os.path.join(raw_root, name)
    traj_path   = os.path.join(dataset_dir, "trajectory.log")
    ply_path    = os.path.join(dataset_dir, f"{name}.ply")
    images_dir  = os.path.join(dataset_dir, "image")
    output_name = args.output_name if args.output_name else name
    output_dir  = os.path.join(output_root, output_name)
    out_img_dir = os.path.join(output_dir, "images")
    sparse_dir  = os.path.join(output_dir, "sparse", "0")

    expected_bins = [
        os.path.join(sparse_dir, "cameras.bin"),
        os.path.join(sparse_dir, "images.bin"),
        os.path.join(sparse_dir, "points3D.bin"),
    ]
    if os.path.isdir(out_img_dir) and os.path.isdir(sparse_dir) and all(
        os.path.isfile(p) for p in expected_bins
    ):
        print(f"\nSkipping {name}: already processed")
        return

    missing = [p for p in [traj_path, ply_path, images_dir] if not os.path.exists(p)]
    if missing:
        print(f"\nSkipping {name}: missing files")
        for p in missing: print(f"  - {p}")
        return

    print(f"\n=== Processing {name} ===")

    # Step 1 — trajectory
    print("Step 1: Reading trajectory...")
    all_poses = read_trajectory(traj_path)
    print(f"  Found {len(all_poses)} poses")
    print(f"  Frame ID range: {min(all_poses.keys())} -> {max(all_poses.keys())}")

    # Step 2 — image list
    print("\nStep 2: Reading image files...")
    all_images = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
    print(f"  Found {len(all_images)} images  |  First: {all_images[0]}  Last: {all_images[-1]}")

    # Step 3 — blur filtering (runs on ALL images BEFORE subsampling)
    use_blur = args.blur_fixed is not None or args.blur_percentile is not None
    if use_blur:
        mode = (f"fixed threshold={args.blur_fixed}"
                if args.blur_fixed is not None
                else f"bottom {args.blur_percentile}% percentile")
        print(f"\nStep 3: Blur filtering [{mode}]  "
              f"[window={args.blur_window}, vote_fraction={args.blur_vote_fraction}]")
        sharp_images, blur_stats = apply_blur_filter(images_dir, all_images, args)
        print(f"  Segment-dropped  : {blur_stats['segment_dropped']}")
        print(f"  Isolated-dropped : {blur_stats['isolated_dropped']}")
        print(f"  Total dropped    : {blur_stats['total_dropped']}  "
              f"({blur_stats['total_dropped'] / len(all_images) * 100:.1f}%)")
        print(f"  Remaining sharp  : {blur_stats['remaining']}")
    else:
        sharp_images = all_images
        print("\nStep 3: Blur filtering SKIPPED (pass --blur-fixed or --blur-percentile to enable)")

    # Step 4 — overexposure filtering (after blur filtering)
    use_overexp = args.overexp_fixed is not None or args.overexp_percentile is not None
    if use_overexp:
        mode = (f"fixed threshold={args.overexp_fixed}"
                if args.overexp_fixed is not None
                else f"top {args.overexp_percentile}% percentile")
        print(f"\nStep 4: Overexposure filtering [{mode}]  "
              f"[window={args.overexp_window}, vote_fraction={args.overexp_vote_fraction}]")
        expo_images, overexp_stats = apply_overexp_filter(images_dir, sharp_images, args)
        print(f"  Segment-dropped  : {overexp_stats['segment_dropped']}")
        print(f"  Isolated-dropped : {overexp_stats['isolated_dropped']}")
        print(f"  Total dropped    : {overexp_stats['total_dropped']}  "
              f"({overexp_stats['total_dropped'] / len(sharp_images) * 100:.1f}%)")
        print(f"  Remaining        : {overexp_stats['remaining']}")
    else:
        expo_images = sharp_images
        print("\nStep 4: Overexposure filtering SKIPPED (pass --overexp-fixed or --overexp-percentile to enable)")

    # Step 5 — subsample from the filtered pool
    subsample = max(1, math.ceil(len(expo_images) / args.target_images))
    print(f"\nStep 5: Subsampling {len(expo_images)} frames "
          f"-> ~{args.target_images} (subsample every {subsample})")
    kept    = []
    skipped = 0
    for i, fname in enumerate(expo_images):
        if i % subsample != 0:
            continue
        fid = parse_fid(fname)
        if fid in all_poses:
            kept.append((fid, fname))
        else:
            skipped += 1
    print(f"  Keeping {len(kept)} frames  (skipped {skipped} with no matching pose)")

    if len(kept) == 0:
        print("\nERROR: No frames matched! Check trajectory frame IDs vs image filenames.")
        print("  Trajectory IDs sample:", list(all_poses.keys())[:5])
        print("  Image fids sample    :", [parse_fid(f) for f in sharp_images[:5]])
        return

    # Step 6 — copy images to output
    print("\nStep 6: Copying images to output folder...")
    os.makedirs(out_img_dir, exist_ok=True)
    for fid, fname in kept:
        src = os.path.join(images_dir, fname)
        dst = os.path.join(out_img_dir, f"{fid:06d}.png")
        shutil.copy(src, dst)
    print(f"  Copied {len(kept)} images -> {out_img_dir}")

    # Step 7 — write COLMAP binaries
    print("\nStep 7: Writing COLMAP binary files...")
    os.makedirs(sparse_dir, exist_ok=True)
    fid_list = [fid for fid, _ in kept]

    write_cameras_bin(os.path.join(sparse_dir, "cameras.bin"))
    print("  OK cameras.bin")
    write_images_bin(os.path.join(sparse_dir, "images.bin"), all_poses, fid_list)
    print("  OK images.bin")

    if args.use_colmap:
        print("  Running COLMAP feature extraction, matching, and triangulation...")
        db_path = os.path.join(output_dir, "database.db")
        if os.path.exists(db_path):
            os.remove(db_path)

        points3d_path = os.path.join(sparse_dir, "points3D.bin")
        if not os.path.exists(points3d_path):
            write_empty_points3d_bin(points3d_path)

        run_colmap([
            args.colmap_exe,
            "feature_extractor",
            "--database_path", db_path,
            "--image_path", out_img_dir,
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "PINHOLE",
            "--ImageReader.camera_params", f"{FX},{FY},{CX},{CY}",
        ])
        run_colmap([
            args.colmap_exe,
            "exhaustive_matcher",
            "--database_path", db_path,
        ])

        tri_dir = os.path.join(output_dir, "sparse", "0_triangulated")
        if os.path.isdir(tri_dir):
            shutil.rmtree(tri_dir)
        os.makedirs(tri_dir, exist_ok=True)
        run_colmap([
            args.colmap_exe,
            "point_triangulator",
            "--database_path", db_path,
            "--image_path", out_img_dir,
            "--input_path", sparse_dir,
            "--output_path", tri_dir,
        ])

        shutil.rmtree(sparse_dir)
        shutil.move(tri_dir, sparse_dir)
        print("  OK points3D.bin (COLMAP triangulation)")
    else:
        write_points3d_bin(os.path.join(sparse_dir, "points3D.bin"), ply_path)
        print("  OK points3D.bin (synthetic)")

    print(f"\n Done! Dataset ready at: {output_dir}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    if args.output_name and not args.scene_id:
        print("--output-name requires --scene-id to avoid collisions")
        raise SystemExit(1)
    print("Scanning for datasets...")
    if args.scene_id:
        datasets = [args.scene_id]
    else:
        datasets = find_datasets(RAW_DATA_ROOT)
    if not datasets:
        print(f"No numeric datasets found in: {RAW_DATA_ROOT}")
        raise SystemExit(1)
    print(f"Found {len(datasets)} dataset(s): {', '.join(datasets)}")
    for name in datasets:
        process_dataset(RAW_DATA_ROOT, OUTPUT_ROOT, name, args)
