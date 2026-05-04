import argparse
import math
import os
import shutil
import struct
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# ── SETTINGS ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_ROOT = os.path.join(SCRIPT_DIR, "data/scenenn/raw")
OUTPUT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "data", "scenenn", "colmap"))
TARGET_IMAGES = 200
FX, FY = 544.47, 544.47
CX, CY = 320.0, 240.0
W, H = 640, 480


def read_trajectory(path):
    poses = {}
    with open(path) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        fid = int(parts[0])
        mat = np.array(
            [
                [float(v) for v in lines[i + 1].split()],
                [float(v) for v in lines[i + 2].split()],
                [float(v) for v in lines[i + 3].split()],
                [float(v) for v in lines[i + 4].split()],
            ]
        )
        poses[fid] = mat
        i += 5
    return poses


def rotmat_to_qvec(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array(
            [
                0.25 / s,
                (R[2, 1] - R[1, 2]) * s,
                (R[0, 2] - R[2, 0]) * s,
                (R[1, 0] - R[0, 1]) * s,
            ]
        )
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array(
            [
                (R[2, 1] - R[1, 2]) / s,
                0.25 * s,
                (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s,
            ]
        )
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array(
            [
                (R[0, 2] - R[2, 0]) / s,
                (R[0, 1] + R[1, 0]) / s,
                0.25 * s,
                (R[1, 2] + R[2, 1]) / s,
            ]
        )
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        return np.array(
            [
                (R[1, 0] - R[0, 1]) / s,
                (R[0, 2] + R[2, 0]) / s,
                (R[1, 2] + R[2, 1]) / s,
                0.25 * s,
            ]
        )


def write_cameras_bin(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<Q", W))
        f.write(struct.pack("<Q", H))
        for p in [FX, FY, CX, CY]:
            f.write(struct.pack("<d", p))


def write_images_bin(path, poses, frame_ids):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(frame_ids)))
        for img_id, fid in enumerate(frame_ids, start=1):
            c2w = poses[fid]
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            qvec = rotmat_to_qvec(R)
            img_name = f"{fid:06d}.png"
            f.write(struct.pack("<i", img_id))
            for q in qvec:
                f.write(struct.pack("<d", q))
            for tv in t:
                f.write(struct.pack("<d", tv))
            f.write(struct.pack("<i", 1))
            f.write(img_name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))


def write_points3d_bin(path, ply_path, n_points=50000):
    import open3d as o3d

    print("Loading mesh for point cloud...")
    mesh = o3d.io.read_triangle_mesh(ply_path)
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    pts = np.asarray(pcd.points)
    cols = (
        (np.asarray(pcd.colors) * 255).astype(np.uint8)
        if len(pcd.colors) > 0
        else np.full((len(pts), 3), 128, dtype=np.uint8)
    )
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for i, (pt, col) in enumerate(zip(pts, cols)):
            f.write(struct.pack("<Q", i + 1))
            for v in pt:
                f.write(struct.pack("<d", v))
            f.write(struct.pack("<BBB", *col))
            f.write(struct.pack("<d", 0.0))
            f.write(struct.pack("<Q", 0))
    print(f"  Wrote {len(pts)} points")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def parse_fid(fname):
    stem = os.path.splitext(fname)[0]
    digits = "".join(filter(str.isdigit, stem))
    return int(digits)


def find_datasets(root_dir):
    if not os.path.isdir(root_dir):
        return []
    return sorted(
        d
        for d in os.listdir(root_dir)
        if d.isdigit() and os.path.isdir(os.path.join(root_dir, d))
    )


def process_dataset(raw_root, output_root, name, target_images):
    dataset_dir = os.path.join(raw_root, name)
    traj_path = os.path.join(dataset_dir, "trajectory.log")
    ply_path = os.path.join(dataset_dir, f"{name}.ply")
    images_dir = os.path.join(dataset_dir, "image")
    output_dir = os.path.join(output_root, name)
    out_img_dir = os.path.join(output_dir, "images")
    sparse_dir = os.path.join(output_dir, "sparse", "0")
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
        for p in missing:
            print(f"  - {p}")
        return

    print(f"\n=== Processing {name} ===")
    print("Step 1: Reading trajectory...")
    all_poses = read_trajectory(traj_path)
    print(f"  Found {len(all_poses)} poses in trajectory")
    print(f"  Frame ID range: {min(all_poses.keys())} -> {max(all_poses.keys())}")

    print("\nStep 2: Reading image files...")
    all_images = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
    print(f"  Found {len(all_images)} images")
    print(f"  First image: {all_images[0]}")
    print(f"  Last image:  {all_images[-1]}")

    subsample = max(1, math.ceil(len(all_images) / target_images))
    print("\nStep 3: Matching images to trajectory poses...")
    print(f"  Using subsample = {subsample} to target ~{target_images} images")
    kept = []
    skipped = 0
    for i, fname in enumerate(all_images):
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
        print("   Trajectory IDs sample:", list(all_poses.keys())[:5])
        print("   Image fids sample:", [parse_fid(f) for f in all_images[:5]])
        return

    print("\nStep 4: Copying subsampled images to output folder...")
    out_img_dir = os.path.join(output_dir, "images")
    os.makedirs(out_img_dir, exist_ok=True)
    for fid, fname in kept:
        src = os.path.join(images_dir, fname)
        dst = os.path.join(out_img_dir, f"{fid:06d}.png")
        shutil.copy(src, dst)
    print(f"  Copied {len(kept)} images -> {out_img_dir}")

    print("\nStep 5: Writing COLMAP binary files...")
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)

    fid_list = [fid for fid, _ in kept]

    write_cameras_bin(os.path.join(sparse_dir, "cameras.bin"))
    print("  OK cameras.bin")

    write_images_bin(os.path.join(sparse_dir, "images.bin"), all_poses, fid_list)
    print("  OK images.bin")

    write_points3d_bin(os.path.join(sparse_dir, "points3D.bin"), ply_path)
    print("  OK points3D.bin")

    print(f" Done! Dataset ready at: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SceneNN to COLMAP (parallel).")
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Scene id to process (can be repeated)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Max parallel scene conversions (default: 2)",
    )
    parser.add_argument(
        "--target-images",
        type=int,
        default=TARGET_IMAGES,
        help="Target number of images per scene (default: 200)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.scene:
        datasets = sorted([d for d in args.scene if d.isdigit()])
    else:
        datasets = find_datasets(RAW_DATA_ROOT)

    if not datasets:
        print(f"No numeric datasets found in: {RAW_DATA_ROOT}")
        return 1

    print(f"Found {len(datasets)} dataset(s): {', '.join(datasets)}")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_dataset, RAW_DATA_ROOT, OUTPUT_ROOT, name, args.target_images
            ): name
            for name in datasets
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Scene {name} failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
