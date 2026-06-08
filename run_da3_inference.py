#!/usr/bin/env python3
"""
run_ablation_poses.py
Run DA3 inference twice per batch:
  1. with_gt_poses   — extrinsics + intrinsics passed explicitly
  2. no_poses        — DA3 estimates everything itself

Outputs PLYs to:
  output_dir/with_gt_poses/batch_XX/gs_ply/0000.ply
  output_dir/no_poses/batch_XX/gs_ply/0000.ply

Usage:
  python run_ablation_poses.py \
      --data_dir   /home/yash/gs-scenenn/data/scenenn/colmap/016_overlap/ \
      --output_dir /home/yash/gs-scenenn/output/scenenn/DA3/016_ablation/
"""

import os
import glob
import time
import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",   required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--batch",      default=None, help="Run single batch only e.g. batch_00")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load DA3 model ──────────────────────────────────────────────────────────
from depth_anything_3.api import DepthAnything3
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").to(device).eval()

# ── Batches ─────────────────────────────────────────────────────────────────
all_batches = sorted(f for f in os.listdir(args.data_dir) if "batch" in f)
if args.batch:
    all_batches = [args.batch]

print(f"Batches to process: {all_batches}\n")

for batch in all_batches:
    batch_path = os.path.join(args.data_dir, batch)
    images     = sorted(glob.glob(os.path.join(batch_path, "images/*.png")))

    ext_np = np.load(os.path.join(batch_path, "selected_poses", "extrinsics.npy"))  # (N,4,4) c2w
    ixt_np = np.load(os.path.join(batch_path, "selected_poses", "intrinsics.npy"))  # (N,3,3)

    c2w    = torch.from_numpy(ext_np).float().to(device)
    w2c_np = torch.linalg.inv(c2w).cpu().numpy()   # (N,4,4) w2c

    print(f"{'='*50}")
    print(f"Batch: {batch}  |  {len(images)} images")

    # ── Run 1: with GT poses ────────────────────────────────────────────────
    out_gt = os.path.join(args.output_dir, "with_gt_poses", batch)
    print(f"\n[1/2] with_gt_poses → {out_gt}")
    with torch.no_grad():
        pred_gt = model.inference(
            images,
            infer_gs=True,
            extrinsics=w2c_np,   # (N,4,4) w2c numpy
            intrinsics=ixt_np,   # (N,3,3) numpy
            render_hw=(480, 640),
            export_dir=out_gt,
            export_format="gs_ply"
        )
    print(f"depth: {pred_gt.depth.shape}  "
          f"ext: {pred_gt.extrinsics.shape}  "
          f"ixt[0]: fx={pred_gt.intrinsics[0,0,0]:.2f} cx={pred_gt.intrinsics[0,0,2]:.2f}")

    # ── Run 2: no poses (DA3 estimates) ────────────────────────────────────
    out_est = os.path.join(args.output_dir, "no_poses", batch)
    print(f"\n[2/2] no_poses → {out_est}")
    with torch.no_grad():
        pred_est = model.inference(
            images,
            infer_gs=True,
            render_hw=(480, 640),
            export_dir=out_est,
            export_format="gs_ply"
        )
    print(f"depth: {pred_est.depth.shape}  "
          f"ext: {pred_est.extrinsics.shape}  "
          f"ixt[0]: fx={pred_est.intrinsics[0,0,0]:.2f} cx={pred_est.intrinsics[0,0,2]:.2f}")

    # Save DA3-estimated w2c extrinsics so eval can render from the correct frame.
    # pred_est.extrinsics is (N,4,4) w2c in DA3's estimated (scale-normalised) world frame.
    np.save(os.path.join(out_est, "pred_extrinsics.npy"), pred_est.extrinsics)
    print(f"  Saved pred_extrinsics.npy  shape={pred_est.extrinsics.shape}")

    print(f"\nDone: {batch}")
    time.sleep(2)

print("\nAll batches done.")
print(f"GT poses PLYs  → {os.path.join(args.output_dir, 'with_gt_poses')}")
print(f"No poses PLYs  → {os.path.join(args.output_dir, 'no_poses')}")
print("\nNext: run eval_da3_scenenn_v4.py on both output dirs to compare metrics.")
