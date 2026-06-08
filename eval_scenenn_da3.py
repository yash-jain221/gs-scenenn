#!/usr/bin/env python3
"""
eval_da3_scenenn_v4.py
Evaluate a DA3-produced 3DGS .ply against SceneNN GT images.

Key fixes vs v3:
  1. Normalise SceneNN c2w poses into DA3's world frame
     (DA3 re-anchors world to first camera via _normalize_extrinsics)
  2. Intrinsics rescaled to eval resolution (SceneNN stores 640x480,
     but DA3 processes at 504x378 — eval uses original 640x480 so
     intrinsics are passed as-is from intrinsics.npy which are already
     at 640x480 pixel-space from scenenn_to_colmap.py)
  3. render_gaussians uses gsplat — no manual covariance math

Expected batch structure:
  batch_00/
    images/
      000014.png  ...
    selected_poses/
      extrinsics.npy   # (N, 4, 4) c2w  — SceneNN world frame
      intrinsics.npy   # (N, 3, 3)      — pixel-space at 640x480
      frame_ids.json   # [14, 27, ...]

Usage:
  python eval_da3_scenenn_v4.py \
      --batch_dir  /path/to/016/batch_00 \
      --ply        /path/to/016/batch_00/gs_ply/0000.ply \
      --mode       train \
      --output_dir ./eval/016/batch_00/train \
      --save_renders

Requirements:
  pip install plyfile gsplat torch torchvision pillow numpy tqdm lpips pytorch-msssim
"""

import os
import re
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from gsplat import rasterization
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--batch_dir",           required=True)
parser.add_argument("--ply",                 required=True)
parser.add_argument("--mode",                default="test", choices=["train", "test"])
parser.add_argument("--test_stride",         type=int, default=5)
parser.add_argument("--output_dir",          default="./eval_output")
parser.add_argument("--save_renders",        action="store_true")
parser.add_argument("--width",               type=int, default=640)
parser.add_argument("--height",              type=int, default=480)
parser.add_argument("--pred_extrinsics_path", default=None,
    help="Path to DA3-estimated w2c .npy (N,4,4) for no-poses eval. "
         "When provided, skips GT-pose normalisation and uses these directly.")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nDevice    : {device}")
print(f"Batch dir : {args.batch_dir}")
print(f"PLY       : {args.ply}")
print(f"Mode      : {args.mode}  (stride={args.test_stride})")

# ---------------------------------------------------------------------------
# Load poses
# ---------------------------------------------------------------------------
poses_dir = Path(args.batch_dir) / "selected_poses"
image_dir  = Path(args.batch_dir) / "images"

c2w_all   = np.load(poses_dir / "extrinsics.npy").astype(np.float64)  # (N,4,4) c2w SceneNN frame
ixt_all   = np.load(poses_dir / "intrinsics.npy").astype(np.float64)  # (N,3,3) pixel-space 640x480
with open(poses_dir / "frame_ids.json") as f:
    frame_ids = json.load(f)

N = len(frame_ids)
assert c2w_all.shape[0] == N
assert ixt_all.shape[0] == N

print(f"\nPoses loaded: extrinsics {c2w_all.shape}, intrinsics {ixt_all.shape}")

# ---------------------------------------------------------------------------
# Normalise poses into DA3's world frame.
#
# Two paths:
#   A) no_poses  — DA3 estimated its own poses; --pred_extrinsics_path points
#      to the saved (N,4,4) w2c that matches the PLY's coordinate frame.
#      Use them directly; no further normalisation is needed.
#
#   B) with_gt_poses — GT c2w from SceneNN must be transformed to match DA3's
#      internal _normalize_extrinsics, which does TWO things:
#        1. First-camera-identity:   w2c_norm = w2c @ c2w[0]
#        2. Scale normalisation:     w2c_norm[..., :3, 3] /= median_dist
#             where median_dist = median of camera distances in step-1 frame.
#      The PLY Gaussians live in this fully-normalised world frame.
# ---------------------------------------------------------------------------
if args.pred_extrinsics_path:
    # --- Path A: no_poses ---
    pred_ext = np.load(args.pred_extrinsics_path).astype(np.float64)   # (N,4,4) or (N,3,4)
    if pred_ext.ndim == 3 and pred_ext.shape[1] == 3:
        # pad (N,3,4) → (N,4,4)
        bottom = np.tile(np.array([[0., 0., 0., 1.]]), (pred_ext.shape[0], 1, 1))
        pred_ext = np.concatenate([pred_ext, bottom], axis=1)
    w2c_norm = pred_ext
    print(f"  [no_poses] Using DA3-estimated extrinsics: {args.pred_extrinsics_path}")
    print(f"  Shape: {w2c_norm.shape}  |  first-cam t: {w2c_norm[0, :3, 3]}")
else:
    # --- Path B: with_gt_poses ---
    w2c_all = np.linalg.inv(c2w_all)                   # (N,4,4) SceneNN w2c

    # Step 1: first-camera-identity transform
    transform  = np.linalg.inv(w2c_all[0:1])           # (1,4,4) = c2w[0]
    w2c_step1  = w2c_all @ transform                    # (N,4,4)

    # Step 2: scale normalisation — replicate DA3's median-distance scaling
    c2w_step1    = np.linalg.inv(w2c_step1)             # (N,4,4)
    translations = c2w_step1[..., :3, 3]                # (N,3) camera positions
    dists        = np.linalg.norm(translations, axis=-1)  # (N,)
    median_dist  = float(np.median(dists))
    median_dist  = max(median_dist, 1e-1)
    w2c_norm     = w2c_step1.copy()
    w2c_norm[..., :3, 3] /= median_dist

    print(f"  [with_gt_poses] First-cam w2c translation: {w2c_norm[0, :3, 3]}")
    print(f"  (should be ~[0,0,0])")
    print(f"  Median camera dist (scale factor):         {median_dist:.4f}")

# ---------------------------------------------------------------------------
# Intrinsics are already 640x480 pixel-space from scenenn_to_colmap.py
# (FX=544.47, CX=320, CY=240). Eval renders at W=640, H=480 → no rescale needed.
# If you ever eval at a different resolution, scale fx/cx by (W/640), fy/cy by (H/480).
# ---------------------------------------------------------------------------
print(f"  Intrinsics[0]: fx={ixt_all[0,0,0]:.2f} fy={ixt_all[0,1,1]:.2f} "
      f"cx={ixt_all[0,0,2]:.2f} cy={ixt_all[0,1,2]:.2f}")

# ---------------------------------------------------------------------------
# Build frame index
# ---------------------------------------------------------------------------
def find_image(frame_id, image_dir):
    for f in image_dir.iterdir():
        if f.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        nums = re.findall(r"\d+", f.stem)
        if nums and int(nums[-1]) == frame_id:
            return f
    return None

frame_index = []
missing = 0
for i, fid in enumerate(frame_ids):
    img_path = find_image(fid, image_dir)
    if img_path is None:
        missing += 1
        continue
    frame_index.append((fid, w2c_norm[i], ixt_all[i], img_path))  # w2c_norm, not c2w!

print(f"  Matched images : {len(frame_index)}/{N}  (missing: {missing})")

if args.mode == "test":
    frame_index = frame_index[::args.test_stride]
    print(f"  Test subset    : {len(frame_index)} frames")

if not frame_index:
    raise RuntimeError("No valid frames found.")

# ---------------------------------------------------------------------------
# Load PLY
# ---------------------------------------------------------------------------
from plyfile import PlyData

print("\nLoading PLY...")
plydata = PlyData.read(args.ply)
v = plydata["vertex"]
print(f"  Gaussians: {len(v):,}")

xyz     = torch.tensor(np.stack([v["x"], v["y"], v["z"]], axis=1),
                       dtype=torch.float32, device=device)

opacity = torch.sigmoid(
    torch.tensor(v["opacity"].astype(np.float32), device=device)
).unsqueeze(1)

scales  = torch.exp(
    torch.tensor(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1),
                 dtype=torch.float32, device=device)
)

quats   = F.normalize(
    torch.tensor(np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1),
                 dtype=torch.float32, device=device),
    dim=1
)

# SH DC → linear RGB:  C0 * f_dc + 0.5,  C0 = 0.282095
colors  = torch.stack([
    torch.tensor(v["f_dc_0"].astype(np.float32), device=device) * 0.282095 + 0.5,
    torch.tensor(v["f_dc_1"].astype(np.float32), device=device) * 0.282095 + 0.5,
    torch.tensor(v["f_dc_2"].astype(np.float32), device=device) * 0.282095 + 0.5,
], dim=1).clamp(0, 1)

# ---------------------------------------------------------------------------
# Renderer  (gsplat — w2c already in DA3 frame)
# ---------------------------------------------------------------------------
def render_gaussians(xyz, scales, quats, colors, opacity, w2c, K, W, H):
    dev = xyz.device

    # w2c is already in DA3 frame — no inversion needed
    w2c_t = torch.tensor(w2c, dtype=torch.float32, device=dev).unsqueeze(0)  # (1,4,4)
    K_t   = torch.tensor(K,   dtype=torch.float32, device=dev).unsqueeze(0)  # (1,3,3)

    rendered, _, _ = rasterization(
        means    = xyz,
        quats    = quats,
        scales   = scales,
        opacities= opacity.squeeze(1),
        colors   = colors,
        viewmats = w2c_t,
        Ks       = K_t,
        width    = W,
        height   = H,
        sh_degree= None,
        near_plane=0.1,
        far_plane =100.0,
    )
    return rendered[0].clamp(0, 1)   # (H, W, 3)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_psnr(pred, gt):
    return (-10.0 * torch.log10(F.mse_loss(pred, gt) + 1e-10)).item()

def compute_ssim(pred, gt):
    try:
        from pytorch_msssim import ssim as _ssim
        p = pred.permute(2,0,1).unsqueeze(0)
        g = gt.permute(2,0,1).unsqueeze(0)
        return _ssim(p, g, data_range=1.0).item()
    except ImportError:
        ws = 11
        p = pred.permute(2,0,1).unsqueeze(0)
        g = gt.permute(2,0,1).unsqueeze(0)
        mu1 = F.avg_pool2d(p, ws, 1, ws//2)
        mu2 = F.avg_pool2d(g, ws, 1, ws//2)
        C1, C2 = 0.01**2, 0.03**2
        num = (2*mu1*mu2 + C1) * (2*F.avg_pool2d(p*g, ws,1,ws//2) - 2*mu1*mu2 + C2)
        den = (mu1**2 + mu2**2 + C1) * (F.avg_pool2d(p**2,ws,1,ws//2) +
               F.avg_pool2d(g**2,ws,1,ws//2) - mu1**2 - mu2**2 + C2)
        return (num / den.clamp(min=1e-8)).mean().item()

_lpips_fn = None
def compute_lpips(pred, gt):
    global _lpips_fn
    try:
        import lpips
        if _lpips_fn is None:
            _lpips_fn = lpips.LPIPS(net="vgg").to(device)
        with torch.no_grad():
            p = pred.permute(2,0,1).unsqueeze(0) * 2 - 1
            g = gt.permute(2,0,1).unsqueeze(0) * 2 - 1
            return _lpips_fn(p, g).item()
    except ImportError:
        return None

# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------
to_tensor = T.ToTensor()
W, H = args.width, args.height

if args.save_renders:
    render_dir = Path(args.output_dir) / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

print(f"\nEvaluating {len(frame_index)} frames...\n")

results = {}
psnr_vals, ssim_vals, lpips_vals = [], [], []

for fid, w2c, K, img_path in tqdm(frame_index, desc="Rendering"):
    with torch.no_grad():
        rendered = render_gaussians(xyz, scales, quats, colors, opacity, w2c, K, W, H)

    gt_pil = Image.open(img_path).convert("RGB").resize((W, H), Image.BILINEAR)
    gt     = to_tensor(gt_pil).permute(1, 2, 0).to(device)

    p  = compute_psnr(rendered, gt)
    s  = compute_ssim(rendered, gt)
    lp = compute_lpips(rendered, gt)

    psnr_vals.append(p)
    ssim_vals.append(s)
    if lp is not None:
        lpips_vals.append(lp)

    results[str(fid)] = {"psnr": p, "ssim": s, "lpips": lp}

    if args.save_renders:
        render_np = (rendered.cpu().numpy() * 255).astype(np.uint8)
        gt_np     = np.array(gt_pil)
        side      = np.concatenate([render_np, gt_np], axis=1)
        Image.fromarray(side).save(render_dir / f"frame_{fid:06d}.png")

mean_psnr  = float(np.mean(psnr_vals))
mean_ssim  = float(np.mean(ssim_vals))
mean_lpips = float(np.mean(lpips_vals)) if lpips_vals else None

print("\n" + "="*52)
print(f"Results [{args.mode} | {len(frame_index)} frames]")
print("="*52)
print(f"PSNR  (dB) ↑ : {mean_psnr:.4f}")
print(f"SSIM       ↑ : {mean_ssim:.4f}")
if mean_lpips is not None:
    print(f"LPIPS      ↓ : {mean_lpips:.4f}")
else:
    print("LPIPS        : N/A (install lpips)")
print("="*52 + "\n")

summary = {
    "mode": args.mode, "batch_dir": str(args.batch_dir), "ply": str(args.ply),
    "n_frames": len(frame_index), "mean_psnr": mean_psnr,
    "mean_ssim": mean_ssim, "mean_lpips": mean_lpips, "per_frame": results,
}

out_path = Path(args.output_dir) / "metrics.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Saved metrics → {out_path}")
if args.save_renders:
    print(f"Saved renders → {render_dir}")
