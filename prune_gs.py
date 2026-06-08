import numpy as np
from plyfile import PlyData, PlyElement
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--opacity_threshold", type=float, default=0.05,
                    help="Remove Gaussians with opacity below this (sigmoid space)")
parser.add_argument("--max_scale", type=float, default=0.5,
                    help="Remove Gaussians larger than this (metres) — kills floaters")
args = parser.parse_args()

plydata = PlyData.read(args.input)
v = plydata['vertex'].data

# Opacity is stored pre-sigmoid in 3DGS PLY
opacity_raw = v['opacity']
opacity = 1 / (1 + np.exp(-opacity_raw))  # sigmoid

# Scale is stored as log(scale)
scale_0 = np.exp(v['scale_0'])
scale_1 = np.exp(v['scale_1'])
scale_2 = np.exp(v['scale_2'])
max_scale = np.maximum(scale_0, np.maximum(scale_1, scale_2))

mask = (opacity > args.opacity_threshold) & (max_scale < args.max_scale)

pruned = v[mask]
n_before = len(v)
n_after = len(pruned)
print(f"Pruned: {n_before:,} → {n_after:,} ({100*(1-n_after/n_before):.1f}% removed)")

PlyData([PlyElement.describe(pruned, 'vertex')]).write(args.output)
print(f"Saved → {args.output}")