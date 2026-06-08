import open3d as o3d
import numpy as np
import copy
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation
import argparse
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Register two 3DGS PLY files using FPFH + Multiscale ICP")
parser.add_argument("--source", type=str, required=True, help="Path to batch 1 .ply")
parser.add_argument("--target", type=str, required=True, help="Path to batch 2 .ply")
parser.add_argument("--output", type=str, default="merged_output.ply", help="Output merged .ply path")
parser.add_argument("--voxel_size", type=float, default=0.08, help="Voxel size in metres (default 0.002 = 2mm)")
parser.add_argument("--dedup_voxel_size", type=float, default=0.003, help="Voxel size for deduplication in metres (default 0.02 = 2cm)")
parser.add_argument("--skip_global", action="store_true", help="Skip RANSAC global registration (use if poses are already roughly aligned)")
args = parser.parse_args()

BATCH1_PLY  = args.source
BATCH2_PLY  = args.target
OUTPUT_PLY  = args.output
VOXEL_SIZE  = args.voxel_size
DEDUP_VOXEL_SIZE = args.dedup_voxel_size  # Use larger voxel size for deduplication


# ── GAUSSIAN PLY HELPERS ──────────────────────────────────────────────────────
def load_gaussian_ply(path):
    print(f"  Loading {os.path.basename(path)}...")
    plydata = PlyData.read(path)
    n = len(plydata['vertex'])
    print(f"    → {n:,} Gaussians")
    return plydata

def ply_to_o3d(plydata):
    """Extract xyz from Gaussian PLY into an Open3D point cloud for registration."""
    v = plydata['vertex']
    xyz = np.stack([np.array(v['x']), np.array(v['y']), np.array(v['z'])], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # Copy colours if available (f_dc_0/1/2 are SH DC coefficients ~ diffuse colour)
    try:
        r = np.array(v['f_dc_0']) * 0.282095 + 0.5  # SH to RGB approx
        g = np.array(v['f_dc_1']) * 0.282095 + 0.5
        b = np.array(v['f_dc_2']) * 0.282095 + 0.5
        colors = np.clip(np.stack([r, g, b], axis=1), 0, 1)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    except Exception:
        pass
    return pcd

def apply_transform_to_gaussians(plydata, T):
    """Apply 4x4 rigid transform T to xyz positions and quaternion rotations in-place."""
    v = plydata['vertex'].data.copy()

    # --- Transform xyz positions ---
    xyz = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float64)
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1))])
    xyz_t = (T @ xyz_h.T).T[:, :3]
    v['x'] = xyz_t[:, 0].astype(np.float32)
    v['y'] = xyz_t[:, 1].astype(np.float32)
    v['z'] = xyz_t[:, 2].astype(np.float32)

    # --- Rotate quaternions ---
    R = T[:3, :3]
    r_matrix = Rotation.from_matrix(R)
    # 3DGS stores quaternion as (w, x, y, z) in rot_0..3
    rot_wxyz = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=1).astype(np.float64)
    rot_xyzw = rot_wxyz[:, [1, 2, 3, 0]]  # scipy uses (x,y,z,w)
    rots = Rotation.from_quat(rot_xyzw)
    rots_transformed = r_matrix * rots
    rot_xyzw_new = rots_transformed.as_quat()
    rot_wxyz_new = rot_xyzw_new[:, [3, 0, 1, 2]]  # back to (w,x,y,z)
    v['rot_0'] = rot_wxyz_new[:, 0].astype(np.float32)
    v['rot_1'] = rot_wxyz_new[:, 1].astype(np.float32)
    v['rot_2'] = rot_wxyz_new[:, 2].astype(np.float32)
    v['rot_3'] = rot_wxyz_new[:, 3].astype(np.float32)

    return PlyData([PlyElement.describe(v, 'vertex')], text=plydata.text)

def merge_gaussian_plys(ply1, ply2):
    v1 = ply1['vertex'].data
    v2 = ply2['vertex'].data
    merged = np.concatenate([v1, v2])
    return PlyData([PlyElement.describe(merged, 'vertex')])


# ── REGISTRATION HELPERS ──────────────────────────────────────────────────────
def preprocess(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return pcd_down, fpfh

def global_registration(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size):
    print("Stage 1: Fast Global Registration (FGR)...")
    print(f"  Downsampled: {len(src_down.points):,} pts")
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=voxel_size * 1.5,
            iteration_number=64,
            maximum_tuple_count=500,
            tuple_scale=0.95,
            decrease_mu=True
        )
    )
    print(f"  Fitness: {result.fitness:.4f}  RMSE: {result.inlier_rmse:.6f}")
    if result.fitness < 0.3:
        print("Low fitness — overlap may be insufficient.")
    return result

def multiscale_icp(source, target, voxel_size, init_T):
    print("Stage 2: Multi-scale Point-to-Plane ICP refinement...")
    scales = [voxel_size, voxel_size / 2, voxel_size / 4]
    iters  = [50, 30, 14]
    current_T = init_T

    for scale, n_iter in zip(scales, iters):
        src_s = source.voxel_down_sample(scale)
        tgt_s = target.voxel_down_sample(scale)
        src_s.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=scale * 2, max_nn=30))
        tgt_s.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=scale * 2, max_nn=30))

        result = o3d.pipelines.registration.registration_icp(
            src_s, tgt_s,
            max_correspondence_distance=scale * 1.5,
            init=current_T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=n_iter)
        )
        current_T = result.transformation
        print(f"  Scale {scale:.4f}m  Fitness: {result.fitness:.4f}  RMSE: {result.inlier_rmse:.6f}")

    return current_T, result

# Add this to register_gaussians_final.py after merging, before saving
def voxel_deduplicate(plydata, voxel_size=0.002):
    v = plydata['vertex'].data
    xyz = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
    opacity = 1 / (1 + np.exp(-v['opacity'].astype(np.float32)))

    print(f"  Voxel deduplication: {len(xyz):,} Gaussians")  # sanity check for memory issues before proceeding with deduplication
    # Voxel indices — no flat encoding, use structured key instead
    voxel_idx = np.floor(xyz / voxel_size).astype(np.int32)
    print(f"  Voxels Created: {len(voxel_idx), voxel_idx[0,:], voxel_idx[-1,:]} (voxel size={voxel_size}m)")
    # Sort by (ix, iy, iz, -opacity) — groups same voxel together, best opacity first
    sort_order = np.lexsort((-opacity,
                              voxel_idx[:, 2],
                              voxel_idx[:, 1],
                              voxel_idx[:, 0]))

    sorted_vox = voxel_idx[sort_order]  # (N, 3)
    print(sorted_vox.shape, sorted_vox[0,:], sorted_vox[-1,:])  # sanity check for memory issues after sorting
    # Find first occurrence of each unique (ix, iy, iz) triplet
    diff = np.any(sorted_vox[1:] != sorted_vox[:-1], axis=1)

                                 
    first = np.concatenate([[True], diff])
    keep_indices = np.sort(sort_order[first])

    deduped = v[keep_indices]
    print(f"  Deduplication: {len(v):,} → {len(deduped):,} "
          f"({100*(1-len(deduped)/len(v)):.1f}% removed)")
    return PlyData([PlyElement.describe(deduped, 'vertex')])

# ── MAIN ──────────────────────────────────────────────────────────────────────
print("\n========================================")
print("  3DGS Batch Registration via Open3D")
print("========================================\n")

# 1. Load full Gaussian PLYs
print("── Loading Gaussians ──")
ply_src = load_gaussian_ply(BATCH1_PLY)
ply_tgt = load_gaussian_ply(BATCH2_PLY)

# 2. Convert to Open3D point clouds for registration
print("\n── Preparing point clouds ──") 
src_pcd = ply_to_o3d(ply_src)
tgt_pcd = ply_to_o3d(ply_tgt)
print(f"  Source: {len(src_pcd.points):,} pts  Target: {len(tgt_pcd.points):,} pts")

# 3. Downsample + compute FPFH features
print("\n── Preprocessing (voxel downsample + FPFH) ──")
src_down, src_fpfh = preprocess(src_pcd, VOXEL_SIZE)
tgt_down, tgt_fpfh = preprocess(tgt_pcd, VOXEL_SIZE)
print(f"  Downsampled: {len(src_down.points):,} / {len(tgt_down.points):,} pts at voxel={VOXEL_SIZE}m")

# 4. Global registration
print("\n── Registration ──")
if args.skip_global:
    print("Skipping global registration (--skip_global set) — using identity init.")
    init_T = np.eye(4)
    global_fitness = 1.0
else:
    global_result = global_registration(src_down, tgt_down, src_fpfh, tgt_fpfh, VOXEL_SIZE)
    init_T = global_result.transformation
    global_fitness = global_result.fitness

# 5. Multi-scale ICP refinement
final_T, icp_result = multiscale_icp(src_pcd, tgt_pcd, VOXEL_SIZE, init_T)

print(f"\n── Final Transform ──")
print(np.round(final_T, 6))
# Sanity check the transform
translation = np.linalg.norm(final_T[:3, 3])
rot = Rotation.from_matrix(final_T[:3,:3])
angle_deg = np.degrees(rot.magnitude())
print(f"\n  Transform sanity:")
print(f"  Translation magnitude: {translation:.4f} m")
print(f"  Rotation magnitude:    {angle_deg:.2f} degrees")
if translation > 5.0:
    print("  ⚠️  WARNING: Translation > 5m — likely a bad registration!")
if angle_deg > 45:
    print("  ⚠️  WARNING: Rotation > 45° — likely a bad registration!")

# 6. Apply transform to full Gaussian PLY (xyz + quaternions)
print("\n── Applying transform to Gaussian attributes ──")
ply_src_aligned = apply_transform_to_gaussians(ply_src, final_T)

# 7. Merge
os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PLY)), exist_ok=True)
print("── Merging submaps ──")
merged_ply = merge_gaussian_plys(ply_src_aligned, ply_tgt)
n_merged = len(merged_ply['vertex'])
print(f"  Total Gaussians after merge: {n_merged:,}")
# 8. Deduplicate overlapping Gaussians using voxel grid (keeps highest-opacity Gaussian per voxel)
print("── Voxel Deduplication ──")
merged_ply.write(OUTPUT_PLY.replace('.ply', '_prededup.ply'))
print("  → Saved pre-dedup version for inspection")
de_ply = voxel_deduplicate(merged_ply, DEDUP_VOXEL_SIZE)
n_deduped = len(de_ply['vertex'])
print(f"  Total Gaussians after deduplication: {n_deduped:,}")
# 9. Save
de_ply.write(OUTPUT_PLY)
print(f"\n✅ Saved → {OUTPUT_PLY}")

# 10. Summary
print("\n========================================")
print("  Registration Summary")
print("========================================")
if not args.skip_global:
    print(f"  Global fitness  : {global_fitness:.4f}  (1.0 = perfect)")
print(f"  ICP fitness     : {icp_result.fitness:.4f}  (>0.6 good, >0.8 great)")
print(f"  ICP RMSE        : {icp_result.inlier_rmse:.6f} m")
print(f"  Voxel size used : {VOXEL_SIZE} m")
print(f"  Total Gaussians : {n_deduped:,}")
print(f"  Output          : {OUTPUT_PLY}")
print("========================================\n")
print("Load the output .ply in SuperSplat: https://playcanvas.com/supersplat/editor")
