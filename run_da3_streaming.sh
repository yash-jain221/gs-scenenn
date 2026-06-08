#!/usr/bin/env bash
set -e

# ──────────────────────────────────────────────────────────────────────────────
# run_da3_streaming.sh <SCENE_ID> [--with-loop]
#
# Runs DA3-Streaming on a SceneNN scene using the full raw image sequence.
# No frame selection needed — streaming handles the full video natively.
#
# Usage:
#   ./run_da3_streaming.sh 016
#   ./run_da3_streaming.sh 021
#   ./run_da3_streaming.sh 016 --with-loop   # enable loop closure (needs SALAD weights)
#
# Outputs:
#   output/scenenn/DA3_streaming/<SCENE_ID>/
#     pcd/               per-chunk + combined_pcd.ply  (point cloud)
#     gs_ply/            combined_gs.ply  (3D Gaussian Splat — eval with eval_scenenn_da3.py)
#     _tmp_gs_chunks/    per-chunk unaligned GS PLYs (kept when delete_temp_files: False)
#     camera_poses.txt   all c2w poses (one 4x4 per line, flattened)
#     intrinsic.txt      fx fy cx cy per frame
#     camera_poses.ply   visualisation of camera trajectory
#     loop_closures.txt  detected loop pairs (if loop enabled)
#     sim3_opt_result.png before/after loop closure plot (if loop enabled)
#
# Eval the GS PLY (streaming cameras are in DA3's frame — use pred_extrinsics from
# camera_poses.txt or feed combined_gs.ply directly into SuperSplat for visualisation).
# ──────────────────────────────────────────────────────────────────────────────

if [ -z "$1" ]; then
    echo "Usage: $0 <SCENE_ID> [--with-loop]"
    echo "  e.g. $0 016"
    echo "       $0 021 --with-loop"
    exit 1
fi

SCENE_ID="$1"
WITH_LOOP=false
if [ "$2" = "--with-loop" ]; then
    WITH_LOOP=true
fi

SCRIPT_DIR="/home/yash/gs-scenenn"
STREAMING_DIR="${SCRIPT_DIR}/externals/depth_anything_3/da3_streaming"
IMAGE_DIR="${SCRIPT_DIR}/data/scenenn/raw/${SCENE_ID}/image"
OUTPUT_DIR="${SCRIPT_DIR}/output/scenenn/DA3_streaming/${SCENE_ID}"
CONFIG="${STREAMING_DIR}/configs/scenenn.yaml"
WEIGHTS_DIR="${STREAMING_DIR}/weights"

# ── Validate inputs ───────────────────────────────────────────────────────────
if [ ! -d "${IMAGE_DIR}" ]; then
    echo "ERROR: Image dir not found: ${IMAGE_DIR}"
    echo "       Expected raw SceneNN scene at data/scenenn/raw/${SCENE_ID}/image/"
    exit 1
fi

N_IMAGES=$(ls "${IMAGE_DIR}"/*.png 2>/dev/null | wc -l)
if [ "$N_IMAGES" -eq 0 ]; then
    echo "ERROR: No PNG images found in ${IMAGE_DIR}"
    exit 1
fi
echo "Found ${N_IMAGES} images in ${IMAGE_DIR}"

# ── Weights setup ─────────────────────────────────────────────────────────────
# DA3 NESTED-GIANT-LARGE weights are already in HuggingFace cache.
# Streaming loads them via safetensors directly (not from_pretrained),
# so we symlink them into the expected ./weights/ directory.

HF_SNAP="/home/yash/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7"

mkdir -p "${WEIGHTS_DIR}"

if [ ! -e "${WEIGHTS_DIR}/model.safetensors" ]; then
    ln -sf "${HF_SNAP}/model.safetensors" "${WEIGHTS_DIR}/model.safetensors"
    echo "Symlinked model.safetensors → HF cache"
fi
if [ ! -e "${WEIGHTS_DIR}/config.json" ]; then
    ln -sf "${HF_SNAP}/config.json" "${WEIGHTS_DIR}/config.json"
    echo "Symlinked config.json → HF cache"
fi

# ── Loop closure (optional) ───────────────────────────────────────────────────
if [ "${WITH_LOOP}" = true ]; then
    if [ ! -f "${WEIGHTS_DIR}/dino_salad.ckpt" ]; then
        echo "Downloading SALAD weights (~340 MiB) for loop closure..."
        curl -L "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt" \
             -o "${WEIGHTS_DIR}/dino_salad.ckpt"
    fi
    # Patch config to enable loop closure for this run using a temp copy
    TMP_CONFIG="${STREAMING_DIR}/configs/_scenenn_loop_tmp.yaml"
    sed 's/loop_enable: False/loop_enable: True/' "${CONFIG}" > "${TMP_CONFIG}"
    CONFIG="${TMP_CONFIG}"
    echo "Loop closure ENABLED (SALAD loaded)"
else
    echo "Loop closure DISABLED (pass --with-loop to enable, requires SALAD download)"
fi

# ── Skip if output already complete ──────────────────────────────────────────
COMBINED_PLY="${OUTPUT_DIR}/pcd/combined_pcd.ply"
if [ -f "${COMBINED_PLY}" ]; then
    echo "[SKIP] Output already exists: ${COMBINED_PLY}"
    echo "       Delete ${OUTPUT_DIR} to re-run."
    exit 0
fi

mkdir -p "${OUTPUT_DIR}"

# ── Run ───────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  DA3-Streaming  |  Scene: ${SCENE_ID}"
echo "════════════════════════════════════════════════════════"
echo "  Images     : ${IMAGE_DIR}  (${N_IMAGES} frames)"
echo "  Output     : ${OUTPUT_DIR}"
echo "  Config     : ${CONFIG}"
echo "  Loop       : ${WITH_LOOP}"
echo "════════════════════════════════════════════════════════"
echo ""

# Must run from inside da3_streaming/ because da3_streaming.py imports
# loop_utils and other modules using relative paths.
# depth_anything_3 package lives in externals/depth_anything_3/src/ — add to PYTHONPATH.
cd "${STREAMING_DIR}"
export PYTHONPATH="${SCRIPT_DIR}/externals/depth_anything_3/src:${PYTHONPATH}"

python da3_streaming.py \
    --image_dir  "${IMAGE_DIR}" \
    --config     "${CONFIG}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Done!  Scene: ${SCENE_ID}"
echo "  Point cloud : ${OUTPUT_DIR}/pcd/combined_pcd.ply"
echo "  GS PLY      : ${OUTPUT_DIR}/gs_ply/combined_gs.ply"
echo "  Camera poses: ${OUTPUT_DIR}/camera_poses.txt"
echo "════════════════════════════════════════════════════════"