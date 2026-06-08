#!/usr/bin/env bash
set -e

# ──────────────────────────────────────────────────────────────────────────────
# Usage:
#   ./run_da3.sh <EXPERIMENT_NAME>
#
# Examples:
#   ./run_da3.sh 016_overlap
#   ./run_da3.sh 005_batched
#
# Pipeline:
#   1. scenenn_to_colmap.py     → data/scenenn/colmap/<EXPERIMENT>/
#   2. run_ablation_poses.py    → output/scenenn/DA3/<EXPERIMENT>/with_gt_poses/
#                                 output/scenenn/DA3/<EXPERIMENT>/no_poses/
#   3. eval_da3_scenenn_v4.py   → eval/scenenn/DA3/<EXPERIMENT>/with_gt_poses/
#                                 eval/scenenn/DA3/<EXPERIMENT>/no_poses/
#
# Fixed constants:
#   --target-frames 400   --batch-size 60   --batch-overlap 12 (20% of 60)
# ──────────────────────────────────────────────────────────────────────────────

if [ -z "$1" ]; then
    echo "Usage: $0 <EXPERIMENT_NAME>"
    echo "  e.g. $0 016_overlap"
    exit 1
fi

EXPERIMENT="$1"

# ── Paths ────────────────────────────────────────────────────────────────────
SCENE_ID="${EXPERIMENT%%_*}"                              # "016" from "016_overlap"
DATA_DIR="data/scenenn/colmap/${EXPERIMENT}"
OUTPUT_DIR="output/scenenn/DA3/${EXPERIMENT}"
EVAL_DIR="eval/scenenn/DA3/${EXPERIMENT}"

GT_PLY_BASE="${OUTPUT_DIR}/with_gt_poses"
EST_PLY_BASE="${OUTPUT_DIR}/no_poses"
GT_EVAL_BASE="${EVAL_DIR}/with_gt_poses"
EST_EVAL_BASE="${EVAL_DIR}/no_poses"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$EVAL_DIR"

chmod 775 "$DATA_DIR" "$OUTPUT_DIR" "$EVAL_DIR"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  DA3 Pipeline  |  Experiment: ${EXPERIMENT}"
echo "════════════════════════════════════════════════════════"
echo "  Scene ID    : ${SCENE_ID}"
echo "  Data dir    : ${DATA_DIR}"
echo "  Output dir  : ${OUTPUT_DIR}"
echo "  Eval dir    : ${EVAL_DIR}"
echo "════════════════════════════════════════════════════════"
echo ""

# ── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
has_any() {
    local path="$1"
    [ -d "$path" ] && [ "$(find "$path" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" != "" ]
}

run_step_if_missing_dir() {
    local name="$1"
    local check_dir="$2"
    shift 2
    if has_any "$check_dir"; then
        echo "[SKIP] ${name} (already exists: ${check_dir})"
        return 0
    fi
    echo ""
    echo "[RUN ] ${name}"
    "$@"
}

run_step_if_missing_file() {
    local name="$1"
    local check_file="$2"
    shift 2
    if [ -f "$check_file" ]; then
        echo "[SKIP] ${name} (already exists: ${check_file})"
        return 0
    fi
    echo ""
    echo "[RUN ] ${name}"
    "$@"
}

# ── STEP 1: Convert SceneNN → COLMAP batches ─────────────────────────────────
run_step_if_missing_dir "scenenn_to_colmap (${EXPERIMENT})" \
    "${DATA_DIR}/batch_00" \
    python scenenn_to_colmap.py \
        --scene_dir    "data/scenenn/raw/${SCENE_ID}" \
        --output_dir "${DATA_DIR}" \
        --target_frames 400 \
        --batch_size    60 \
        --batch_overlap 12 \
        --skip_colmap

# Discover all batches
BATCHES=( $(ls -d "${DATA_DIR}"/batch_* 2>/dev/null | sort) )
if [ ${#BATCHES[@]} -eq 0 ]; then
    echo "ERROR: No batches found in ${DATA_DIR}"
    exit 1
fi
echo ""
echo "Found ${#BATCHES[@]} batches: $(basename -a "${BATCHES[@]}" | tr '\n' ' ')"

# ── STEP 2: DA3 inference (both conditions) ───────────────────────────────────
# Check if both conditions are already done by looking for first batch PLY
FIRST_BATCH=$(basename "${BATCHES[0]}")
GT_DONE="${GT_PLY_BASE}/${FIRST_BATCH}/gs_ply/0000.ply"
EST_DONE="${EST_PLY_BASE}/${FIRST_BATCH}/gs_ply/0000.ply"

if [ -f "$GT_DONE" ] && [ -f "$EST_DONE" ]; then
    echo "[SKIP] run_ablation_poses (PLYs already exist)"
else
    echo ""
    echo "[RUN ] run_ablation_poses (all batches)"
    python externals/depth_anything_3/src/run_da3_inference.py\
        --data_dir   "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}"
fi

# ── STEP 3: Eval — with_gt_poses ─────────────────────────────────────────────
echo ""
echo "── Evaluating: with_gt_poses ──────────────────────────"
for BATCH_PATH in "${BATCHES[@]}"; do
    BATCH=$(basename "$BATCH_PATH")
    PLY="${GT_PLY_BASE}/${BATCH}/gs_ply/0000.ply"
    EVAL_OUT="${GT_EVAL_BASE}/${BATCH}"

    if [ ! -f "$PLY" ]; then
        echo "[WARN] PLY not found, skipping: ${PLY}"
        continue
    fi

    run_step_if_missing_file \
        "eval with_gt_poses/${BATCH}" \
        "${EVAL_OUT}/metrics.json" \
        python eval_scenenn_da3.py \
            --batch_dir  "${BATCH_PATH}" \
            --ply        "${PLY}" \
            --mode       train \
            --output_dir "${EVAL_OUT}" \
            --save_renders
done

# ── STEP 4: Eval — no_poses ──────────────────────────────────────────────────
echo ""
echo "── Evaluating: no_poses ────────────────────────────────"
for BATCH_PATH in "${BATCHES[@]}"; do
    BATCH=$(basename "$BATCH_PATH")
    PLY="${EST_PLY_BASE}/${BATCH}/gs_ply/0000.ply"
    EVAL_OUT="${EST_EVAL_BASE}/${BATCH}"
    PRED_EXT="${EST_PLY_BASE}/${BATCH}/pred_extrinsics.npy"

    if [ ! -f "$PLY" ]; then
        echo "[WARN] PLY not found, skipping: ${PLY}"
        continue
    fi

    if [ ! -f "$PRED_EXT" ]; then
        echo "[WARN] pred_extrinsics.npy not found, skipping no_poses eval: ${PRED_EXT}"
        echo "       Re-run DA3 inference to generate it."
        continue
    fi

    run_step_if_missing_file \
        "eval no_poses/${BATCH}" \
        "${EVAL_OUT}/metrics.json" \
        python eval_scenenn_da3.py \
            --batch_dir           "${BATCH_PATH}" \
            --ply                 "${PLY}" \
            --mode                train \
            --output_dir          "${EVAL_OUT}" \
            --pred_extrinsics_path "${PRED_EXT}" \
            --save_renders
done

# ── STEP 5: Print summary ─────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Summary: ${EXPERIMENT}"
echo "════════════════════════════════════════════════════════"

summarise() {
    local label="$1"
    local base="$2"
    echo ""
    echo "  [${label}]"
    local total_psnr=0
    local total_ssim=0
    local total_lpips=0
    local count=0
    for BATCH_PATH in "${BATCHES[@]}"; do
        BATCH=$(basename "$BATCH_PATH")
        METRICS="${base}/${BATCH}/metrics.json"
        if [ ! -f "$METRICS" ]; then
            echo "    ${BATCH}: metrics.json not found"
            continue
        fi
        PSNR=$(python3  -c "import json; d=json.load(open('${METRICS}')); print(f\"{d['mean_psnr']:.4f}\")")
        SSIM=$(python3  -c "import json; d=json.load(open('${METRICS}')); print(f\"{d['mean_ssim']:.4f}\")")
        LPIPS=$(python3 -c "import json; d=json.load(open('${METRICS}')); v=d['mean_lpips']; print(f\"{v:.4f}\" if v else 'N/A')")
        echo "    ${BATCH}:  PSNR=${PSNR}  SSIM=${SSIM}  LPIPS=${LPIPS}"
        count=$((count + 1))
    done
    if [ "$count" -gt 0 ]; then
        AVG_PSNR=$(python3 -c "
import json, os, glob
files = sorted(glob.glob('${base}/batch_*/metrics.json'))
vals = [json.load(open(f))['mean_psnr'] for f in files]
print(f'{sum(vals)/len(vals):.4f}') if vals else print('N/A')
")
        AVG_SSIM=$(python3 -c "
import json, glob
files = sorted(glob.glob('${base}/batch_*/metrics.json'))
vals = [json.load(open(f))['mean_ssim'] for f in files]
print(f'{sum(vals)/len(vals):.4f}') if vals else print('N/A')
")
        AVG_LPIPS=$(python3 -c "
import json, glob
files = sorted(glob.glob('${base}/batch_*/metrics.json'))
vals = [json.load(open(f))['mean_lpips'] for f in files if json.load(open(f))['mean_lpips']]
print(f'{sum(vals)/len(vals):.4f}') if vals else print('N/A')
")
        echo "    ── MEAN:  PSNR=${AVG_PSNR}  SSIM=${AVG_SSIM}  LPIPS=${AVG_LPIPS}"
    fi
}

summarise "with_gt_poses" "$GT_EVAL_BASE"
summarise "no_poses     " "$EST_EVAL_BASE"

echo ""
echo "  Eval results  → ${EVAL_DIR}"
echo "  PLYs          → ${OUTPUT_DIR}"
echo "════════════════════════════════════════════════════════"
echo ""
