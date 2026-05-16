#!/usr/bin/env bash
set -e

# ──────────────────────────────────────────────────────────────────────────────
# Usage:
#   ./run.sh <SCENE_ID> [OPTIONS]
#
# Blur filtering options (mutually exclusive — pick one):
#   --blur-fixed <N>            Drop frames with Laplacian variance below N
#                               e.g. --blur-fixed 80
#   --blur-percentile <N>       Drop the bottom N% of frames per scene
#                               e.g. --blur-percentile 10
#
# Blur segment options (work with either blur mode):
#   --blur-window <N>           Sliding window size for consecutive detection
#                               Default: 7  (GS-Blur, NeurIPS 2024)
#   --blur-vote-fraction <F>    Fraction of window that must be blurry to drop
#                               the whole segment. Default: 0.5 (Open-Sora 2.0)
#
# Overexposure filtering options (mutually exclusive — pick one):
#   --overexp-fixed <F>          Drop frames with saturated-pixel fraction above F
#                               e.g. --overexp-fixed 0.12
#   --overexp-percentile <N>     Drop the top N% of frames per scene by overexp score
#                               e.g. --overexp-percentile 10
#
# Overexposure segment options (work with either overexp mode):
#   --overexp-window <N>         Sliding window size for consecutive detection
#                               Default: 7
#   --overexp-vote-fraction <F>  Fraction of window that must be overexposed to drop
#                               the whole segment. Default: 0.5
#
# Other options:
#   --output-name <NAME>        Override output folder name (e.g. 021_200)
#   --target-images <N>         Images to subsample to (default: 400)
#
# Examples:
#   ./run.sh 021                                     # no blur filtering
#   ./run.sh 021 --blur-fixed 80                     # fixed threshold = 80
#   ./run.sh 021 --blur-percentile 10                # drop bottom 10% per scene
#   ./run.sh 021 --blur-fixed 80 --blur-window 5 --blur-vote-fraction 0.6
#   ./run.sh 021 --blur-fixed 80 --target-images 300
#   ./run.sh 021 --output-name 021_200 --blur-percentile 10
#   ./run.sh 021 --overexp-fixed 0.12
#   ./run.sh 021 --overexp-percentile 10 --overexp-window 5 --overexp-vote-fraction 0.6
# ──────────────────────────────────────────────────────────────────────────────

if [ -z "$1" ]; then
    echo "Usage: $0 <SCENE_ID> [OPTIONS]"
    exit 1
fi

SCENE_ID="$1"
shift

OUTPUT_NAME="$SCENE_ID"
PASSTHRU_ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --output-name)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        *)
            PASSTHRU_ARGS+=("$1")
            shift
            ;;
    esac
done

COLMAP_DIR="data/scenenn/colmap/${OUTPUT_NAME}"
OUTPUT_DIR="output/scenenn/${OUTPUT_NAME}"
mkdir -p "$COLMAP_DIR" "$OUTPUT_DIR"

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
has_any() {
    local path="$1"
    [ -d "$path" ] && [ "$(find "$path" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" != "" ]
}

has_render_outputs() {
    local model_path="$1"
    find "$model_path/test" -type f -name "*.png" -path "*/ours_*/renders/*" 2>/dev/null \
        | head -n 1 | grep -q .
}

run_step_if_missing() {
    local name="$1"
    local check_path="$2"
    shift 2
    if [ -e "$check_path" ]; then
        echo "Skipping ${name} (already done)"
        return 0
    fi
    echo "Running ${name}..."
    "$@"
}

run_step_if_missing_dir() {
    local name="$1"
    local check_dir="$2"
    shift 2
    if has_any "$check_dir"; then
        echo "Skipping ${name} (already done)"
        return 0
    fi
    echo "Running ${name}..."
    "$@"
}

find_latest_checkpoint() {
    local model_path="$1"
    local latest=""
    local latest_iter=0
    local f
    for f in "$model_path"/chkpnt*.pth; do
        [ -e "$f" ] || continue
        local base
        base="$(basename "$f")"
        local iter
        iter="${base#chkpnt}"
        iter="${iter%.pth}"
        case "$iter" in
            *[!0-9]*|"") continue ;;
        esac
        if [ "$iter" -gt "$latest_iter" ]; then
            latest_iter="$iter"
            latest="$f"
        fi
    done
    [ -n "$latest" ] && echo "$latest"
    return 0
}

# ── PIPELINE ──────────────────────────────────────────────────────────────────
run_step_if_missing "download" "data/scenenn/raw/oni/${SCENE_ID}.oni" \
    python download_scenenn.py "$SCENE_ID"

run_step_if_missing_dir "playback" "data/scenenn/raw/${SCENE_ID}/image" \
        ./externals/scenenn/playback/playback \
        "data/scenenn/raw/oni/${SCENE_ID}.oni" \
        "data/scenenn/raw/${SCENE_ID}/"

# All remaining CLI args (blur flags etc.) are forwarded to the python script
run_step_if_missing_dir "convert_colmap" "${COLMAP_DIR}/sparse" \
    python convert_scenenn_to_colmap.py --scene-id "${SCENE_ID}" --output-name "${OUTPUT_NAME}" "${PASSTHRU_ARGS[@]}"

START_CHECKPOINT="$(find_latest_checkpoint "${OUTPUT_DIR}")"
START_CHECKPOINT_ARG=""
if [ -n "$START_CHECKPOINT" ]; then
    echo "Resuming training from checkpoint: ${START_CHECKPOINT}"
    START_CHECKPOINT_ARG="--start_checkpoint ${START_CHECKPOINT}"
fi

run_step_if_missing "train" \
    "${OUTPUT_DIR}/point_cloud/iteration_30000/point_cloud.ply" \
    python externals/gaussian-splatting/train.py \
        -s "${COLMAP_DIR}" \
        -m "${OUTPUT_DIR}" \
        --eval \
        --data_device cpu \
        --iterations 30000 \
        --test_iterations 15000 30000 \
        --save_iterations 7000 15000 30000 \
        --checkpoint_iterations 7000 15000 30000 \
        ${START_CHECKPOINT_ARG}

if has_render_outputs "${OUTPUT_DIR}"; then
    echo "Skipping render (already done)"
else
    echo "Running render..."
    python externals/gaussian-splatting/render.py \
        -m "${OUTPUT_DIR}" \
        -s "${COLMAP_DIR}"
fi

run_step_if_missing "metrics" "${OUTPUT_DIR}/results.json" \
    python externals/gaussian-splatting/metrics.py \
        -m "${OUTPUT_DIR}"
