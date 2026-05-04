#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--list scenes.txt] [--download-workers N] [--playback-workers N] [--convert-workers N] [--train-workers N] [--no-train] <scene_id>..."
}

has_any() {
    local path="$1"
    [ -d "$path" ] && [ "$(find "$path" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" != "" ]
}

has_render_outputs() {
    local model_path="$1"
    find "$model_path/test" -type f -name "*.png" -path "*/ours_*/renders/*" 2>/dev/null | head -n 1 | grep -q .
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

run_with_limit() {
    local max="$1"
    shift
    while [ "$(jobs -pr | wc -l)" -ge "$max" ]; do
        sleep 1
    done
    "$@" &
}

run_stage() {
    local stage_name="$1"
    local max_workers="$2"
    local func_name="$3"

    echo "== Stage: ${stage_name} (workers=${max_workers}) =="
    for sid in "${SCENES[@]}"; do
        run_with_limit "$max_workers" "$func_name" "$sid"
    done
    wait
}

is_numeric() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

DOWNLOAD_WORKERS=8
PLAYBACK_WORKERS=3
CONVERT_WORKERS=2
TRAIN_WORKERS=1
DO_TRAIN=1
LIST_FILE=""
SCENES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list)
            LIST_FILE="$2"
            shift 2
            ;;
        --download-workers)
            DOWNLOAD_WORKERS="$2"
            shift 2
            ;;
        --playback-workers)
            PLAYBACK_WORKERS="$2"
            shift 2
            ;;
        --convert-workers)
            CONVERT_WORKERS="$2"
            shift 2
            ;;
        --train-workers)
            TRAIN_WORKERS="$2"
            shift 2
            ;;
        --no-train)
            DO_TRAIN=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            SCENES+=("$1")
            shift
            ;;
    esac
done

if [ -n "$LIST_FILE" ]; then
    while IFS= read -r line; do
        line="${line%%#*}"
        line="${line%/}"
        line="${line//$'\r'/}"
        line="$(echo "$line" | xargs)"
        [ -z "$line" ] && continue
        SCENES+=("$line")
    done < "$LIST_FILE"
fi

if [ ${#SCENES[@]} -eq 0 ]; then
    usage
    exit 1
fi

SANITIZED_SCENES=()
for sid in "${SCENES[@]}"; do
    sid="${sid%/}"
    sid="${sid//$'\r'/}"
    sid="$(echo "$sid" | xargs)"
    if is_numeric "$sid"; then
        SANITIZED_SCENES+=("$sid")
    else
        echo "Skipping non-numeric scene id: $sid"
    fi
done

SCENES=("${SANITIZED_SCENES[@]}")
if [ ${#SCENES[@]} -eq 0 ]; then
    echo "No valid scene ids provided"
    exit 1
fi

run_download() {
    local sid="$1"
    run_step_if_missing "download [$sid]" "data/scenenn/raw/oni/${sid}.oni" \
        python download_scenenn_parallel.py "$sid"
}

run_playback() {
    local sid="$1"
    if [ ! -f "data/scenenn/raw/oni/${sid}.oni" ]; then
        echo "Skipping playback [$sid] (missing oni)"
        return 0
    fi
    run_step_if_missing_dir "playback [$sid]" "data/scenenn/raw/${sid}/image" \
        ./externals/scenenn/playback/playback "data/scenenn/raw/oni/${sid}.oni" "data/scenenn/raw/${sid}/"
}

run_convert() {
    local sid="$1"
    if [ ! -d "data/scenenn/raw/${sid}" ]; then
        echo "Skipping convert_colmap [$sid] (missing raw)"
        return 0
    fi
    run_step_if_missing_dir "convert_colmap [$sid]" "data/scenenn/colmap/${sid}/sparse" \
        python convert_scenenn_to_colmap_parallel.py --scene "$sid"
}

run_train() {
    local sid="$1"
    run_step_if_missing "train [$sid]" "output/scenenn/${sid}/point_cloud/iteration_30000/point_cloud.ply" \
        python externals/gaussian-splatting/train.py -s "data/scenenn/colmap/${sid}" \
            -m "output/scenenn/${sid}" \
            --eval \
            --iterations 30000 \
            --test_iterations 15000 30000 \
            --save_iterations 7000 15000 30000

    if has_render_outputs "output/scenenn/${sid}"; then
        echo "Skipping render [$sid] (already done)"
    else
        echo "Running render [$sid]..."
        python externals/gaussian-splatting/render.py -m "output/scenenn/${sid}" -s "data/scenenn/colmap/${sid}"
    fi

    run_step_if_missing "metrics [$sid]" "output/scenenn/${sid}/results.json" \
        python externals/gaussian-splatting/metrics.py -m "output/scenenn/${sid}"
}

run_stage "download" "$DOWNLOAD_WORKERS" run_download
run_stage "playback" "$PLAYBACK_WORKERS" run_playback
run_stage "convert" "$CONVERT_WORKERS" run_convert

if [ "$DO_TRAIN" -eq 1 ]; then
    run_stage "train" "$TRAIN_WORKERS" run_train
fi
