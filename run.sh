if [ -z "$1" ]; then
	echo "Usage: $0 <scene_id>"
	exit 1
fi

SCENE_ID="$1"

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

run_step_if_missing "download" "data/scenenn/raw/oni/${SCENE_ID}.oni" \
	python download_scenenn.py "$SCENE_ID"

run_step_if_missing_dir "playback" "data/scenenn/raw/${SCENE_ID}" \
	./externals/scenenn/playback/playback "data/scenenn/raw/oni/${SCENE_ID}.oni" "data/scenenn/raw/${SCENE_ID}/"

run_step_if_missing_dir "convert_colmap" "data/scenenn/colmap/${SCENE_ID}/sparse" \
	python convert_scenenn_to_colmap.py

run_step_if_missing "train" "output/scenenn/${SCENE_ID}/point_cloud/iteration_30000/point_cloud.ply" \
	python externals/gaussian-splatting/train.py -s "data/scenenn/colmap/${SCENE_ID}"   -m "output/scenenn/${SCENE_ID}"   --eval   --iterations 30000   --test_iterations 15000 30000   --save_iterations 7000 15000 30000

if has_render_outputs "output/scenenn/${SCENE_ID}"; then
	echo "Skipping render (already done)"
else
	echo "Running render..."
	python externals/gaussian-splatting/render.py -m "output/scenenn/${SCENE_ID}" -s "data/scenenn/colmap/${SCENE_ID}"
fi

run_step_if_missing "metrics" "output/scenenn/${SCENE_ID}/results.json" \
	python externals/gaussian-splatting/metrics.py -m "output/scenenn/${SCENE_ID}"