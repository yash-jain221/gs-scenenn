# SceneNN to 3D Gaussian Splatting Pipeline

This repo contains a simple end-to-end pipeline to:

1. Download a SceneNN scene by ID.
2. Extract frames from the `.oni` file.
3. Convert the dataset to COLMAP format.
4. Train a 3D Gaussian Splatting (3DGS) model.
5. Render the train/test views.
6. Compute metrics (PSNR, SSIM, LPIPS).

The full pipeline is implemented in [run.sh](run.sh).

## Quick Start

Run the pipeline with a SceneNN scene ID:

```bash
./run.sh 005
```

Re-running the script will skip steps that already have expected outputs.

## Pipeline Steps and Outputs

### 1) Download SceneNN data

Command:

```bash
python scenenn/download_scenenn.py <scene_id>
```

Output:

- `scenenn/raw_data/oni/<scene_id>.oni`

### 2) Playback to extract frames

Command:

```bash
./scenenn/playback/playback "scenenn/raw_data/oni/<scene_id>.oni" "scenenn/raw_data/<scene_id>/"
```

Output:

- `scenenn/raw_data/<scene_id>/` (extracted frames and metadata)

### 3) Convert to COLMAP format

Command:

```bash
python scenenn/convert_scenenn_to_colmap.py
```

Output:

- `data/<scene_id>/images/`
- `data/<scene_id>/sparse/`

### 4) Train Gaussian Splatting model

Command:

```bash
python gaussian-splatting/train.py -s "data/<scene_id>" -m "output/<scene_id>" --eval --iterations 30000 --test_iterations 15000 30000 --save_iterations 7000 15000 30000
```

Output:

- `output/<scene_id>/point_cloud/iteration_30000/point_cloud.ply`
- `output/<scene_id>/cfg_args`

### 5) Render train/test views

Command:

```bash
python gaussian-splatting/render.py -m <model_path> -s <data_path>
```

Output:

- `<model_path>/test/ours_*/renders/*.png`
- `<model_path>/test/ours_*/gt/*.png`

### 6) Compute metrics

Command:

```bash
python gaussian-splatting/metrics.py -m <model_path>
```

Output:

- `<model_path>/results.json`
- `<model_path>/per_view.json`

## Notes

- The script uses real output files to decide whether to skip a step. If you change expected outputs, update the checks in [run.sh](run.sh).
- If you want to force a re-run of a step, delete its output files and run the script again.

