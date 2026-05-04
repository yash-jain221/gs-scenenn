import argparse
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "https://hkust-vgd.ust.hk/scenenn/main"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        print(f"  Failed: {url} ({exc})")


def download_scene(scene_id: str, root: Path) -> None:
    scene_dir = root / "data" / "scenenn" / "raw" / scene_id
    oni_dir = root / "data" / "scenenn" / "raw" / "oni"

    urls = [
        (f"{BASE_URL}/oni/{scene_id}.oni", oni_dir / f"{scene_id}.oni"),
        (f"{BASE_URL}/{scene_id}/trajectory.log", scene_dir / "trajectory.log"),
        (f"{BASE_URL}/{scene_id}/{scene_id}.ply", scene_dir / f"{scene_id}.ply"),
    ]

    for url, dest in urls:
        download(url, dest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SceneNN .oni, trajectory, and ply files (parallel scenes)."
    )
    parser.add_argument(
        "scene_ids",
        nargs="+",
        help="Scene IDs like 005 011 014",
    )
    parser.add_argument(
        "--root",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Root folder containing raw_data (default: script dir)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Max parallel scene downloads (default: 8)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()

    scene_ids = [sid for sid in args.scene_ids if sid.isdigit()]
    skipped = [sid for sid in args.scene_ids if not sid.isdigit()]
    for sid in skipped:
        print(f"Skipping non-numeric scene id: {sid}")

    if not scene_ids:
        print("No valid scene ids provided")
        return 1

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_scene, sid, root): sid for sid in scene_ids}
        for future in as_completed(futures):
            sid = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Scene {sid} failed: {exc}")

    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
