import argparse
import os
import sys
import urllib.request
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
        (f"{BASE_URL}/{scene_id}/{scene_id}.ply", scene_dir / f"{scene_id}.ply")
    ]

    for url, dest in urls:
        download(url, dest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SceneNN .oni, trajectory, and ply files."
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()

    for scene_id in args.scene_ids:
        if not scene_id.isdigit():
            print(f"Skipping non-numeric scene id: {scene_id}")
            continue
        download_scene(scene_id, root)

    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
