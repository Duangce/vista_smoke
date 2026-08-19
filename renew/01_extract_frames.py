"""
Extract frames from all *_back.mp4 videos under data/
and prepare a per-sample metadata CSV for Vista fine-tuning.

Output structure:
    renew/frames/<scenario>/<maneuver>/<vehicle>/<clip_name>/<frame_idx>.png
    renew/metadata.csv  (one row per video clip)
    renew/metadata.json
"""
import argparse
import csv
import json
from pathlib import Path

import cv2


DATA_ROOT = Path("/workspace/ws/Vista/data")
OUT_ROOT = Path("/workspace/ws/Vista/renew")
FRAME_ROOT = OUT_ROOT / "frames"
FPS = 30
# Vista training/inference default resolution: 1024x576 (width x height).
# Store frames already at this size so training skips the center-crop/resize step.
RESOLUTION = (1024, 576)  # (width, height)
IMG_EXT = ".png"


def extract_frames(video_path: Path, out_dir: Path, fps: int = FPS, size: tuple = RESOLUTION,
                    img_ext: str = IMG_EXT):
    """Extract frames using OpenCV and save as JPG/PNG.

    Returns:
        (n_frames, (width, height)) tuple.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = (size if size is not None else (None, None))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = fps

    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_size = (width, height) if width is not None and height is not None else (native_w, native_h)

    # Number of source frames to skip between saves
    if src_fps == fps:
        sample_every = 1
    else:
        sample_every = max(1, round(src_fps / fps))

    saved = 0
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % sample_every == 0:
            if width is not None and height is not None:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            # OpenCV reads BGR; write RGB to match training conventions
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out_path = out_dir / f"{saved + 1:04d}{img_ext}"
            cv2.imwrite(str(out_path), frame_rgb)
            saved += 1
        count += 1

    cap.release()
    return saved, out_size


def main(max_clips: int = None):
    FRAME_ROOT.mkdir(parents=True, exist_ok=True)
    records = []

    video_paths = sorted(DATA_ROOT.rglob("*_back.mp4"))
    if max_clips is not None and max_clips > 0:
        video_paths = video_paths[:max_clips]
    total = len(video_paths)
    print(f"Found {total} videos to process")

    for idx, video_path in enumerate(video_paths, start=1):
        rel = video_path.relative_to(DATA_ROOT)
        parts = rel.with_suffix("").parts
        clip_name = parts[-1]
        if clip_name.endswith("_back"):
            clip_name = clip_name[:-5]
        scenario, maneuver, vehicle = parts[:3]

        csv_path = video_path.with_name(f"{clip_name}.csv")
        if not csv_path.exists():
            print(f"[{idx}/{total} SKIP] No matching CSV for {rel}")
            continue

        out_dir = FRAME_ROOT / scenario / maneuver / vehicle / clip_name
        try:
            n_frames, out_size = extract_frames(video_path, out_dir)
        except Exception as e:
            print(f"[{idx}/{total} ERROR] {rel}: {e}")
            continue

        records.append({
            "scenario": scenario,
            "maneuver": maneuver,
            "vehicle": vehicle,
            "clip": clip_name,
            "video_path": str(video_path),
            "csv_path": str(csv_path),
            "frame_dir": str(out_dir),
            "num_frames": int(n_frames),
            "fps": FPS,
            "resolution": str(out_size),
        })
        print(f"[{idx}/{total} OK] {scenario}/{maneuver}/{vehicle}/{clip_name}: {n_frames} frames")

    # Write metadata
    if records:
        metadata_csv = OUT_ROOT / "metadata.csv"
        with open(metadata_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

        metadata_json = OUT_ROOT / "metadata.json"
        with open(metadata_json, "w") as f:
            json.dump(records, f, indent=2)

        print(f"\nSaved metadata for {len(records)} clips to {metadata_csv} and {metadata_json}")
    else:
        print("\nNo records written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-clips", type=int, default=None,
                        help="Only process the first N videos (for quick testing).")
    args = parser.parse_args()
    main(max_clips=args.max_clips)
