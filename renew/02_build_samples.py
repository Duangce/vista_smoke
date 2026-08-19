"""
Build Vista training samples from extracted frames and per-frame ego trajectories.

Input:
    renew/metadata.json   (produced by 01_extract_frames.py)
    data/.../*.csv        (per-frame ego pose, speed, steer, phase)

Output:
    renew/annotations/asphalt_wetness0.json
        List of samples, each sample is a dict with:
            frames:  list of frame paths relative to data_root
            traj:    list of 10 floats [x0,y0, x1,y1, x2,y2, x3,y3, x4,y4] in ego-local coords
            cmd:     int command id
            speed:   list of 5 future speeds (m/s)
            angle:   list of 5 future steer angles (normalized)
            goal:    list of 2 floats [gx, gy]
            z:       float camera focal length (placeholder)

Note on Vista slicing:
    NuScenesDataset reads traj[2:]  -> 8 floats (4 waypoints)
                       speed[1:]   -> 4 floats
                       angle[1:]   -> 4 floats
    Therefore we store the full sequences including t=0.
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd


META_PATH = Path("/workspace/ws/Vista/renew/metadata.json")
DATA_ROOT = Path("/workspace/ws/Vista/renew")  # frames live under here
OUT_PATH = Path("/workspace/ws/Vista/renew/annotations/asphalt_wetness0.json")

NUM_FRAMES = 25          # frames per sample (condition + future)
TRAJ_POINTS = 5          # number of trajectory waypoints (t=0,0.5,1.0,1.5,2.0s)
SPEED_POINTS = 5
ANGLE_POINTS = 5
TRAJ_HORIZON_S = 2.0     # trajectory covers 0..2 seconds


def phase_to_command(phase: str) -> int:
    p = str(phase).strip().lower()
    if "right" in p:
        return 0
    if "left" in p:
        return 1
    if p in ("stop", "brake"):
        return 2
    return 3  # straight / accel / cruise / unknown


def world_to_ego(dx_world, dy_world, yaw_deg):
    """Convert world-coordinate displacement into ego-local (forward, left).

    Ego frame: x = forward, y = left, z = up.
    The source data uses yaw ~ +/-180 deg with the vehicle travelling along
    decreasing world-x.  Lateral direction is flipped so that phase='Left'
    produces a positive leftward displacement.
    """
    yaw = math.radians(yaw_deg)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    # World x axis points to ego forward when yaw=0.
    forward = dx_world * cos_y + dy_world * sin_y
    # Flip sign so that the dataset's 'Left' maneuvers map to positive left.
    left = dx_world * sin_y - dy_world * cos_y
    return forward, left


def _nearest_index(df, start_idx, target_t):
    idx = start_idx
    while idx < len(df) - 1 and df["t"].iloc[idx] < target_t:
        idx += 1
    return idx


def extract_trajectory(df, start_idx, n_points=TRAJ_POINTS, horizon_s=TRAJ_HORIZON_S):
    """Sample (forward, left) relative displacements at t=0..horizon_s seconds."""
    dt = horizon_s / (n_points - 1) if n_points > 1 else 0.0
    traj = []
    t0 = df["t"].iloc[start_idx]
    x0 = df["x"].iloc[start_idx]
    y0 = df["y"].iloc[start_idx]
    yaw0 = df["yaw"].iloc[start_idx]

    for k in range(n_points):
        target_t = t0 + k * dt
        idx = _nearest_index(df, start_idx, target_t)
        x = df["x"].iloc[idx]
        y = df["y"].iloc[idx]
        forward, left = world_to_ego(x - x0, y - y0, yaw0)
        traj.extend([forward, left])
    return traj


def extract_speed_angle(df, start_idx, n_points=SPEED_POINTS, horizon_s=TRAJ_HORIZON_S):
    """Sample speed and steer at t=0..horizon_s seconds."""
    dt = horizon_s / (n_points - 1) if n_points > 1 else 0.0
    speed, angle = [], []
    t0 = df["t"].iloc[start_idx]

    for k in range(n_points):
        target_t = t0 + k * dt
        idx = _nearest_index(df, start_idx, target_t)
        speed.append(float(df["speed_ms"].iloc[idx]))
        # Normalize steer to a roughly [-1, 1] range using a fixed max of 45 deg.
        steer_deg = float(df["steer"].iloc[idx])
        angle.append(steer_deg / 45.0)

    return speed, angle


def extract_goal(df, start_idx, horizon_s=TRAJ_HORIZON_S):
    """Return the (forward, left) displacement to the goal point at horizon_s."""
    t0 = df["t"].iloc[start_idx]
    x0 = df["x"].iloc[start_idx]
    y0 = df["y"].iloc[start_idx]
    yaw0 = df["yaw"].iloc[start_idx]

    target_t = t0 + horizon_s
    idx = _nearest_index(df, start_idx, target_t)

    x = df["x"].iloc[idx]
    y = df["y"].iloc[idx]
    return list(world_to_ego(x - x0, y - y0, yaw0))


def main(max_clips: int = None):
    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    if max_clips is not None and max_clips > 0:
        metadata = metadata[:max_clips]

    all_samples = []

    for clip_meta in metadata:
        csv_path = Path(clip_meta["csv_path"])
        frame_dir = Path(clip_meta["frame_dir"])
        num_frames = clip_meta["num_frames"]

        df = pd.read_csv(csv_path)

        # Ensure CSV length >= frame count; if not, truncate to min length
        usable_frames = min(num_frames, len(df))

        # Slide a window of NUM_FRAMES across the clip
        max_start = usable_frames - NUM_FRAMES
        if max_start < 0:
            print(f"[SKIP] {clip_meta['clip']} too short ({usable_frames} frames)")
            continue

        for start in range(0, max_start + 1, NUM_FRAMES):
            end = start + NUM_FRAMES
            if end > usable_frames:
                break

            # Build relative frame paths (relative to DATA_ROOT)
            frames = [
                str(Path(frame_dir).relative_to(DATA_ROOT) / f"{i + 1:04d}.png")
                for i in range(start, end)
            ]

            traj = extract_trajectory(df, start)
            speed, angle = extract_speed_angle(df, start)
            goal = extract_goal(df, start)

            # command derived from the phase at the condition frame
            cmd = phase_to_command(df["phase"].iloc[start])

            sample = {
                "frames": frames,
                "traj": traj,
                "cmd": cmd,
                "speed": speed,
                "angle": angle,
                "goal": goal,
                "z": 1000.0,  # placeholder focal length; set if calibration is available
            }
            all_samples.append(sample)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(all_samples, f, indent=2)

    print(f"Built {len(all_samples)} training samples from {len(metadata)} clips")
    print(f"Saved annotation to {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-clips", type=int, default=None,
                        help="Only process the first N clips listed in metadata.json.")
    args = parser.parse_args()
    main(max_clips=args.max_clips)
