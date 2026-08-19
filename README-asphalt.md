# Vista on Asphalt Wetness Dataset

This repo extends [OpenDriveLab/Vista](https://github.com/OpenDriveLab/Vista) for fine-tuning on the internal `asphalt_wetness0` driving videos.

## What was added

- `renew/01_extract_frames.py` — extract frames from `data/*_back.mp4` videos
- `renew/02_build_samples.py` — build training samples with trajectory / command / speed / angle / goal
- `renew/03_test_dataloader.py` — sanity-check the data loader
- `renew/finetune_asphalt.sh` — one-click preprocessing + fine-tuning
- `vwm/data/subsets/asphalt.py` — `AsphaltDataset` registered in Vista
- `configs/training/asphalt_finetune.yaml` — training config for asphalt data
- `requirements-asphalt.txt` — verified dependency versions

## Quick start

### 1. Clone & prepare data

```bash
git clone https://github.com/Duangce/vista_smoke.git
cd vista_smoke

# Put your videos + CSVs under data/
# Expected structure:
# data/asphalt_wetness0/<scenario>/<maneuver>/<vehicle>/<clip>_back.mp4
# data/asphalt_wetness0/<scenario>/<maneuver>/<vehicle>/<clip>.csv
```

### 2. Environment

```bash
python -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements-asphalt.txt
```

### 3. Download Vista pre-trained weights

```bash
python renew/download_vista_weights.py
# -> ckpts/vista.safetensors
```

### 4. Fine-tune

```bash
# Quick smoke test (first 3 clips only)
bash renew/finetune_asphalt.sh --quick

# Full fine-tuning
bash renew/finetune_asphalt.sh
```

## Training config highlights

| Key | Value |
|---|---|
| resolution | 1024×576 (frames are resized to this) |
| num_frames | 25 |
| batch_size | 1 |
| max_epochs | 10 |
| actions | trajectory, command, speed, angle, goal |

## Notes

- `data/`, `ckpts/`, `renew/frames/`, `renew/annotations/`, `.venv/`, `.local/` are excluded from Git.
- The original video view is rear/top-down, different from Vista's front-facing pre-training. Fine-tune with care.
- `command` IDs follow Vista: `0=right`, `1=left`, `2=stop`, `3=straight`.
