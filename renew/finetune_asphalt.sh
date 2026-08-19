#!/usr/bin/env bash
# Fine-tune Vista on the asphalt_wetness0 dataset.
# Usage:
#   bash renew/finetune_asphalt.sh              # full data
#   bash renew/finetune_asphalt.sh --quick      # first 3 clips only (smoke test)

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

cd "$PROJECT_ROOT"

QUICK=0
if [[ "$1" == "--quick" ]]; then
    QUICK=1
    echo "[QUICK MODE] Using first 3 clips only."
fi

# ------------------------------------------------------------------
# 1. Data preprocessing (skip if frames already exist)
# ------------------------------------------------------------------
if [ ! -d "$PROJECT_ROOT/renew/frames" ]; then
    echo "[1/3] Extracting frames..."
    if [ "$QUICK" -eq 1 ]; then
        "$VENV_PYTHON" "$PROJECT_ROOT/renew/01_extract_frames.py" --max-clips 3
    else
        "$VENV_PYTHON" "$PROJECT_ROOT/renew/01_extract_frames.py"
    fi
else
    echo "[1/3] Frames already exist, skipping extraction."
fi

if [ ! -f "$PROJECT_ROOT/renew/annotations/asphalt_wetness0.json" ]; then
    echo "[2/3] Building training samples..."
    if [ "$QUICK" -eq 1 ]; then
        "$VENV_PYTHON" "$PROJECT_ROOT/renew/02_build_samples.py" --max-clips 3
    else
        "$VENV_PYTHON" "$PROJECT_ROOT/renew/02_build_samples.py"
    fi
else
    echo "[2/3] Annotation already exists, skipping sample building."
fi

# ------------------------------------------------------------------
# 2. Sanity check
# ------------------------------------------------------------------
echo "[3/3] Sanity check on AsphaltDataset..."
"$VENV_PYTHON" - <<'PY'
from vwm.data.subsets import AsphaltDataset
ds = AsphaltDataset()
print(f"AsphaltDataset: {len(ds)} samples")
item = ds[0]
print(f"img_seq: {item['img_seq'].shape}")
for key in ("trajectory", "command", "speed", "angle", "goal"):
    if key in item:
        print(f"{key}: {item[key].shape}")
PY

# ------------------------------------------------------------------
# 3. Launch training
# ------------------------------------------------------------------
echo "Launching fine-tuning..."
"$VENV_PYTHON" "$PROJECT_ROOT/train.py" \
    --base "$PROJECT_ROOT/configs/training/asphalt_finetune.yaml" \
    --finetune "$PROJECT_ROOT/ckpts/vista.safetensors" \
    --n_devices 1 \
    --logdir "$PROJECT_ROOT/logs/asphalt_finetune"
