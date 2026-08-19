"""Download the official Vista pretrained weights from HuggingFace."""
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "OpenDriveLab/Vista"
FILENAME = "vista.safetensors"
LOCAL_DIR = Path("/workspace/ws/Vista/ckpts")


def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID}/{FILENAME} ...")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(LOCAL_DIR),
        local_dir_use_symlinks=False,
    )
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
