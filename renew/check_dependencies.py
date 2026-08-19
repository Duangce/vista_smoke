"""Check which Python packages required by Vista are missing or broken."""
import importlib
import sys

REQUIREMENTS = [
    # core
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("torchaudio", "torchaudio"),
    ("numpy", "numpy"),
    ("PIL", "pillow"),
    ("cv2", "opencv-python"),
    ("pandas", "pandas"),
    ("tqdm", "tqdm"),
    ("scipy", "scipy"),
    # training
    ("pytorch_lightning", "pytorch-lightning"),
    ("omegaconf", "omegaconf"),
    ("einops", "einops"),
    ("kornia", "kornia"),
    ("open_clip", "open-clip-torch"),
    ("transformers", "transformers"),
    ("safetensors", "safetensors"),
    ("imageio", "imageio"),
    ("imageio_ffmpeg", "imageio-ffmpeg"),
    ("matplotlib", "matplotlib"),
    ("natsort", "natsort"),
    ("packaging", "packaging"),
    ("wandb", "wandb"),
    ("tensorboardX", "tensorboardx"),
    ("fsspec", "fsspec"),
    ("webdataset", "webdataset"),
    ("timm", "timm"),
    ("fire", "fire"),
    ("fairscale", "fairscale"),
    # optional / inference helpers
    ("ninja", "ninja"),
    ("xformers", "xformers"),
    ("deepspeed", "deepspeed"),
    ("invisible_watermark", "invisible-watermark"),
    ("streamlit", "streamlit"),
]

missing = []
broken = []

for module_name, pkg_name in REQUIREMENTS:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"OK   {module_name:24s} {pkg_name:24s} {version}")
    except ImportError as e:
        print(f"MISS {module_name:24s} {pkg_name:24s} -> {e}")
        missing.append((module_name, pkg_name))
    except Exception as e:
        print(f"BROK {module_name:24s} {pkg_name:24s} -> {e}")
        broken.append((module_name, pkg_name, e))

print("\n" + "=" * 60)
if missing:
    print(f"Missing packages ({len(missing)}):")
    for _, pkg in missing:
        print(f"  pip install {pkg}")
if broken:
    print(f"Broken packages ({len(broken)}):")
    for mod, pkg, e in broken:
        print(f"  {pkg}: {e}")
if not missing and not broken:
    print("All checked packages are importable.")
