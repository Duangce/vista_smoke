from .util import get_configs_path, instantiate_from_config

# Delay heavy model imports until they are actually needed.  This lets the data
# pipeline be imported/tested without pulling in pytorch_lightning, kornia,
# open_clip, transformers, etc.
def __getattr__(name):
    if name in ("AutoencodingEngine", "DiffusionEngine"):
        from .models import AutoencodingEngine, DiffusionEngine
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "0.1.0"
