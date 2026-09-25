"""Public API for TPMS Lab. Mesh coordinates use millimetres."""

from .model import Config
from .volume import generate_volume, write_nastran
from .io import save_model

__version__ = "0.3.0rc1"
__all__ = ["Config", "generate", "generate_volume", "write_nastran", "save_model", "list_families"]


def generate(config=None, *, progress=None):
    """Return a mesh dictionary with points, tetrahedra, boundaries and report.

    ``config`` may be Config, a configuration dictionary, or None for defaults.
    Array connectivity is zero-based. Coordinates are millimetres.
    """
    if config is None:
        config = Config()
    if isinstance(config, dict):
        config = Config.from_dict(config)
    if not isinstance(config, Config):
        raise TypeError("config must be Config or dict")
    return generate_volume(config, progress=progress or (lambda message: None))


def list_families():
    """Return a copy of the built-in formula metadata, including aliases."""
    from .model import FAMILIES

    return {name: dict(data) for name, data in FAMILIES.items()}
