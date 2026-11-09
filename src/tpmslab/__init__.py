"""Public API for TPMS Lab. Mesh coordinates use millimetres."""

from .model import Config
from .volume import generate_volume, write_nastran
from .io import save_model

__version__ = "0.3.9"
__all__ = ["Config", "generate", "generate_volume", "write_nastran", "save_model", "list_families", "remesh_in_comsol", "condition_mesh", "quality_audit", "run_comsol", "optimize_mesh"]


def generate(config=None, *, progress=None, condition=False, optimize=False):
    """Return a mesh dictionary with points, tetrahedra, boundaries and report.

    ``config`` may be Config, a configuration dictionary, or None for defaults.
    Array connectivity is zero-based. Coordinates are millimetres.
    ``optimize=True`` runs :func:`optimize_mesh` (cut-layer reconnection,
    interior smoothing and surface-constrained boundary smoothing) on the
    result; ``condition=True`` runs interior conditioning only.
    """
    if config is None:
        config = Config()
    if isinstance(config, dict):
        config = Config.from_dict(config)
    if not isinstance(config, Config):
        raise TypeError("config must be Config or dict")
    if type(condition) is not bool or type(optimize) is not bool:
        raise TypeError("condition and optimize must be booleans")
    model = generate_volume(config, progress=progress or (lambda message: None))
    if optimize:
        return optimize_mesh(model, progress=progress)
    return condition_mesh(model, progress=progress) if condition else model


def condition_mesh(model, **kwargs):
    """Opt-in interior conditioning with fixed boundaries and audited rollback."""
    from .conditioning import condition_mesh as _condition

    return _condition(model, **kwargs)


def optimize_mesh(model, **kwargs):
    """Cut-layer reconnection, interior smoothing and implicit-surface sliding."""
    from .optimize import optimize_mesh as _optimize

    return _optimize(model, **kwargs)


def quality_audit(model, **kwargs):
    """Independently reconstruct topology and apply optional quality limits."""
    from .conditioning import quality_audit as _audit

    return _audit(model, **kwargs)


def list_families():
    """Return a copy of the built-in formula metadata, including aliases."""
    from .model import FAMILIES

    return {name: dict(data) for name, data in FAMILIES.items()}


def remesh_in_comsol(*args, **kwargs):
    """Optionally remesh a verified two-phase export with licensed COMSOL."""
    from .comsol_remesh import remesh_in_comsol as _remesh

    return _remesh(*args, **kwargs)


def run_comsol(directory, **kwargs):
    """Run a saved model in COMSOL with a separate directory per attempt."""
    from .comsol import run_comsol as _run

    return _run(directory, **kwargs)
