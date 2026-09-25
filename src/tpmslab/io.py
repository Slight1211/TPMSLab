"""Deterministic model export and file-integrity manifest."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np
from .volume import write_nastran


def save_model(model, directory):
    """Write NAS, NPZ, STL preview, parameters, checks and hashes to a NEW folder.

    Existing directories are rejected. STL is a preview, not the solver mesh.
    """
    folder = Path(directory).resolve()
    if folder.exists():
        raise FileExistsError(f"Output already exists: {folder}")
    folder.mkdir(parents=True)
    write_nastran(model, folder / "mesh.nas")
    np.savez_compressed(
        folder / "mesh.npz",
        points_mm=model["points"],
        tetrahedra=model["tetra"],
        boundary=model["boundary"],
        boundary_ids=model["boundary_ids"],
        domain_ids=model["domains"],
        **{
            k: model[k]
            for k in ("phase_ids", "interface", "solid_boundary", "fluid_boundary")
            if k in model
        },
    )
    model["surface"].export(folder / "preview.stl")
    if "fluid_surface" in model:
        model["fluid_surface"].export(folder / "fluid_preview.stl")
        (folder / "domains.json").write_text(
            json.dumps(
                {
                    "domains": model["report"]["domain_map"],
                    "boundaries": model["report"]["boundary_tag_names"],
                },
                indent=2,
            ),
            encoding="utf8",
        )
    for filename, data in (
        ("config.json", model["report"]["config"]),
        ("report.json", model["report"]),
    ):
        (folder / filename).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8"
        )
    environment = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "scipy", "trimesh"):
        environment[name] = importlib.metadata.version(name)
    (folder / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf8")
    refresh_manifest(folder)
    return folder


def refresh_manifest(directory):
    """Refresh checksums after optional COMSOL artifacts change."""
    folder = Path(directory)
    manifest = {
        p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(folder.iterdir())
        if p.is_file() and p.name != "manifest.json"
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    return folder
