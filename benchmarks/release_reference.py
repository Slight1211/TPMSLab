"""Validate an installed distribution against the versioned Gyroid reference."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import tpmslab
from tpmslab import Config, generate, save_model
from tpmslab.comsol import build_mph
from tpmslab.comsol_audit import audit_saved_flow
from tpmslab.verification import audit_interface, audit_openings


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--solve", action="store_true")
    a = p.parse_args()
    src = a.source.resolve()
    out = a.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    installed = Path(tpmslab.__file__).resolve().parent
    if installed == src / "src" / "tpmslab":
        raise RuntimeError("Install the wheel first; this check must use the installed package")
    assert importlib.metadata.version("tpmslab") == "0.3.0"
    provenance = json.loads((src / "validation/provenance.json").read_text(encoding="utf8"))
    for record in provenance["algorithm_files_unchanged"]:
        original = src / record["path"]
        packaged = installed / Path(record["path"]).name
        assert digest(original) == record["git_lf_sha256"], record["path"]
        assert digest(packaged) == record["git_lf_sha256"], packaged
    checksums = json.loads((src / "validation/checksums.json").read_text(encoding="utf8"))
    for relative, expected in checksums.items():
        assert digest(src / "validation" / relative) == expected, relative
    cfg = json.loads((src / "validation/reference_configs/Gyroid.json").read_text(encoding="utf8"))
    start = time.perf_counter()
    mesh = generate(Config(**cfg))
    interface = audit_interface(mesh)
    assert interface["valid"], interface
    tet = mesh["points"][mesh["tetra"]]
    volumes = np.abs(np.linalg.det(tet[:, 1:] - tet[:, :1])) / 6
    vs = float(volumes[mesh["phase_ids"] == 1].sum())
    vf = float(volumes[mesh["phase_ids"] == 2].sum())
    data = json.loads((src / "validation/summary.json").read_text(encoding="utf8"))
    reference = next(row for row in data["refinement"] if row["n"] == cfg["resolution"])
    assert len(mesh["tetra"]) == reference["tetrahedra"]
    assert np.isclose(vs, reference["solid_volume"], rtol=1e-11, atol=1e-10)
    assert np.isclose(vf, reference["fluid_volume"], rtol=1e-11, atol=1e-10)
    assert np.isclose(vs + vf, np.prod(cfg["size"]), rtol=1e-12)
    ids, counts = np.unique(mesh["domains"], return_counts=True)
    assert dict(zip(map(int, ids), map(int, counts))) == {1: 18305, 2: 9978, 3: 10009}
    folder = save_model(mesh, out)
    archive = np.load(folder / "mesh.npz")
    assert np.array_equal(archive["tetrahedra"], mesh["tetra"])
    assert np.array_equal(archive["points_mm"], mesh["points"])
    result = {
        "kind": "fresh_installed_distribution_reference_check",
        "historical_measurements_overwritten": False,
        "version": importlib.metadata.version("tpmslab"),
        "python": platform.python_version(),
        "executable": sys.executable,
        "installed_module": str(installed),
        "scientific_baseline_commit": provenance["scientific_baseline_commit"],
        "config": cfg,
        "tetrahedra": len(mesh["tetra"]),
        "domain_tetrahedra": dict(zip(map(int, ids), map(int, counts))),
        "solid_volume_mm3": vs,
        "fluid_volume_mm3": vf,
        "interface": interface,
        "openings": audit_openings(mesh, cfg["size"]),
        "mesh_and_export_seconds": time.perf_counter() - start,
        "comsol_requested": a.solve,
    }
    if a.solve:
        build_mph(folder, mesh["report"], solve=True)
        verification = json.loads((folder / "comsol_verification.json").read_text())
        assert verification["solved"] and verification["reopen_verified"]
        result["comsol_verification"] = verification
        result["saved_flow_audit"] = audit_saved_flow(
            folder / "model_solved.mph",
            interface["interface_area_mm2"],
            mesh["report"]["domain_map"],
            folder / "independent_reopen_audit",
        )
        q = verification["outlet_flux_m3_s"]
        result["outlet_flow_relative_difference_from_historical"] = abs(
            q - reference["outlet_flow"]
        ) / abs(reference["outlet_flow"])
        assert result["outlet_flow_relative_difference_from_historical"] < 1e-5
    (folder / "release_reference_check.json").write_text(
        json.dumps(result, indent=2), encoding="utf8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
