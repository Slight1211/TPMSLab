"""Independent face-incidence audit for complementary volume meshes."""

import numpy as np


def audit_interface(model):
    """Count invalid/missing/duplicate interface faces without trusting the report."""
    points = np.asarray(model["points"])
    tetra = np.asarray(model["tetra"])
    phases = np.asarray(model["phase_ids"])
    interface = np.sort(np.asarray(model["interface"]), axis=1)
    # Independent local-face ordering; canonical faces do not depend on orientation.
    faces = np.sort(tetra[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3), axis=1)
    keys, inverse, counts = np.unique(faces, axis=0, return_inverse=True, return_counts=True)
    solid = np.bincount(inverse, weights=np.repeat(phases == 1, 4), minlength=len(keys))
    fluid = np.bincount(inverse, weights=np.repeat(phases == 2, 4), minlength=len(keys))
    expected = {tuple(k) for k in keys[(counts == 2) & (solid == 1) & (fluid == 1)]}
    provided = [tuple(k) for k in interface]
    unique = set(provided)
    invalid = sum(k not in expected for k in provided)
    p = points[interface]
    area = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum() / 2
    return {
        "interface_triangles": len(interface),
        "invalid_interface_triangles": int(invalid),
        "missing_interface_triangles": len(expected - unique),
        "duplicate_interface_triangles": len(provided) - len(unique),
        "nonmanifold_faces": int(np.sum(counts > 2)),
        "interface_area_mm2": float(area),
        "valid": not (
            invalid or expected - unique or len(provided) != len(unique) or np.any(counts > 2)
        ),
    }


def audit_openings(model, size):
    """Measure each fluid component's two z openings from its tetrahedra."""
    points = np.asarray(model["points"])
    tetra = np.asarray(model["tetra"])
    phases = np.asarray(model["phase_ids"])
    domains = np.asarray(model["domains"])
    result = []
    for domain in np.unique(domains[phases == 2]):
        cells = tetra[domains == domain]
        faces = np.sort(
            cells[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3), axis=1
        )
        faces, counts = np.unique(faces, axis=0, return_counts=True)
        faces = faces[counts == 1]
        p = points[faces]
        areas = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1) / 2
        entry = {"domain_id": int(domain), "tetrahedra": len(cells)}
        for name, z in [("inlet", 0.0), ("outlet", size[2])]:
            mask = np.all(np.abs(p[:, :, 2] - z) < max(size) * 1e-9, axis=1)
            entry[name + "_area_mesh_mm2"] = float(areas[mask].sum())
        result.append(entry)
    return result
