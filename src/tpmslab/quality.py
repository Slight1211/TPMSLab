"""Conservative candidate selection without moving any boundary vertex.

This is a local heuristic, not a minimum-angle guarantee or global optimizer.
Canonical polygon triangles are identical in both candidates. A centroid fan
is accepted only if its worst mean-ratio quality strictly improves and its
signed-volume magnitude sum agrees with the pulling fan.
"""

import numpy as np


def tetra_metrics(points):
    p = np.asarray(points, dtype=float)
    volumes = (
        np.abs(
            np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
        )
        / 6
    )
    lengths = sum(
        np.sum((p[:, a] - p[:, b]) ** 2, axis=1)
        for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    )
    quality = np.divide(
        12 * (3 * volumes) ** (2 / 3), lengths, out=np.zeros_like(volumes), where=lengths > 0
    )
    return volumes, quality


def choose_fan(nodes, anchor, vertices, triangles, strategy):
    baseline = [[anchor, *tri] for tri in triangles if anchor not in tri]
    if strategy == "pulling" or not baseline:
        return baseline, None, None
    # Convert only the current polytope, never the full accumulated node list.
    base_xyz = np.array([[nodes[i] for i in tet] for tet in baseline])
    vb, qb = tetra_metrics(base_xyz)
    if np.any(vb <= 0) or qb.min() >= 0.3:
        return baseline, None, None
    point = np.mean([nodes[i] for i in vertices], axis=0)
    candidate = np.array([[point, *[nodes[i] for i in tri]] for tri in triangles])
    vc, qc = tetra_metrics(candidate)
    if np.any(vc <= 0) or qc.min() <= qb.min() * (1 + 1e-6):
        return baseline, None, None
    if not np.isclose(vc.sum(), vb.sum(), rtol=1e-10, atol=0):
        return baseline, None, None
    return [[len(nodes), *tri] for tri in triangles], point, float(qc.min() / qb.min())
