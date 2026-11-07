"""Bounded, off-grid checks for grossly under-resolved fields.

These probes detect some missed interfaces; they are not a geometry error
bound. Mesh and calibration refinement remain necessary for custom formulas.
Custom formulas are rejected when a probe finds a missed interface; built-in
families (some of which carry second and third harmonics) are probed too, and
a miss is reported as a warning so that coarse studies stay reproducible.
"""

import numpy as np

from .model import field_and_threshold


def audit_custom_sampling(config, lattice_fields, corner_ids, voxel_indices, spacing):
    custom = config.family == "Custom" or (
        config.combine == "blend" and config.family_secondary == "Custom"
    )
    # Neither a repeated midpoint grid nor a shifted regular grid alone is
    # sufficient. Use two interior probes with different coordinates per axis.
    offsets = np.asarray([[0.2113248654, 0.3660254038, 0.6180339887],
                          [0.7886751346, 0.6339745962, 0.3819660113]])
    missed_void = missed_solid = 0
    for start in range(0, len(voxel_indices), 4096):
        stop = start + 4096
        corners = lattice_fields[:, corner_ids[start:stop]]
        full = np.all(corners >= 0, axis=(0, 2))
        empty = np.any(np.max(corners, axis=2) < 0, axis=0)
        void = np.zeros(len(full), dtype=bool)
        solid = np.zeros(len(full), dtype=bool)
        for offset in offsets:
            xyz = (voxel_indices[start:stop] + offset) * spacing
            f, q = field_and_threshold(config, *xyz.T)
            if config.mode == "sheet":
                fields = np.asarray([q - f, q + f])
            else:
                fields = np.asarray([q + f if config.mode == "solid_above" else q - f])
            fields = np.broadcast_to(fields, (len(lattice_fields), len(full)))
            tolerance = max(1.0, float(np.max(np.abs(fields)))) * 1e-10
            void |= full & np.any(fields < -tolerance, axis=0)
            solid |= empty & np.all(fields > tolerance, axis=0)
        missed_void += int(void.sum())
        missed_solid += int(solid.sum())
    result = {"performed": True, "voxels_checked": len(voxel_indices),
              "probes_per_voxel": len(offsets), "missed_void_voxels": missed_void,
              "missed_solid_voxels": missed_solid, "custom_field": custom,
              "policy": "reject" if custom else "warn",
              "scope": "two off-grid probes; not a bound on geometric or calibration error"}
    if (missed_void or missed_solid) and custom:
        raise ValueError(
            f"Custom field is under-resolved: off-grid probes found {missed_void} voxels "
            f"with missed void and {missed_solid} with missed solid. Increase resolution "
            "and check m_cal refinement before COMSOL export."
        )
    return result
