"""Runs inside the microgen venv (numpy<2). Usage: worker.py family q_table.npz resolution out.npz"""
import sys
import time
import numpy as np
import pyvista as pv
from microgen import Tpms

FUN = {
    "Gyroid": lambda X, Y, Z: np.sin(X)*np.cos(Y) + np.sin(Z)*np.cos(X) + np.sin(Y)*np.cos(Z),
    "Diamond": lambda X, Y, Z: np.cos(X)*np.cos(Y)*np.cos(Z) - np.sin(X)*np.sin(Y)*np.sin(Z),
    "Primitive Schwartz": lambda X, Y, Z: np.cos(X) + np.cos(Y) + np.cos(Z),
    "IWP": lambda X, Y, Z: 2*(np.cos(X)*np.cos(Y)+np.cos(Y)*np.cos(Z)+np.cos(Z)*np.cos(X)) - (np.cos(2*X)+np.cos(2*Y)+np.cos(2*Z)),
}
family, table, res, out = sys.argv[1], np.load(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
zs, qs = table["z"], table["q"]
phase = np.deg2rad(table["phase_degrees"])
f = FUN[family]
def surface(a, b, c):
    return f(a + np.pi + phase[0], b + np.pi + phase[1], c + np.pi + phase[2])
def offset(x, y, z):
    return 2.0 * np.interp(np.asarray(z) + 2.5, zs, qs)
t0 = time.perf_counter()
tpms = Tpms(surface_function=surface, offset=offset, cell_size=5.0, repeat_cell=1, resolution=res)
grid = tpms.generate_grid_vtk(type_part="sheet")
tets = grid.triangulate()
t1 = time.perf_counter()
cells = tets.cells_dict.get(pv.CellType.TETRA, np.zeros((0, 4), int))
other = {int(k): len(v) for k, v in tets.cells_dict.items() if k != pv.CellType.TETRA}
np.savez(out, points=np.asarray(tets.points, float) + 2.5, tets=np.asarray(cells, np.int64), seconds=t1 - t0,
         raw_cells={int(k): len(v) for k, v in grid.cells_dict.items()}.__repr__(), other=repr(other))
print("ok", len(cells), "tets", round(t1 - t0, 2), "s", "raw", {int(k): len(v) for k, v in grid.cells_dict.items()}, "other", other)
