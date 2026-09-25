"""Reproduce geometry, performance and sequential licensed solver refinement checks.
Run from a source checkout with --out NEW_DIRECTORY. Windows COMSOL is optional.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from tpmslab import Config, generate, save_model
from tpmslab.comsol import build_mph
from tpmslab.comsol_audit import audit_saved_flow
from tpmslab.verification import audit_interface


def peak_rss_mib():
    if os.name == "nt":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t)
                for n in [
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                ]
            ]

        c = Counters()
        c.cb = ctypes.sizeof(c)
        get = ctypes.windll.kernel32.GetCurrentProcess
        get.restype = wintypes.HANDLE
        info = ctypes.windll.psapi.GetProcessMemoryInfo
        info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        info.restype = wintypes.BOOL
        ok = info(get(), ctypes.byref(c), c.cb)
        if not ok:
            raise OSError("GetProcessMemoryInfo failed")
        return c.PeakWorkingSetSize / 2**20
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (2**20 if sys.platform == "darwin" else 1024)


def worker(folder, n, mode, strategy):
    config = Config(resolution=n, domain_mode=mode, quality_strategy=strategy)
    start = time.perf_counter()
    model = generate(config)
    generation = time.perf_counter() - start
    start = time.perf_counter()
    save_model(model, folder)
    exporting = time.perf_counter() - start
    result = dict(
        resolution=n,
        mode=mode,
        strategy=strategy,
        generation_s=generation,
        export_s=exporting,
        peak_process_rss_mib=peak_rss_mib(),
        python=platform.python_version(),
        report=model["report"],
    )
    if mode == "solid_fluid":
        result["interface_audit"] = audit_interface(model)
        if not result["interface_audit"]["valid"]:
            raise RuntimeError("Independent interface audit failed")
    (folder / "measurement.json").write_text(json.dumps(result, indent=2), "utf8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resolutions", nargs="+", type=int, default=[8, 12, 16, 20])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--solve", action="store_true")
    p.add_argument("--max-solve-tetrahedra", type=int, default=180_000)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--mode", default="solid_fluid")
    p.add_argument("--strategy", default="quality_fan")
    args = p.parse_args()
    if args.worker:
        worker(args.out, args.resolutions[0], args.mode, args.strategy)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    cases = [(n, args.strategy) for n in args.resolutions]
    for n, strategy in cases:
        for mode in ["solid_fluid", "solid"]:
            repeats = args.repeats if mode == "solid_fluid" else 1
            for rep in range(repeats):
                folder = args.out / f"{mode}_n{n}_{strategy}_r{rep + 1}"
                print("GENERATE", folder, flush=True)
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--out",
                    str(folder),
                    "--resolutions",
                    str(n),
                    "--mode",
                    mode,
                    "--strategy",
                    strategy,
                ]
                subprocess.run(cmd, check=True, timeout=600)
                record = json.loads((folder / "measurement.json").read_text("utf8"))
                record["folder"] = str(folder.resolve())
                record["repeat"] = rep + 1
                if args.solve and rep == 0:
                    print("SOLVE", folder, flush=True)
                    start = time.perf_counter()
                    record["comsol"] = build_mph(
                        folder,
                        record["report"],
                        solve=True,
                        max_solve_tetrahedra=args.max_solve_tetrahedra,
                        progress=lambda s: print(s, flush=True),
                    )
                    record["comsol_bridge_wall_s"] = time.perf_counter() - start
                    if mode == "solid_fluid":
                        record["comsol_audit"] = audit_saved_flow(
                            folder / "model_solved.mph",
                            record["interface_audit"]["interface_area_mm2"],
                            record["report"]["domain_map"],
                            folder / "solver_audit",
                        )
                results.append(record)
                (args.out / "results.json").write_text(json.dumps(results, indent=2), "utf8")
                print("DONE", n, mode, rep + 1, flush=True)


if __name__ == "__main__":
    main()
