"""COMSOL Java bridge (verified on 6.2.0.290; 6.3+ API handled by reflection).

Generated MPH files embed the volume mesh. No edits to the user's existing
MPH files; each export gets a new attempt directory. All COMSOL launches in
this package go through :func:`run_comsol_class`, which applies the same
isolated preferences, short scratch/recovery paths, log capture and marker
parsing to the elasticity/flow bridge, the quality remesh and the audits.
"""

import contextlib
import json
import re
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np

MESH_COMPAT = r"""static void independentMesh(Model m)throws Exception {
  MeshSequence mesh=m.component("c").mesh("mesh");
  try {MeshSequence.class.getMethod("geometricModel",String.class).invoke(mesh,"");}
  catch(NoSuchMethodException ex) {
   MeshSequence.class.getMethod("geometricModel",Boolean.TYPE).invoke(mesh,Boolean.TRUE);
  }
 }"""

TEMPLATE = r"""import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.*;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;
public class TPMSBuild {
 static String ROOT=@@ROOT@@;
 static boolean SOLVE=@@SOLVE@@;
 @@MESH_COMPAT@@
 static void face(Model m,String tag,double value) {
  m.component("c").selection().create(tag,"Box");
  m.component("c").selection(tag).geom("g",2);
  m.component("c").selection(tag).set("condition","inside");
  m.component("c").selection(tag).set("xmin",-1.0);
  m.component("c").selection(tag).set("xmax",@@LX@@+1.0);
  m.component("c").selection(tag).set("ymin",-1.0);
  m.component("c").selection(tag).set("ymax",@@LY@@+1.0);
  m.component("c").selection(tag).set("zmin",value-@@TOL@@);
  m.component("c").selection(tag).set("zmax",value+@@TOL@@);
  if(m.component("c").selection(tag).entities(2).length==0) throw new RuntimeException("Empty boundary "+tag);
 }
 public static void main(String[] args)throws Exception {
  Model m=ModelUtil.create("Model");
  m.label("TPMS Lab - direct solid volume mesh");m.modelPath(ROOT);
  m.param().set("E0","@@E@@[Pa]");m.param().set("nu0","@@NU@@");
  m.param().set("rho0","@@RHO@@[kg/m^3]");m.param().set("uamp","@@DISP@@[mm]");
  m.component().create("c",true);m.component("c").geom().create("g",3);
  m.component("c").geom("g").lengthUnit("mm");
  m.component("c").mesh().create("mesh","g");independentMesh(m);
  m.component("c").mesh("mesh").create("imp","Import");
  m.component("c").mesh("mesh").feature("imp").set("source","nastran");
  m.component("c").mesh("mesh").feature("imp").set("filename",ROOT+"mesh.nas");
  m.component("c").mesh("mesh").feature("imp").set("data","mesh");
  m.component("c").mesh("mesh").feature("imp").set("allowshellpartition",false);
  m.component("c").mesh("mesh").feature("imp").set("facepartition","minimal");
  m.component("c").mesh("mesh").run();m.component("c").geometricModel("mesh");
  m.component("c").sorder("linear");
  m.component("c").physics().create("solid","SolidMechanics","g");
  m.component("c").physics("solid").prop("ShapeProperty").set("order_displacement",@@ORDER@@);
  for(String prop:new String[]{"E","nu","rho"}) m.component("c").physics("solid").feature("lemm1").set(prop+"_mat","userdef");
  m.component("c").physics("solid").feature("lemm1").set("E","E0");
  m.component("c").physics("solid").feature("lemm1").set("nu","nu0");
  m.component("c").physics("solid").feature("lemm1").set("rho","rho0");
  @@BOUNDARIES@@
  m.component("c").mesh("mesh").feature("imp").set("filename","mesh.nas");
  m.save(ROOT+"model_unsolved.mph");
  double volume=0,energy=0,maxdisp=0;
  if(SOLVE) {
   System.out.println("TPMS_SOLVE_START"); m.study("std").run();
   m.result().numerical().create("integrals","IntVolume");m.result().numerical("integrals").selection().all();
   m.result().numerical("integrals").set("expr",new String[]{"1","solid.Ws"});
   m.result().numerical("integrals").set("unit",new String[]{"m^3","J"});
   double[][] v=m.result().numerical("integrals").getReal();volume=v[0][0];energy=v[1][0];
   m.result().numerical().create("umax","MaxVolume");m.result().numerical("umax").selection().all();
   m.result().numerical("umax").set("expr","solid.disp");m.result().numerical("umax").set("unit",new String[]{"m"});maxdisp=m.result().numerical("umax").getReal()[0][0];
   if(maxdisp<0.99*m.param().evaluate("uamp") || maxdisp>100*m.param().evaluate("uamp")) throw new RuntimeException("Displacement unit/magnitude check failed");
   if(!Double.isFinite(volume)||!Double.isFinite(energy)||!Double.isFinite(maxdisp)||energy<=0||maxdisp<=0) throw new RuntimeException("Invalid result");
   if(Math.abs(volume-@@VOL@@)/@@VOL@@>1e-6) throw new RuntimeException("Imported volume mismatch");
   m.result().create("pg1","PlotGroup3D");m.result("pg1").label("Displacement - static demonstration");
   m.result("pg1").create("surf1","Surface");m.result("pg1").feature("surf1").set("expr","solid.disp");
   m.save(ROOT+"model_solved.mph");
  }
  String saved=ROOT+(SOLVE?"model_solved.mph":"model_unsolved.mph");
  ModelUtil.remove("Model");m=ModelUtil.load("Check",saved);
  int tetra=m.component("c").mesh("mesh").getNumElem("tet");
  if(tetra!=@@TETS@@)throw new RuntimeException("Embedded mesh mismatch: "+tetra);
  if(SOLVE) {
   double[][] v=m.result().numerical("integrals").getReal();
   if(Math.abs(v[0][0]-volume)>Math.abs(volume)*1e-10||Math.abs(v[1][0]-energy)>Math.abs(energy)*1e-10) throw new RuntimeException("Reopen result mismatch");
  }
  String text="{\"import_verified\":true,\"reopen_verified\":true,\"solved\":"+SOLVE+",\"tetrahedra\":"+tetra+",\"volume_m3\":"+volume+",\"strain_energy_J\":"+energy+",\"max_displacement_m\":"+maxdisp+",\"element_order\":@@ORDER@@,\"convergence_verified\":false}";
  System.out.println("TPMS_REOPEN_VERIFIED "+text);ModelUtil.remove("Check");
 }
}"""
BOUNDARIES = r"""
  face(m,"bottom",0.0);face(m,"top",@@LZ@@);
  m.component("c").physics("solid").create("fix1","Fixed",2);
  m.component("c").physics("solid").feature("fix1").selection().named("bottom");
  m.component("c").physics("solid").create("disp1","Displacement2",2);
  m.component("c").physics("solid").feature("disp1").selection().named("top");
  m.component("c").physics("solid").feature("disp1").setIndex("Direction","prescribed",2);
  m.component("c").physics("solid").feature("disp1").setIndex("U0","-uamp",2);
  m.study().create("std");m.study("std").create("stat","Stationary");
"""

VERSION_PATTERN = re.compile(r"COMSOL(?: Multiphysics)?\s+(\d+(?:\.\d+){1,3})")


def detect_comsol():
    """Return the COMSOL ``bin/win64`` directory or None.

    ``COMSOL_BIN`` wins; otherwise the newest ``COMSOL*`` installation found
    under ``C:/Program Files/COMSOL`` and ``D:/Program Files/COMSOL`` is used.
    The directory actually used is recorded in every attempt's ``run.json``.
    """
    paths = [os.environ.get("COMSOL_BIN", "")]
    for parent in [Path("C:/Program Files/COMSOL"), Path("D:/Program Files/COMSOL")]:
        if parent.exists():
            installs = sorted((p for p in parent.iterdir() if p.is_dir()),
                              key=lambda p: p.name, reverse=True)
            paths += [str(p / "Multiphysics/bin/win64") for p in installs]
    for raw in paths:
        p = Path(raw)
        if (p / "comsolbatch.exe").is_file() and (p / "comsolcompile.exe").is_file():
            return p
    return None


def _validate_element_order(element_order):
    if type(element_order) is not int or element_order not in (1, 2):
        raise ValueError("element_order must be 1 (linear) or 2 (quadratic)")
    return element_order


def export_java(folder, report, solve=False, material=None, *, element_order=1):
    folder = Path(folder).resolve()
    element_order = _validate_element_order(element_order)
    if report["config"].get("domain_mode") == "solid_fluid":
        from .comsol_dual import export_dual_java

        return export_dual_java(folder, report, solve, element_order=element_order)
    material = material or {"E": 1.5e9, "nu": 0.3, "rho": 950.0}
    if any(
        type(material.get(k)) not in (int, float) or not np.isfinite(material[k])
        for k in ("E", "nu", "rho")
    ):
        raise ValueError("Material parameters must be finite numbers.")
    if not (
        1e3 <= material["E"] <= 1e13 and 0 <= material["nu"] < 0.49 and 0 < material["rho"] <= 1e6
    ):
        raise ValueError("Material parameters are outside the supported range.")
    size = report["config"]["size"]
    connected = report["volume_components"] == 1
    if solve and not connected:
        raise ValueError("The static elasticity demo requires connected material. Adjust the structure to remove disconnected solids.")
    text = TEMPLATE.replace("@@BOUNDARIES@@", BOUNDARIES if connected else "").replace("@@MESH_COMPAT@@", MESH_COMPAT)
    values = {
        "ROOT": json.dumps(folder.as_posix() + "/", ensure_ascii=True),
        "SOLVE": str(solve).lower(),
        "LX": repr(float(size[0])),
        "LY": repr(float(size[1])),
        "LZ": repr(float(size[2])),
        "TOL": repr(max(size) * 1e-7),
        "VOL": repr(report["volume_mm3"] * 1e-9),
        "TETS": str(report["tetrahedra"]),
        "E": repr(float(material["E"])),
        "NU": repr(float(material["nu"])),
        "RHO": repr(float(material["rho"])),
        "DISP": repr(float(size[2]) * 0.001),
        "ORDER": str(element_order),
    }
    for key, value in values.items():
        text = text.replace("@@" + key + "@@", value)
    (folder / "TPMSBuild.java").write_text(text, encoding="ascii")
    (folder / "simulation_settings.json").write_text(
        json.dumps(
            {
                "material": material,
                "test": "bottom fixed, top z displacement -0.001*Lz; lateral top free; linear geometry",
                "element_order": element_order,
                "displacement_shape_order": element_order,
                "solve_requested": solve,
            },
            indent=2,
        ),
        encoding="utf8",
    )
    return folder / "TPMSBuild.java"


# ---------------------------------------------------------------------------
# Shared launcher
# ---------------------------------------------------------------------------

def _validate_launch_limits(build_timeout_s, processes):
    if type(build_timeout_s) is not int or not 1 <= build_timeout_s <= 3600:
        raise ValueError("build_timeout_s must be an integer in [1, 3600]")
    if type(processes) is not int or not 1 <= processes <= 64:
        raise ValueError("processes must be an integer in [1, 64]")


@contextlib.contextmanager
def runtime_directory(folder):
    """Short native scratch/recovery/workspace paths for one COMSOL launch.

    COMSOL creates long native recovery filenames. A deep model directory can
    exceed Windows path limits even when the input NAS imports correctly, so
    scratch files go to the system temporary directory while every output
    stays in ``folder``. ``runtime.json`` records the location used.
    """
    with tempfile.TemporaryDirectory(prefix="tpms_", ignore_cleanup_errors=True) as temporary:
        runtime = Path(temporary).resolve()
        if runtime.parent != Path(tempfile.gettempdir()).resolve():
            raise RuntimeError("Unexpected COMSOL temporary directory")
        for name in ("tmp", "recovery", "workspace"):
            (runtime / name).mkdir()
        _write_json(Path(folder) / "runtime.json", {
            "directory": str(runtime), "temporary": True,
            "purpose": "short COMSOL scratch/recovery paths; outputs remain in the attempt directory"})
        yield runtime


def stage_preferences(preferences_directory, destination):
    """Copy explicit ``comsol.prefs``/``comsolserver.prefs`` next to one launch.

    Returns ``(local_directory, {"source": ..., "sha256": {...}})``. Global
    COMSOL preferences are never read or written by this package.
    """
    requested = Path(preferences_directory).resolve()
    from .io import file_sha256

    preference_files = [p for p in (requested / "comsol.prefs", requested / "comsolserver.prefs")
                        if p.is_file()]
    if not preference_files:
        raise ValueError("preferences_directory must contain comsol.prefs or comsolserver.prefs.")
    local = Path(destination)
    local.mkdir()
    for p in preference_files:
        shutil.copyfile(p, local / p.name)
    return local, {"source": str(requested),
                   "sha256": {p.name: file_sha256(p) for p in preference_files}}


def comsol_version_from_log(text):
    """Return the COMSOL version string printed in a batch log, or None."""
    match = VERSION_PATTERN.search(text or "")
    return match.group(1) if match else None


def run_comsol_class(folder, java, *, marker, build_timeout_s, progress=lambda s: None,
                     messages=("Compiling the COMSOL interface...", "Running COMSOL..."),
                     preferences_directory=None, runtime=None, processes=2,
                     compile_timeout_s=180):
    """Compile ``java`` with comsolcompile and run it with comsolbatch.

    ``compile.log``, ``build.log`` (stdout) and ``comsol.log`` (batch log) are
    written next to the class. The last ``marker {json}`` line in the logs is
    parsed and returned together with launch metadata. Raises RuntimeError on
    non-zero exit, a missing class file, a COMSOL security denial, or a
    missing marker.
    """
    _validate_launch_limits(build_timeout_s, processes)
    folder = Path(folder).resolve()
    java = Path(java)
    exe = detect_comsol()
    if not exe:
        raise RuntimeError("COMSOL was not found. Set COMSOL_BIN to the folder containing comsolbatch.exe and comsolcompile.exe.")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    options = []
    if preferences_directory is not None:
        options += ["-prefsdir", str(Path(preferences_directory).resolve()), "-migrateprefs", "off"]
    if runtime is not None:
        runtime = Path(runtime)
        options += ["-tmpdir", str(runtime / "tmp"), "-recoverydir", str(runtime / "recovery"),
                    "-data", str(runtime / "workspace"), "-autosave", "off"]
    commands = [
        ("compile", [str(exe / "comsolcompile.exe"), *options, str(java)], compile_timeout_s),
        ("build", [str(exe / "comsolbatch.exe"), *options, "-np", str(processes),
                   "-inputfile", str(java.with_suffix(".class")),
                   "-batchlog", str(folder / "comsol.log")], build_timeout_s),
    ]
    for (name, cmd, timeout), message in zip(commands, messages):
        progress(message)
        with (folder / (name + ".log")).open("w", encoding="utf8") as log:
            process = subprocess.Popen(
                cmd, cwd=folder, stdout=log, stderr=subprocess.STDOUT, creationflags=flags
            )
            try:
                code = process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=log, stderr=log, creationflags=flags,
                    )
                else:
                    process.kill()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                if isinstance(exc, KeyboardInterrupt):
                    raise
                raise RuntimeError(
                    "COMSOL timed out. Logs were retained; reduce the resolution or explicitly increase build_timeout_s and retry."
                )
        if code:
            raise RuntimeError(f"COMSOL {name} failed. See {name}.log and comsol.log.")
        if name == "compile" and not java.with_suffix(".class").is_file():
            raise RuntimeError(f"COMSOL compilation produced no {java.stem}.class, even though the launcher returned zero. See compile.log.")
    log_text = "\n".join(p.read_text(encoding="utf8", errors="replace")
                         for p in (folder / "build.log", folder / "comsol.log") if p.is_file())
    if "AccessControlException" in log_text:
        raise RuntimeError("COMSOL security denied file-system access. See comsol.log and the File system access preference before retrying this trusted workflow.")
    found = re.findall(re.escape(marker) + r" (\{[^\r\n]+\})", log_text)
    if not found:
        raise RuntimeError(
            f"COMSOL did not complete reopening verification ({marker} record missing). See build.log for possible licence, import, or solver errors."
        )
    result = json.loads(found[-1])
    info = {"comsol_bin": str(exe.resolve()), "comsol_version": comsol_version_from_log(log_text),
            "processes": processes, "marker_occurrences": len(found)}
    return result, info


def _execute_mph(folder, report, solve=False, material=None, progress=lambda s: None, *,
                 max_solve_tetrahedra=180_000, build_timeout_s=600, preferences_directory=None,
                 element_order=1, processes=2):
    with runtime_directory(folder) as runtime:
        return _execute_mph_runtime(folder, report, solve, material, progress,
                                    max_solve_tetrahedra=max_solve_tetrahedra,
                                    build_timeout_s=build_timeout_s,
                                    preferences_directory=preferences_directory,
                                    runtime_directory=runtime, element_order=element_order,
                                    processes=processes)


def _execute_mph_runtime(
    folder,
    report,
    solve=False,
    material=None,
    progress=lambda s: None,
    *,
    max_solve_tetrahedra=180_000,
    build_timeout_s=600,
    preferences_directory=None,
    runtime_directory=None,
    element_order=1,
    processes=2,
):
    if type(max_solve_tetrahedra) is not int or not 1 <= max_solve_tetrahedra <= 1_000_000:
        raise ValueError("max_solve_tetrahedra must be an integer in [1, 1000000]")
    _validate_launch_limits(build_timeout_s, processes)
    folder = Path(folder).resolve()
    if not detect_comsol():
        raise RuntimeError("COMSOL was not found. The NAS file is available; set COMSOL_BIN before creating an MPH model.")
    if solve and report["tetrahedra"] > max_solve_tetrahedra:
        raise ValueError(
            f"The demo solver limit is {max_solve_tetrahedra:,} volume elements. Reduce the resolution or explicitly set a research-specific limit."
        )
    java = export_java(folder, report, solve, material, element_order=element_order)
    result, info = run_comsol_class(
        folder, java, marker="TPMS_REOPEN_VERIFIED", build_timeout_s=build_timeout_s,
        progress=progress,
        messages=("Compiling the COMSOL interface...",
                  "Importing the mesh into COMSOL and solving the demo..." if solve
                  else "Creating and reopening the COMSOL mesh-defined model..."),
        preferences_directory=preferences_directory, runtime=runtime_directory,
        processes=processes,
    )
    if info.get("comsol_version"):
        result = {**result, "comsol_version": info["comsol_version"]}
    (folder / "comsol_verification.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    if result.get("solved") != solve:
        raise RuntimeError("The solution verification record does not match this request.")
    report.update(comsol_verified=True, comsol_solved=bool(result["solved"]), comsol=result)
    (folder / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8"
    )
    from .io import refresh_manifest

    refresh_manifest(folder)
    progress("The COMSOL model was reopened and verified.")
    return result


def _write_json(path, value):
    """Replace a small status record atomically; never serialize NaN."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                             encoding="utf8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_mph(folder, report, solve=False, material=None, progress=lambda s: None, *,
              max_solve_tetrahedra=180_000, build_timeout_s=600, preferences_directory=None,
              element_order=1, processes=2):
    """Run a fresh COMSOL attempt for a saved model, preserving previous runs.

    Logs, Java, settings and MPH files live in ``comsol_runs/<attempt_id>``.
    ``report.comsol_run`` describes the latest attempt; failure clears the
    current success flags. Returned evidence includes ``output_directory``.
    ``element_order=2`` requests quadratic displacement / P2+P1 flow shape
    functions; the shipped evidence was produced with the default of 1.
    """
    if type(solve) is not bool:
        raise ValueError("solve must be a boolean")
    if type(max_solve_tetrahedra) is not int or not 1 <= max_solve_tetrahedra <= 1_000_000:
        raise ValueError("max_solve_tetrahedra must be an integer in [1, 1000000]")
    _validate_launch_limits(build_timeout_s, processes)
    _validate_element_order(element_order)
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Saved model directory does not exist: {folder}")
    from .io import file_sha256, refresh_manifest

    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:12]
    attempt = folder / "comsol_runs" / attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    state = {"attempt_id": attempt_id, "status": "running", "solve_requested": solve,
             "started_utc": datetime.now(timezone.utc).isoformat(),
             "output_directory": str(attempt), "material_requested": material,
             "max_solve_tetrahedra": max_solve_tetrahedra,
             "build_timeout_s": build_timeout_s,
             "element_order": element_order, "processes": processes,
             "source_generator_version": report.get("generator_version"),
             "source_mesh_sha256": report.get("mesh_sha256")}
    report.update(comsol_verified=False, comsol_solved=False, comsol_run=state)
    report.pop("comsol", None)
    # Old top-level evidence must not masquerade as the current attempt.
    # Preserve it in the new attempt rather than deleting historical evidence.
    legacy = folder / "comsol_verification.json"
    try:
        if legacy.is_file():
            shutil.copyfile(legacy, attempt / "previous_top_level_verification.json")
            legacy.unlink()
        _write_json(attempt / "run.json", state)
        _write_json(folder / "report.json", report)
        exe = detect_comsol()
        if exe is None:
            raise RuntimeError("COMSOL was not found. Set COMSOL_BIN to the folder containing comsolbatch.exe and comsolcompile.exe.")
        state["comsol_bin"] = str(exe.resolve())
        mesh_path = folder / "mesh.nas"
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("A saved-model manifest is required before COMSOL import.")
        input_manifest = folder / "input_manifest.json"
        manifest = json.loads((input_manifest if input_manifest.is_file() else manifest_path).read_text(encoding="utf8"))
        recorded = manifest.get("mesh.nas", {}).get("sha256")
        actual = file_sha256(mesh_path)
        if not recorded or recorded != actual:
            raise ValueError("mesh.nas does not match its saved manifest; COMSOL import rejected.")
        state["input_nas_sha256"] = actual
        config_path = folder / "config.json"
        for name in ("config.json", "mesh.npz", "domains.json"):
            if name in manifest and file_sha256(folder / name) != manifest[name]["sha256"]:
                raise ValueError(f"{name} does not match its saved input manifest.")
        saved_config = json.loads(config_path.read_text(encoding="utf8"))
        if saved_config != json.loads(json.dumps(report.get("config"), allow_nan=False)):
            raise ValueError("The report configuration does not match config.json.")
        state["config_sha256"] = file_sha256(config_path)
        if not input_manifest.is_file():
            _write_json(input_manifest, {name: manifest[name] for name in (
                "mesh.nas", "config.json", "mesh.npz", "domains.json") if name in manifest})
        state["input_verified"] = True
        shutil.copyfile(mesh_path, attempt / "mesh.nas")
        shutil.copyfile(config_path, attempt / "config.json")
        if file_sha256(attempt / "mesh.nas") != actual:
            raise ValueError("Mesh changed while preparing COMSOL input.")
        if (folder / "domains.json").is_file():
            shutil.copyfile(folder / "domains.json", attempt / "domains.json")
        _write_json(attempt / "report.json", report)
        local_preferences = None
        if preferences_directory is not None:
            local_preferences, staged = stage_preferences(preferences_directory, attempt / "preferences")
            state["preferences_source"] = staged["source"]
            state["preferences_directory"] = str(local_preferences)
            state["preferences_input_sha256"] = staged["sha256"]
        result = _execute_mph(attempt, report, solve, material, progress,
                              max_solve_tetrahedra=max_solve_tetrahedra,
                              build_timeout_s=build_timeout_s, preferences_directory=local_preferences,
                              element_order=element_order, processes=processes)
        if result.get("import_verified") is not True or result.get("reopen_verified") is not True:
            raise RuntimeError("COMSOL evidence did not verify both import and reopening.")
        if result.get("solved") is not solve:
            raise RuntimeError("COMSOL evidence does not match the requested solve mode.")
        if result.get("comsol_version"):
            state["comsol_version"] = result["comsol_version"]
        result = {**result, "attempt_id": attempt_id, "output_directory": str(attempt),
                  "input_nas_sha256": actual, "config_sha256": state["config_sha256"]}
        report.update(comsol_verified=True, comsol_solved=solve, comsol=result)
        state["status"] = "succeeded"
        _write_json(attempt / "comsol_verification.json", result)
        _write_json(folder / "comsol_verification.json", result)
        return result
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                     error=f"{type(exc).__name__}: {exc}")
        report.update(comsol_verified=False, comsol_solved=False)
        report.pop("comsol", None)
        raise
    finally:
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        settings = attempt / "simulation_settings.json"
        if settings.is_file():
            state["simulation_settings_sha256"] = file_sha256(settings)
        _write_json(attempt / "run.json", state)
        _write_json(attempt / "report.json", report)
        _write_json(folder / "report.json", report)
        refresh_manifest(attempt)
        if state.get("input_verified") or (folder / "input_manifest.json").is_file():
            refresh_manifest(folder)


def run_comsol(directory, *, solve=True, material=None, progress=lambda s: None, **kwargs):
    """Load a saved Python export and create a separately recorded COMSOL run."""
    folder = Path(directory).resolve()
    report = json.loads((folder / "report.json").read_text(encoding="utf-8-sig"))
    return build_mph(folder, report, solve=solve, material=material, progress=progress, **kwargs)
