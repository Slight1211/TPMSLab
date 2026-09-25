"""Local-only engineering UI and CLI; all assets and model data stay local."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import io
import json
import logging
import os
from pathlib import Path
import secrets
import threading
import time
import uuid
import webbrowser
import zipfile
import numpy as np
import trimesh
from flask import Flask, jsonify, request, send_file, send_from_directory, abort
from werkzeug.serving import make_server
from tpmslab.model import Config, FAMILIES
from tpmslab.volume import generate_volume
from tpmslab.comsol import build_mph, detect_comsol, export_java

ROOT = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("TPMSLAB_OUTPUT_DIR", str(Path.cwd() / "tpmslab-output")))
RUNS.mkdir(parents=True, exist_ok=True)
app = Flask(__name__, static_folder=str(ROOT / "static"))
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
TOKEN = secrets.token_hex(24)
JOBS = {}
LOCK = threading.Lock()
POOL = ThreadPoolExecutor(max_workers=1)
BUSY = False


@app.before_request
def local_only():
    if request.host.split(":")[0] not in ("127.0.0.1", "localhost"):
        abort(403)
    if request.method == "POST" and request.headers.get("X-TPMS-Token") != TOKEN:
        abort(403)


@app.after_request
def headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def home():
    return (ROOT / "static/index.html").read_text(encoding="utf8").replace("__TOKEN__", TOKEN)


@app.get("/api/info")
def info():
    return jsonify(
        families=FAMILIES,
        defaults=Config().to_dict(),
        comsol=bool(detect_comsol()),
        runs_path=str(RUNS),
        latest_job=next(reversed(JOBS), None),
    )


def get_job(key):
    if key not in JOBS:
        abort(404)
    return JOBS[key]


def public_job(job):
    return {k: v for k, v in job.items() if k not in ("model", "folder")}


def progress(job, message):
    job["message"] = message
    job.setdefault("events", []).append(message)
    job["events"] = job["events"][-12:]


def save_model(model, folder):
    from .io import save_model as export_model, refresh_manifest

    export_model(model, folder)
    export_java(folder, model["report"])
    refresh_manifest(folder)


def generate_task(job, config):
    global BUSY
    try:
        model = generate_volume(config, lambda msg: progress(job, msg))
        save_model(model, job["folder"])
        job.update(model=model, report=model["report"], status="ready")
        progress(job, "实体计算域已生成，可导出 NAS 或创建 MPH。")
    except Exception as exc:
        logging.exception("generation failed")
        job.update(status="error", error=str(exc))
        progress(job, str(exc))
    finally:
        with LOCK:
            BUSY = False


@app.post("/api/generate")
def generate():
    global BUSY
    try:
        config = Config.from_dict(request.get_json())
    except (ValueError, TypeError, SyntaxError) as exc:
        return jsonify(error=str(exc)), 400
    with LOCK:
        if BUSY:
            return jsonify(error="已有任务在运行，请等待完成。"), 409
        BUSY = True
        # Free old in-memory meshes; generated files remain on disk.
        for old in JOBS.values():
            old.pop("model", None)
        key = uuid.uuid4().hex[:16]
        folder = RUNS / (time.strftime("%Y%m%d-%H%M%S") + "_" + key)
        job = {
            "id": key,
            "status": "working",
            "message": "开始生成…",
            "folder": folder,
            "events": [],
        }
        JOBS[key] = job
    POOL.submit(generate_task, job, config)
    return jsonify(id=key)


@app.get("/api/job/<key>")
def state(key):
    return jsonify(public_job(get_job(key)))


@app.get("/api/preview/<key>")
def preview(key):
    job = get_job(key)
    if "model" not in job:
        abort(404)
    # Flat triangle positions, little-endian float32, millimetres.
    phase = request.args.get("phase", "solid")
    if phase not in ("solid", "fluid"):
        abort(400)
    surface_key = "fluid_surface" if phase == "fluid" else "surface"
    if surface_key not in job["model"]:
        abort(404)
    data = np.asarray(job["model"][surface_key].triangles, dtype="<f4").tobytes()
    return send_file(io.BytesIO(data), mimetype="application/octet-stream")


def comsol_task(job, solve, material):
    global BUSY
    try:
        report = job["report"]
        evidence = job["folder"] / "comsol_verification.json"
        if evidence.exists():
            evidence.unlink()
        build_mph(job["folder"], report, solve, material, lambda msg: progress(job, msg))
        job.update(status="ready", report=report)
    except Exception as exc:
        logging.exception("COMSOL operation failed")
        job.update(status="ready", operation_error=str(exc))
        progress(job, str(exc))
    finally:
        with LOCK:
            BUSY = False


@app.post("/api/comsol/<key>")
def comsol(key):
    global BUSY
    job = get_job(key)
    data = request.get_json() or {}
    solve = data.get("solve", False)
    if type(solve) is not bool:
        return jsonify(error="solve 必须为布尔值"), 400
    if "report" not in job:
        return jsonify(error="请先生成模型。"), 400
    if (
        solve
        and job["report"]["config"].get("domain_mode") != "solid_fluid"
        and job["report"]["volume_components"] != 1
    ):
        return jsonify(error="静力学演示要求单连通材料，请调整相位、密度或胞元。"), 400
    if not detect_comsol():
        return jsonify(error="未找到 COMSOL，请配置 COMSOL_BIN。"), 400
    with LOCK:
        if BUSY:
            return jsonify(error="已有任务在运行。"), 409
        BUSY = True
        job.update(status="working")
        job.pop("operation_error", None)
    POOL.submit(comsol_task, job, solve, data.get("material"))
    return jsonify(id=key)


ALLOWED = {
    "fluid_preview.stl",
    "domains.json",
    "mesh.nas",
    "mesh.npz",
    "preview.stl",
    "config.json",
    "report.json",
    "TPMSBuild.java",
    "model_unsolved.mph",
    "model_solved.mph",
    "comsol_verification.json",
    "comsol.log",
    "compile.log",
    "build.log",
    "simulation_settings.json",
    "environment.json",
    "manifest.json",
}


@app.get("/api/download/<key>/<name>")
def download(key, name):
    job = get_job(key)
    if job["status"] == "working":
        abort(409)
    if name == "package.zip":
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
            for p in job["folder"].iterdir():
                if p.name in ALLOWED:
                    z.write(p, p.name)
        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name="TPMS_" + key + ".zip",
            mimetype="application/zip",
        )
    if name not in ALLOWED:
        abort(404)
    return send_from_directory(job["folder"], name, as_attachment=True)


def restore_latest():
    for folder in sorted(RUNS.iterdir(), reverse=True):
        if (
            not folder.is_dir()
            or not (folder / "report.json").exists()
            or not (folder / "mesh.npz").exists()
        ):
            continue
        try:
            report = json.loads((folder / "report.json").read_text(encoding="utf8"))
            with np.load(folder / "mesh.npz", allow_pickle=False) as archive:
                model = {
                    "points": archive["points_mm"],
                    "tetra": archive["tetrahedra"],
                    "boundary": archive["boundary"],
                    "boundary_ids": archive["boundary_ids"],
                    "domains": archive["domain_ids"],
                    "report": report,
                }
                for name in ("phase_ids", "interface", "solid_boundary", "fluid_boundary"):
                    if name in archive:
                        model[name] = archive[name]
            if "fluid_boundary" in model:
                model["fluid_surface"] = trimesh.Trimesh(
                    model["points"].copy(), model["fluid_boundary"].copy(), process=False
                )
                model["fluid_surface"].remove_unreferenced_vertices()
            model["surface"] = trimesh.Trimesh(
                model["points"].copy(),
                model.get("solid_boundary", model["boundary"]).copy(),
                process=False,
            )
            model["surface"].remove_unreferenced_vertices()
            key = folder.name.rsplit("_", 1)[-1]
            JOBS[key] = {
                "id": key,
                "folder": folder,
                "model": model,
                "report": report,
                "status": "ready",
                "message": "已恢复最近生成的模型。",
                "events": ["已恢复本机最近一次生成结果。"],
            }
            return
        except Exception:
            logging.exception("Could not restore previous model")


def serve(port=0, no_browser=False):
    restore_latest()
    server = make_server("127.0.0.1", port, app, threaded=True)
    url = f"http://127.0.0.1:{server.server_port}"
    (RUNS / "last_session.json").write_text(json.dumps({"url": url, "pid": os.getpid()}))
    print("TPMS Lab: " + url, flush=True)
    if not no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    server.serve_forever()
