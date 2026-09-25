"""COMSOL 6.3 Java bridge. Generated MPH embeds the volume mesh.
No edits to user's existing MPH files; each export gets a new job directory.
"""

import json
import re
import os
from pathlib import Path
import subprocess
import numpy as np

TEMPLATE = r"""import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.*;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;
public class TPMSBuild {
 static String ROOT=@@ROOT@@;
 static boolean SOLVE=@@SOLVE@@;
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
  m.component("c").mesh().create("mesh","g");m.component("c").mesh("mesh").geometricModel("");
  m.component("c").mesh("mesh").create("imp","Import");
  m.component("c").mesh("mesh").feature("imp").set("source","nastran");
  m.component("c").mesh("mesh").feature("imp").set("filename",ROOT+"mesh.nas");
  m.component("c").mesh("mesh").feature("imp").set("data","mesh");
  m.component("c").mesh("mesh").feature("imp").set("allowshellpartition",false);
  m.component("c").mesh("mesh").feature("imp").set("facepartition","minimal");
  m.component("c").mesh("mesh").run();m.component("c").geometricModel("mesh");
  m.component("c").sorder("linear");
  m.component("c").physics().create("solid","SolidMechanics","g");
  m.component("c").physics("solid").prop("ShapeProperty").set("order_displacement",1);
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
   if(Math.abs(v[0][0]-volume)>1e-15||Math.abs(v[1][0]-energy)>Math.abs(energy)*1e-10) throw new RuntimeException("Reopen result mismatch");
  }
  String text="{\"import_verified\":true,\"reopen_verified\":true,\"solved\":"+SOLVE+",\"tetrahedra\":"+tetra+",\"volume_m3\":"+volume+",\"strain_energy_J\":"+energy+",\"max_displacement_m\":"+maxdisp+",\"convergence_verified\":false}";
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


def detect_comsol():
    paths = [
        os.environ.get("COMSOL_BIN", ""),
        r"D:/Program Files/COMSOL/COMSOL63/Multiphysics/bin/win64",
        r"C:/Program Files/COMSOL/COMSOL63/Multiphysics/bin/win64",
    ]
    for parent in [Path("C:/Program Files/COMSOL"), Path("D:/Program Files/COMSOL")]:
        if parent.exists():
            paths += [str(p / "Multiphysics/bin/win64") for p in parent.iterdir() if p.is_dir()]
    for raw in paths:
        p = Path(raw)
        if (p / "comsolbatch.exe").is_file() and (p / "comsolcompile.exe").is_file():
            return p
    return None


def export_java(folder, report, solve=False, material=None):
    folder = Path(folder).resolve()
    if report["config"].get("domain_mode") == "solid_fluid":
        from .comsol_dual import export_dual_java

        return export_dual_java(folder, report, solve)
    material = material or {"E": 1.5e9, "nu": 0.3, "rho": 950.0}
    if any(
        type(material.get(k)) not in (int, float) or not np.isfinite(material[k])
        for k in ("E", "nu", "rho")
    ):
        raise ValueError("材料参数必须是有限数。")
    if not (
        1e3 <= material["E"] <= 1e13 and 0 <= material["nu"] < 0.49 and 0 < material["rho"] <= 1e6
    ):
        raise ValueError("材料参数超出支持范围。")
    size = report["config"]["size"]
    connected = report["volume_components"] == 1
    if solve and not connected:
        raise ValueError("静力学演示要求材料连通；当前存在多个独立实体，请先调整结构。")
    text = TEMPLATE.replace("@@BOUNDARIES@@", BOUNDARIES if connected else "")
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
    }
    for key, value in values.items():
        text = text.replace("@@" + key + "@@", value)
    (folder / "TPMSBuild.java").write_text(text, encoding="ascii")
    (folder / "simulation_settings.json").write_text(
        json.dumps(
            {
                "material": material,
                "test": "bottom fixed, top z displacement -0.001*Lz; lateral top free; P1, linear geometry",
                "solve_requested": solve,
            },
            indent=2,
        ),
        encoding="utf8",
    )
    return folder / "TPMSBuild.java"


def build_mph(
    folder,
    report,
    solve=False,
    material=None,
    progress=lambda s: None,
    *,
    max_solve_tetrahedra=180_000,
):
    if type(max_solve_tetrahedra) is not int or not 1 <= max_solve_tetrahedra <= 1_000_000:
        raise ValueError("max_solve_tetrahedra must be an integer in [1, 1000000]")
    folder = Path(folder).resolve()
    exe = detect_comsol()
    if not exe:
        raise RuntimeError("未找到 COMSOL。已提供 NAS；请设置 COMSOL_BIN 后再创建 MPH。")
    if solve and report["tetrahedra"] > max_solve_tetrahedra:
        raise ValueError(
            f"演示求解上限为 {max_solve_tetrahedra:,} 个体单元；请降低采样数或明确设置研究用上限。"
        )
    java = export_java(folder, report, solve, material)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    for name, cmd, timeout in [
        ("compile", [str(exe / "comsolcompile.exe"), str(java)], 180),
        (
            "build",
            [
                str(exe / "comsolbatch.exe"),
                "-np",
                "2",
                "-inputfile",
                str(java.with_suffix(".class")),
                "-batchlog",
                str(folder / "comsol.log"),
            ],
            600,
        ),
    ]:
        progress(
            "编译 COMSOL 接口…"
            if name == "compile"
            else ("COMSOL 导入并执行演示求解…" if solve else "COMSOL 创建并重开实体模型…")
        )
        with (folder / (name + ".log")).open("w", encoding="utf8") as log:
            process = subprocess.Popen(
                cmd, cwd=folder, stdout=log, stderr=subprocess.STDOUT, creationflags=flags
            )
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=log,
                        stderr=log,
                        creationflags=flags,
                    )
                else:
                    process.kill()
                raise RuntimeError("COMSOL 运行超时。日志已保留；请降低分辨率后重试。")
        if code:
            raise RuntimeError(f"COMSOL {name} 失败，请查看 {name}.log 与 comsol.log。")
    evidence = folder / "comsol_verification.json"
    log_text = (folder / "build.log").read_text(encoding="utf8", errors="replace")
    found = re.findall(r"TPMS_REOPEN_VERIFIED (\{[^\r\n]+\})", log_text)
    if not found:
        raise RuntimeError(
            "COMSOL 未完成重开验证，请查看 build.log（可能是许可、导入或求解错误）。"
        )
    result = json.loads(found[-1])
    evidence.write_text(json.dumps(result, indent=2), encoding="utf8")
    if result.get("solved") != solve:
        raise RuntimeError("求解验证记录与本次请求不一致。")
    report.update(comsol_verified=True, comsol_solved=bool(result["solved"]), comsol=result)
    (folder / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8"
    )
    from .io import refresh_manifest

    refresh_manifest(folder)
    progress("COMSOL 模型已重新打开并验证。")
    return result
