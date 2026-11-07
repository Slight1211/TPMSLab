"""Optional COMSOL-native quality remeshing of a verified two-phase mesh.

The core Python mesh remains unchanged. COMSOL replaces its surface and
tetrahedral elements while preserving imported NASTRAN domain selections.
The result is accepted only after save/reopen, volume, interface-area, and
selection checks. This is a solver-dependent workflow, not a new tetrahedral
meshing algorithm or a guarantee of improved quality for every design.

The launch goes through :func:`tpmslab.comsol.run_comsol_class`, so the
6.2/6.3 mesh API reflection, isolated preference files and short scratch
paths behave exactly as in the elasticity/flow bridge. This mode has not yet
been exercised on a licensed COMSOL in the shipped evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from . import comsol as _comsol
from .comsol import MESH_COMPAT, run_comsol_class, runtime_directory, stage_preferences
from .quality import tetra_metrics
from . import __version__


JAVA = r'''import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.*;
import java.util.regex.*;
public class TPMSQualityRemesh {
 static String ROOT=@@ROOT@@;
 static double VOLUME_SIZE=@@VOLUME_SIZE@@;
 static double FACE_SIZE=@@FACE_SIZE@@;
 static String FACE_REP=@@FACE_REP@@;
 static int[] DOMAIN_PID=@@DOMAIN_PID@@;
 static int[] DOMAIN_PHASE=@@DOMAIN_PHASE@@;
 static int[] BOUNDARY_PID=@@BOUNDARY_PID@@;
 static double EXPECTED_BOX_VOLUME=@@BOX_VOLUME@@;
 @@MESH_COMPAT@@
 static int[] pid(Model m,int value,int dim){
  Pattern p=Pattern.compile("\\bID\\s+"+value+"\\b");
  for(String tag:m.component("c").selection().tags())
   if(p.matcher(m.component("c").selection(tag).label()).find()){
    int[] ids=m.component("c").selection(tag).entities(dim);
    if(ids.length>0)return ids;
   }
 throw new RuntimeException("Missing NASTRAN PID "+value+" at dimension "+dim);
 }
 static String faceKey(int a,int b,int c){
  int[] v=new int[]{a,b,c};Arrays.sort(v);
  return v[0]+","+v[1]+","+v[2];
 }
 static String audit(Model m){
  int[][] tet=m.component("c").mesh("mesh").getElem("tet");
  double[][] xyz=m.component("c").mesh("mesh").getVertex();
  int[] domain=m.component("c").mesh("mesh").getElemEntity("tet");
  if(tet.length!=4||tet[0].length<1||domain.length!=tet[0].length)
   throw new RuntimeException("Missing tetrahedra");
  double[] quality=new double[domain.length];double total=0;
  HashMap<Integer,Double> volumes=new HashMap<>();
  for(int i=0;i<domain.length;i++){
   double[][] p=new double[4][3];
   for(int k=0;k<4;k++)for(int j=0;j<3;j++)p[k][j]=xyz[j][tet[k][i]];
   double[] a=new double[3],b=new double[3],c=new double[3];
   for(int j=0;j<3;j++){a[j]=p[1][j]-p[0][j];b[j]=p[2][j]-p[0][j];c[j]=p[3][j]-p[0][j];}
   double v=Math.abs(a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0]))/6;
   if(!(v>0))throw new RuntimeException("Nonpositive tetrahedron volume");
   double e2=0;for(int u=0;u<4;u++)for(int w=u+1;w<4;w++)
    for(int j=0;j<3;j++)e2+=(p[u][j]-p[w][j])*(p[u][j]-p[w][j]);
   quality[i]=12*Math.pow(3*v,2.0/3)/e2;
   total+=v;volumes.put(domain[i],volumes.getOrDefault(domain[i],0.0)+v);
  }
  if(Math.abs(total-EXPECTED_BOX_VOLUME)>Math.max(1e-8,EXPECTED_BOX_VOLUME*1e-8))
   throw new RuntimeException("Total mesh volume mismatch: "+total);
  HashSet<Integer> seen=new HashSet<>(),solidIds=new HashSet<>();
  double[] domainVolumes=new double[DOMAIN_PID.length];
  for(int k=0;k<DOMAIN_PID.length;k++)for(int id:pid(m,DOMAIN_PID[k],3)){
   if(!seen.add(id))throw new RuntimeException("Overlapping domain selections");
   domainVolumes[k]+=volumes.getOrDefault(id,0.0);
   if(DOMAIN_PHASE[k]==1)solidIds.add(id);
  }
  if(seen.size()!=volumes.size())throw new RuntimeException("Unlabelled domain after remeshing");
  for(int id:BOUNDARY_PID)pid(m,id,2);
  int[] interfaceIds=pid(m,8,2);HashSet<Integer> faceSel=new HashSet<>();
  for(int id:interfaceIds)faceSel.add(id);
  int[][] tri=m.component("c").mesh("mesh").getElem("tri");
  int[] faceEntity=m.component("c").mesh("mesh").getElemEntity("tri");
  double interfaceArea=0;
  for(int i=0;i<faceEntity.length;i++)if(faceSel.contains(faceEntity[i])){
   double[] a=new double[3],b=new double[3];
   for(int j=0;j<3;j++){a[j]=xyz[j][tri[1][i]]-xyz[j][tri[0][i]];
                       b[j]=xyz[j][tri[2][i]]-xyz[j][tri[0][i]];}
   double x=a[1]*b[2]-a[2]*b[1],y=a[2]*b[0]-a[0]*b[2],z=a[0]*b[1]-a[1]*b[0];
   interfaceArea+=Math.sqrt(x*x+y*y+z*z)/2;
  }
  if(!(interfaceArea>0))throw new RuntimeException("Empty remeshed interface");
  HashMap<String,int[]> tetraFaces=new HashMap<>();
  int[][] localFaces={{0,1,2},{0,1,3},{0,2,3},{1,2,3}};
  for(int i=0;i<domain.length;i++){
   int phase=solidIds.contains(domain[i])?1:2;
   for(int[] f:localFaces){
    String key=faceKey(tet[f[0]][i],tet[f[1]][i],tet[f[2]][i]);
    int[] old=tetraFaces.get(key);
    if(old==null)tetraFaces.put(key,new int[]{phase,1});
    else{
     old[1]++;
     if(old[1]>2)throw new RuntimeException("Nonmanifold remeshed tetra face");
     if(old[0]!=phase)old[0]=3;
    }
   }
  }
  HashSet<String> crossPhaseFaces=new HashSet<>();
  for(Map.Entry<String,int[]> e:tetraFaces.entrySet())
   if(e.getValue()[1]==2&&e.getValue()[0]==3)crossPhaseFaces.add(e.getKey());
  HashSet<String> declaredInterface=new HashSet<>();
  for(int i=0;i<faceEntity.length;i++)if(faceSel.contains(faceEntity[i])){
   String key=faceKey(tri[0][i],tri[1][i],tri[2][i]);
   if(!declaredInterface.add(key))throw new RuntimeException("Duplicate remeshed interface triangle");
  }
  if(crossPhaseFaces.isEmpty()||!crossPhaseFaces.equals(declaredInterface))
   throw new RuntimeException("Remeshed cross-phase faces differ from declared interface");
  double[] solidQuality=new double[domain.length];int nSolid=0;
  for(int i=0;i<domain.length;i++)if(solidIds.contains(domain[i]))
   solidQuality[nSolid++]=quality[i];
  if(nSolid==0)throw new RuntimeException("Empty solid domain");
  solidQuality=Arrays.copyOf(solidQuality,nSolid);
  Arrays.sort(solidQuality);Arrays.sort(quality);
  StringBuilder dv=new StringBuilder("[");
  for(int k=0;k<domainVolumes.length;k++){
   if(k>0)dv.append(",");dv.append(domainVolumes[k]);
  }
  dv.append("]");
  return "{\"tetrahedra\":"+domain.length+
   ",\"solid_tetrahedra\":"+nSolid+
   ",\"domain_volumes_mm3\":"+dv+
   ",\"total_volume_mm3\":"+total+
   ",\"interface_area_mm2\":"+interfaceArea+
   ",\"interface_selection_entities\":"+interfaceIds.length+
   ",\"cross_phase_faces\":"+crossPhaseFaces.size()+
   ",\"interface_face_incidence_verified\":true"+
   ",\"q01_solid\":"+solidQuality[(int)Math.floor(.01*(nSolid-1))]+
   ",\"qmin_all\":"+quality[0]+"}";
 }
 public static void main(String[] args)throws Exception{
  Model m=ModelUtil.create("Model");m.modelPath(ROOT);m.label("TPMS Lab - quality remeshed solid and pore fluid");
  m.component().create("c",true);m.component("c").geom().create("g",3);
  m.component("c").geom("g").lengthUnit("mm");
  m.component("c").mesh().create("mesh","g");independentMesh(m);
  m.component("c").mesh("mesh").create("imp","Import");
  m.component("c").mesh("mesh").feature("imp").set("source","nastran");
  m.component("c").mesh("mesh").feature("imp").set("filename",ROOT+"mesh.nas");
  m.component("c").mesh("mesh").feature("imp").set("data","mesh");
  m.component("c").mesh("mesh").feature("imp").set("materialsplit",true);
  m.component("c").mesh("mesh").feature("imp").set("selcreation",true);
  m.component("c").mesh("mesh").feature("imp").set("allowshellpartition",false);
  m.component("c").mesh("mesh").feature("imp").set("facepartition","minimal");
  m.component("c").mesh("mesh").run("imp");
  m.component("c").mesh("mesh").create("del","DeleteEntities");
  m.component("c").mesh("mesh").feature("del").selection().geom("g",3);
  m.component("c").mesh("mesh").feature("del").selection().all();
  m.component("c").mesh("mesh").feature("del").set("unmesheddom",true);
  m.component("c").mesh("mesh").create("rmf","RemeshFaces");
  m.component("c").mesh("mesh").feature("rmf").selection().all();
  m.component("c").mesh("mesh").feature("rmf").set("facerep",FACE_REP);
  if(FACE_SIZE>0){
   m.component("c").mesh("mesh").feature("rmf").create("szf","Size");
   m.component("c").mesh("mesh").feature("rmf").feature("szf").set("hauto",FACE_SIZE);
  }
  m.component("c").mesh("mesh").create("ftet","FreeTet");
  m.component("c").mesh("mesh").feature("ftet").feature("size").set("hauto",VOLUME_SIZE);
  m.component("c").mesh("mesh").run();m.component("c").geometricModel("mesh");
  String initial=audit(m);System.out.println("TPMS_QUALITY_AUDIT "+initial);
  m.component("c").mesh("mesh").feature("imp").set("filename","mesh.nas");
  m.save(ROOT+"quality_remeshed.mph");ModelUtil.remove("Model");
  m=ModelUtil.load("Check",ROOT+"quality_remeshed.mph");
  String reopened=audit(m);ModelUtil.remove("Check");
  if(!initial.equals(reopened))throw new RuntimeException("Reopened mesh audit mismatch");
  System.out.println("TPMS_QUALITY_REOPEN_VERIFIED "+reopened);
 }
}'''


def _solid_quality(folder: Path) -> float:
    with np.load(folder / "mesh.npz") as mesh:
        phase = mesh["phase_ids"] == 1
        _, quality = tetra_metrics(mesh["points_mm"][mesh["tetrahedra"][phase]])
    return float(np.quantile(quality, 0.01))


def remesh_in_comsol(
    source_folder,
    output_folder,
    *,
    volume_size=7.0,
    face_size=0.0,
    face_representation="curved",
    max_domain_volume_shift_pp=0.3,
    max_interface_area_shift_percent=2.0,
    temp_directory=None,
    timeout_s=900,
    preferences_directory=None,
    processes=2,
):
    """Create a separately audited COMSOL quality mesh in a new directory.

    The returned model is mesh-defined and needs a licensed COMSOL (6.2 or
    newer API via reflection) Windows installation. ``face_size=0`` uses
    COMSOL's default face sizing. ``temp_directory`` is accepted for
    compatibility; scratch, recovery and workspace paths are always short
    temporary directories recorded in ``runtime.json``. Failed audits retain
    their logs and MPH for investigation and raise.
    """
    source = Path(source_folder).resolve()
    output = Path(output_folder).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not all((source / name).is_file() for name in ("mesh.nas", "mesh.npz", "report.json")):
        raise FileNotFoundError("Source requires mesh.nas, mesh.npz, and report.json")
    report = json.loads((source / "report.json").read_text(encoding="utf8"))
    if report.get("config", {}).get("domain_mode") != "solid_fluid":
        raise ValueError("Quality remeshing requires a verified solid_fluid mesh")
    if not report.get("interface_conforming") or not report.get("domain_map"):
        raise ValueError("Source two-phase interface and domain metadata were not verified")
    for name, value in (("volume_size", volume_size), ("face_size", face_size)):
        if type(value) not in (int, float) or not np.isfinite(value) or not (1 <= value <= 9 if name == "volume_size" else 0 <= value <= 9):
            raise ValueError(f"{name} must be a finite COMSOL size between {'1' if name == 'volume_size' else '0'} and 9")
    if face_representation not in ("curved", "linear"):
        raise ValueError("face_representation must be curved or linear")
    if type(timeout_s) is not int or not 1 <= timeout_s <= 3600:
        raise ValueError("timeout_s must be an integer in [1, 3600]")
    for name, value in (("max_domain_volume_shift_pp", max_domain_volume_shift_pp),
                        ("max_interface_area_shift_percent", max_interface_area_shift_percent)):
        if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
    exe = _comsol.detect_comsol()  # module attribute so tests can stub it
    if not exe:
        raise RuntimeError("COMSOL is required for the optional quality-remesh mode")
    if temp_directory is not None:
        Path(temp_directory).resolve().mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True)
    shutil.copyfile(source / "mesh.nas", output / "mesh.nas")
    domains = report["domain_map"]
    domain_pids = [int(d["nastran_pid"]) for d in domains]
    domain_phase = [1 if d["phase"] == "solid" else 2 for d in domains]
    boundary_pids = sorted({int(p) for d in domains for p in d["exterior_boundary_ids"]})
    box_volume = float(np.prod(report["config"]["size"]))
    values = {
        "ROOT": json.dumps(output.as_posix() + "/"),
        "VOLUME_SIZE": str(float(volume_size)),
        "FACE_SIZE": str(float(face_size)),
        "FACE_REP": json.dumps(face_representation),
        "DOMAIN_PID": "new int[]{" + ",".join(map(str, domain_pids)) + "}",
        "DOMAIN_PHASE": "new int[]{" + ",".join(map(str, domain_phase)) + "}",
        "BOUNDARY_PID": "new int[]{" + ",".join(map(str, boundary_pids)) + "}",
        "BOX_VOLUME": repr(box_volume),
        "MESH_COMPAT": MESH_COMPAT,
    }
    java = JAVA
    for key, value in values.items():
        java = java.replace("@@" + key + "@@", value)
    java_path = output / "TPMSQualityRemesh.java"
    java_path.write_text(java, encoding="ascii")
    local_preferences = None
    launch = {}
    if preferences_directory is not None:
        local_preferences, staged = stage_preferences(preferences_directory, output / "preferences")
        launch["preferences_source"] = staged["source"]
        launch["preferences_input_sha256"] = staged["sha256"]
    with runtime_directory(output) as runtime:
        result, info = run_comsol_class(
            output, java_path, marker="TPMS_QUALITY_REOPEN_VERIFIED", build_timeout_s=timeout_s,
            messages=("Compiling the COMSOL remesh interface...", "Remeshing in COMSOL..."),
            preferences_directory=local_preferences, runtime=runtime, processes=processes,
        )
    if info["marker_occurrences"] != 1:
        raise RuntimeError(f"COMSOL remesh was not reopened and audited exactly once; see {output}")
    launch.update(comsol_bin=info["comsol_bin"], comsol_version=info["comsol_version"],
                  processes=processes)
    old_q01 = _solid_quality(source)
    if result["q01_solid"] <= old_q01:
        raise RuntimeError("Remeshed solid quality did not improve; model retained for inspection")
    old_volumes = [float(d["volume_mm3"]) for d in domains]
    changes_pp = [abs(a - b) / box_volume * 100
                  for a, b in zip(result["domain_volumes_mm3"], old_volumes)]
    if len(changes_pp) != len(domains) or max(changes_pp) > max_domain_volume_shift_pp:
        raise RuntimeError("Remeshing changed a phase-domain volume beyond the allowed tolerance")
    area_shift = abs(result["interface_area_mm2"] - report["interface_area_mm2"]) / report["interface_area_mm2"] * 100
    if area_shift > max_interface_area_shift_percent:
        raise RuntimeError("Remeshing changed interface area beyond the allowed tolerance")
    evidence = {
        **result,
        "verified": True,
        "quality_mode": "COMSOL RemeshFaces + FreeTet",
        "source_generator_version": report["generator_version"],
        "quality_bridge_version": __version__,
        "source_mesh_sha256": hashlib.sha256((source / "mesh.nas").read_bytes()).hexdigest(),
        "quality_p01_solid_before": old_q01,
        "domain_volume_shift_pp": changes_pp,
        "interface_area_shift_percent": area_shift,
        "volume_size": float(volume_size),
        "face_size": float(face_size),
        "face_representation": face_representation,
        "comsol_dependency": True,
        **launch,
    }
    (output / "quality_remesh_verification.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf8"
    )
    return evidence


def main():
    parser = argparse.ArgumentParser(description="Remesh a verified TPMS Lab two-phase model in COMSOL")
    parser.add_argument("source_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--volume-size", type=float, default=7.0)
    parser.add_argument("--face-size", type=float, default=0.0)
    parser.add_argument("--face-representation", choices=("curved", "linear"), default="curved")
    parser.add_argument("--temp-directory", type=Path)
    parser.add_argument("--preferences-directory", type=Path)
    args = parser.parse_args()
    result = remesh_in_comsol(args.source_folder, args.output_folder,
                              volume_size=args.volume_size, face_size=args.face_size,
                              face_representation=args.face_representation,
                              temp_directory=args.temp_directory,
                              preferences_directory=args.preferences_directory)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
