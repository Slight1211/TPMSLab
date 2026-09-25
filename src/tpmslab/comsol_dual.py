"""COMSOL solid/fluid selections and optional fixed-wall creeping-flow smoke test."""

import json
from pathlib import Path

TEMPLATE = r"""import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.*;
import java.util.regex.*;
public class TPMSBuild {
 static String ROOT=@@ROOT@@;
 static boolean SOLVE=@@SOLVE@@;
 static void select(Model m,String name,int dim,int[] ids) {
  m.component("c").selection().create(name,"Explicit");
  m.component("c").selection(name).geom("g",dim);
  m.component("c").selection(name).set(ids);
  m.component("c").selection(name).label(name);
 }
 static int[] pid(Model m,int value,int dim) {
  Pattern p=Pattern.compile("\\bID\\s+"+value+"\\b");
  for(String tag:m.component("c").selection().tags()) {
   if(p.matcher(m.component("c").selection(tag).label()).find()) {
    int[] ids=m.component("c").selection(tag).entities(dim);
    if(ids.length>0)return ids;
   }
  }
  throw new RuntimeException("Missing NASTRAN PID "+value+" dimension "+dim);
 }
 static int[] join(List<Integer> ids) {return ids.stream().mapToInt(Integer::intValue).distinct().toArray();}
 static void verify(Model m) {
  if(m.component("c").mesh("mesh").getNumElem("tet")!=@@TETS@@)throw new RuntimeException("Tetrahedron count mismatch");
  int[] es=m.component("c").mesh("mesh").getElemEntity("tet");
  int[][] ts=m.component("c").mesh("mesh").getElem("tet");
  double[][] ps=m.component("c").mesh("mesh").getVertex();
  HashMap<Integer,Double> vol=new HashMap<>();
  for(int i=0;i<es.length;i++) {
   double[] a=new double[3],b=new double[3],c=new double[3];
   for(int j=0;j<3;j++){a[j]=ps[j][ts[1][i]]-ps[j][ts[0][i]];b[j]=ps[j][ts[2][i]]-ps[j][ts[0][i]];c[j]=ps[j][ts[3][i]]-ps[j][ts[0][i]];}
   double v=Math.abs(a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0]))/6;
   vol.put(es[i],vol.getOrDefault(es[i],0.0)+v);
  }
  @@CHECK_VOLUMES@@
  if(m.component("c").selection("solid_domains").entities(3).length!=@@SOLID_COUNT@@ || m.component("c").selection("fluid_domains").entities(3).length!=@@FLUID_COUNT@@)throw new RuntimeException("Phase selection mismatch");
  if(m.component("c").selection("solid_fluid_interface").entities(2).length==0)throw new RuntimeException("Missing interface");
 }
 static double integral(Model m,String tag,String selection,String expr) {
  m.result().numerical().create(tag,"IntSurface");
  m.result().numerical(tag).selection().named(selection);
  m.result().numerical(tag).set("expr",expr);
  m.result().numerical(tag).set("unit","m^3/s");
  return m.result().numerical(tag).getReal()[0][0];
 }
 public static void main(String[] args)throws Exception {
  Model m=ModelUtil.create("Model");m.modelPath(ROOT);m.label("TPMS Lab - solid and pore fluid domains");
  m.component().create("c",true);m.component("c").geom().create("g",3);m.component("c").geom("g").lengthUnit("mm");
  m.component("c").mesh().create("mesh","g");m.component("c").mesh("mesh").geometricModel("");
  m.component("c").mesh("mesh").create("imp","Import");
  m.component("c").mesh("mesh").feature("imp").set("source","nastran");
  m.component("c").mesh("mesh").feature("imp").set("filename",ROOT+"mesh.nas");
  m.component("c").mesh("mesh").feature("imp").set("data","mesh");
  m.component("c").mesh("mesh").feature("imp").set("materialsplit",true);
  m.component("c").mesh("mesh").feature("imp").set("selcreation",true);
  m.component("c").mesh("mesh").feature("imp").set("allowshellpartition",false);
  m.component("c").mesh("mesh").feature("imp").set("facepartition","minimal");
  m.component("c").mesh("mesh").run();m.component("c").geometricModel("mesh");
  List<Integer> solid=new ArrayList<>(),fluid=new ArrayList<>();
  @@SELECTIONS@@
  select(m,"solid_domains",3,join(solid));select(m,"fluid_domains",3,join(fluid));
  verify(m);
  m.component("c").material().create("mat_s","Common");m.component("c").material("mat_s").label("Solid - demonstration properties");
  m.component("c").material("mat_s").selection().named("solid_domains");
  m.component("c").material("mat_s").propertyGroup("def").set("youngsmodulus","1.5e9[Pa]");
  m.component("c").material("mat_s").propertyGroup("def").set("poissonsratio","0.3");
  m.component("c").material("mat_s").propertyGroup("def").set("density","950[kg/m^3]");
  m.component("c").material().create("mat_f","Common");m.component("c").material("mat_f").label("Pore fluid - demonstration properties");
  m.component("c").material("mat_f").selection().named("fluid_domains");
  m.component("c").material("mat_f").propertyGroup("def").set("density","1000[kg/m^3]");
  m.component("c").material("mat_f").propertyGroup("def").set("dynamicviscosity","0.001[Pa*s]");
  if(SOLVE) {
   m.component("c").sorder("linear");
   m.component("c").physics().create("spf","CreepingFlow","g");
   m.component("c").physics("spf").selection().named("fluid_domains");
   m.component("c").physics("spf").create("inl1","InletBoundary",2);
   m.component("c").physics("spf").feature("inl1").selection().named("fluid_zmin");
   m.component("c").physics("spf").feature("inl1").set("BoundaryCondition","Pressure");
   m.component("c").physics("spf").feature("inl1").set("p0","0.01[Pa]");
   m.component("c").physics("spf").create("out1","OutletBoundary",2);
   m.component("c").physics("spf").feature("out1").selection().named("fluid_zmax");
   m.component("c").physics("spf").feature("out1").set("p0","0[Pa]");
   m.study().create("std");m.study("std").create("stat","Stationary");
  }
  m.component("c").mesh("mesh").feature("imp").set("filename","mesh.nas");
  m.save(ROOT+"model_unsolved.mph");
  double qin=0,qout=0,balance=0;
  if(SOLVE) {
   m.study("std").run();
   qin=integral(m,"qin","fluid_zmin","-w");qout=integral(m,"qout","fluid_zmax","w");
   balance=Math.abs(qin+qout)/Math.max(Math.abs(qin),Math.abs(qout));
   if(!Double.isFinite(qin)||!Double.isFinite(qout)||!(qin<0)||!(qout>0)||balance>0.05)throw new RuntimeException("Flow or conservation check failed: "+qin+" "+qout+" "+balance);
   m.result().create("pg1","PlotGroup3D");m.result("pg1").label("Pore fluid speed - fixed-wall creeping-flow demonstration");
   m.result("pg1").create("surf1","Surface");m.result("pg1").feature("surf1").set("expr","spf.U");
   m.save(ROOT+"model_solved.mph");
  }
  ModelUtil.remove("Model");m=ModelUtil.load("Check",ROOT+(SOLVE?"model_solved.mph":"model_unsolved.mph"));verify(m);
  if(SOLVE && Math.abs(m.result().numerical("qout").getReal()[0][0]-qout)>Math.abs(qout)*1e-10)throw new RuntimeException("Reopened flow result mismatch");
  String result="{\"import_verified\":true,\"reopen_verified\":true,\"phase_volumes_verified\":true,\"solid_domains\":@@SOLID_COUNT@@,\"fluid_domains\":@@FLUID_COUNT@@,\"interface_selection_verified\":true,\"solved\":"+SOLVE+",\"test\":\"fixed_wall_creeping_flow\",\"inlet_outward_flux_m3_s\":"+qin+",\"outlet_flux_m3_s\":"+qout+",\"relative_flux_imbalance\":"+balance+",\"fsi_verified\":false,\"convergence_verified\":false}";
  System.out.println("TPMS_REOPEN_VERIFIED "+result);ModelUtil.remove("Check");
 }
}"""


def export_dual_java(folder, report, solve=False):
    folder = Path(folder).resolve()
    if solve and any(
        not {16, 17}.issubset(d["exterior_boundary_ids"])
        for d in report["domain_map"]
        if d["phase"] == "fluid"
    ):
        raise ValueError(
            "流动演示要求每个流体域都连通 Z- 和 Z+；请创建未求解 MPH 后按各流道设置边界。"
        )
    selections, checks = [], []
    for d in report["domain_map"]:
        var = f"d{d['id']}"
        selections.append(
            f'int[] {var}=pid(m,{d["nastran_pid"]},3); select(m,"{d["phase"]}_{d["id"]}",3,{var}); for(int id:{var}) {d["phase"]}.add(id);'
        )
        checks.append(
            f'double v{d["id"]}=0; for(int id:m.component("c").selection("{d["phase"]}_{d["id"]}").entities(3)) v{d["id"]}+=vol.getOrDefault(id,0.0); if(Math.abs(v{d["id"]}-{d["volume_mm3"]})>{max(d["volume_mm3"] * 1e-8, 1e-10)})throw new RuntimeException("Volume mismatch for {var}: "+v{d["id"]});'
        )
    boundary_ids = {8}
    for d in report["domain_map"]:
        boundary_ids.update(d["exterior_boundary_ids"])
    for pid in sorted(boundary_ids):
        selections.append(f'select(m,"{report["boundary_tag_names"][str(pid)]}",2,pid(m,{pid},2));')
    text = TEMPLATE
    values = dict(
        ROOT=json.dumps(folder.as_posix() + "/"),
        SOLVE=str(solve).lower(),
        TETS=str(report["tetrahedra"]),
        SOLID_COUNT=str(report["solid_components"]),
        FLUID_COUNT=str(report["fluid_components"]),
        SELECTIONS="\n  ".join(selections),
        CHECK_VOLUMES="\n  ".join(checks),
    )
    for key, value in values.items():
        text = text.replace("@@" + key + "@@", value)
    (folder / "TPMSBuild.java").write_text(text, encoding="ascii")
    (folder / "simulation_settings.json").write_text(
        json.dumps(
            dict(
                test="fixed-wall creeping flow in pore fluid only; no deformation/FSI",
                solve_requested=solve,
                inlet="fluid_zmin, 0.01 Pa",
                outlet="fluid_zmax, 0 Pa",
                walls="no slip on solid-fluid interface and other fluid exterior faces",
                fluid_density_kg_m3=1000,
                fluid_dynamic_viscosity_Pa_s=0.001,
                convergence_verified=False,
            ),
            indent=2,
        ),
        encoding="utf8",
    )
    return folder / "TPMSBuild.java"
