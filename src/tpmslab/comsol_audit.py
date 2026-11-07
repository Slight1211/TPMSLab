"""Read-only audits of saved COMSOL pore-flow solutions.

The launch uses the shared :func:`tpmslab.comsol.run_comsol_class` so that
isolated preference files and short scratch paths apply here as well.
"""

import json
import hashlib
from pathlib import Path
from . import comsol as _comsol
from .comsol import run_comsol_class, runtime_directory, stage_preferences

JAVA = r"""import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.*;
public class TPMSAudit {
 static double integral(Model m,String tag,String selection,String expr,String unit) {
  m.result().numerical().create(tag,"IntSurface");
  m.result().numerical(tag).selection().named(selection);
  m.result().numerical(tag).set("expr",expr);
  m.result().numerical(tag).set("unit",unit);
  return m.result().numerical(tag).getReal()[0][0];
 }
 static String intersect(Model m,String tag,int[] adjacent,String plane) {
  HashSet<Integer> a=new HashSet<>();for(int i:adjacent)a.add(i);
  int[] ids=Arrays.stream(m.component("c").selection(plane).entities(2)).filter(a::contains).toArray();
  if(ids.length==0)throw new RuntimeException("Empty channel opening "+tag);
  m.component("c").selection().create(tag,"Explicit");m.component("c").selection(tag).geom("g",2);m.component("c").selection(tag).set(ids);return tag;
 }
 public static void main(String[] args)throws Exception {
  Model m=ModelUtil.load("Audit",@@MODEL@@);
  double area=integral(m,"auditArea","solid_fluid_interface","1","mm^2");
  double eta=Math.abs(area-@@AREA@@)/@@AREA@@;
  if(!Double.isFinite(area)||eta>1e-8)throw new RuntimeException("Interface area mismatch: "+area);
  StringBuilder rows=new StringBuilder(); double sumIn=0,sumOut=0;
  int[] domainIds=new int[]{@@IDS@@};
  for(int id:domainIds) {
   String adj="auditAdj"+id;m.component("c").selection().create(adj,"Adjacent");
   m.component("c").selection(adj).set("entitydim",3);m.component("c").selection(adj).set("outputdim",2);
   m.component("c").selection(adj).set("input",new String[]{"fluid_"+id});
   int[] boundaries=m.component("c").selection(adj).entities(2);
   String in=intersect(m,"auditIn"+id,boundaries,"fluid_zmin"),out=intersect(m,"auditOut"+id,boundaries,"fluid_zmax");
   double ai=integral(m,"auditAi"+id,in,"1","mm^2"),ao=integral(m,"auditAo"+id,out,"1","mm^2");
   double qi=integral(m,"auditQi"+id,in,"-w","m^3/s"),qo=integral(m,"auditQo"+id,out,"w","m^3/s");
   double balance=Math.abs(qi+qo)/Math.max(Math.abs(qi),Math.abs(qo));
   if(!Double.isFinite(balance)||!(qi<0)||!(qo>0)||balance>0.05)throw new RuntimeException("Channel balance failed: "+id+" "+balance);
   if(rows.length()>0)rows.append(",");
   rows.append("{\"domain_id\":"+id+",\"inlet_area_mm2\":"+ai+",\"outlet_area_mm2\":"+ao+",\"inlet_outward_flux_m3_s\":"+qi+",\"outlet_flux_m3_s\":"+qo+",\"relative_flux_imbalance\":"+balance+",\"solver_domain_ids\":"+Arrays.toString(m.component("c").selection("fluid_"+id).entities(3))+"}");
   sumIn+=qi;sumOut+=qo;
  }
  System.out.println("TPMS_AUDIT {\"interface_area_comsol_mm2\":"+area+",\"interface_area_relative_error\":"+eta+",\"channels\":["+rows+"],\"channel_sum_inlet_m3_s\":"+sumIn+",\"channel_sum_outlet_m3_s\":"+sumOut+"}");
  ModelUtil.remove("Audit");
 }
}"""


def audit_saved_flow(model_path, interface_area_mm2, domain_map, output, *,
                     preferences_directory=None, timeout_s=600, processes=2):
    """Audit the stored solution without changing its MPH; write separate audit files.

    ``compile.log``, ``build.log`` and ``comsol.log`` are written to ``output``
    together with ``audit.json``.
    """
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    exe = _comsol.detect_comsol()  # module attribute so tests can stub it
    if exe is None:
        raise RuntimeError("COMSOL installation not found")
    text = JAVA.replace("@@MODEL@@", json.dumps(Path(model_path).resolve().as_posix()))
    text = text.replace("@@AREA@@", repr(float(interface_area_mm2)))
    from .comsol_dual import spanning_fluid_domains

    # Only channels that connect Z- and Z+ carry flow; closed pores and dead
    # ends are excluded from the physics by export_dual_java as well.
    spanning, _ = spanning_fluid_domains(domain_map)
    if not spanning:
        raise ValueError("No fluid domain connects Z- and Z+; nothing to audit")
    text = text.replace("@@IDS@@", ",".join(map(str, spanning)))
    java = output / "TPMSAudit.java"
    java.write_text(text, encoding="ascii")
    local_preferences = None
    launch = {}
    if preferences_directory is not None:
        local_preferences, staged = stage_preferences(preferences_directory, output / "preferences")
        launch["preferences_source"] = staged["source"]
        launch["preferences_input_sha256"] = staged["sha256"]
    with runtime_directory(output) as runtime:
        result, info = run_comsol_class(
            output, java, marker="TPMS_AUDIT", build_timeout_s=timeout_s,
            messages=("Compiling the COMSOL audit...", "Auditing the saved COMSOL solution..."),
            preferences_directory=local_preferences, runtime=runtime, processes=processes,
        )
    result.update(launch, comsol_bin=info["comsol_bin"], comsol_version=info["comsol_version"])
    result["interface_area_mesh_mm2"] = interface_area_mm2
    result["source_mph_sha256"] = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    mapping = {d["id"]: d for d in domain_map}
    for channel in result["channels"]:
        channel["tetrahedra"] = mapping[channel["domain_id"]]["tetrahedra"]
        channel["volume_mm3"] = mapping[channel["domain_id"]]["volume_mm3"]
    mesh_path = Path(model_path).parent / "mesh.npz"
    if mesh_path.exists():
        import numpy as np
        from .verification import audit_openings

        with np.load(mesh_path) as a:
            mesh = dict(
                points=a["points_mm"],
                tetra=a["tetrahedra"],
                phase_ids=a["phase_ids"],
                domains=a["domain_ids"],
            )
        openings = {d["domain_id"]: d for d in audit_openings(mesh, mesh["points"].max(axis=0))}
        for channel in result["channels"]:
            for name in ("inlet", "outlet"):
                area = openings[channel["domain_id"]][name + "_area_mesh_mm2"]
                error = abs(channel[name + "_area_mm2"] - area) / area
                if error > 1e-8:
                    raise RuntimeError("Imported channel opening area mismatch")
                channel[name + "_area_mesh_mm2"] = area
                channel[name + "_area_relative_error"] = error
    (output / "audit.json").write_text(json.dumps(result, indent=2), "utf8")
    return result
