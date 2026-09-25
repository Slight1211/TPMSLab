import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.*;
public class ExportPaper {
 static String ROOT="C:/Users/Administrator/Documents/Codex/2026-09-24/xi/";
 static void export(String input,String name,boolean flow) throws Exception {
  Model m=ModelUtil.load("M",ROOT+input+"/model_solved.mph");
  m.result().numerical().create("paperEval","Eval");
  if(flow)m.result().numerical("paperEval").selection().named("fluid_domains");
  String[] expr=flow?new String[]{"x/1[mm]","y/1[mm]","z/1[mm]","spf.U/1[mm/s]","p/1[Pa]"}:new String[]{"x/1[mm]","y/1[mm]","z/1[mm]","solid.disp/1[um]","solid.mises/1[MPa]"};
  m.result().numerical("paperEval").set("expr",expr);
  double[][][] data=m.result().numerical("paperEval").getData();
  int[][] elems=m.result().numerical("paperEval").getElements();
  ByteArrayOutputStream bytes=new ByteArrayOutputStream(); DataOutputStream out=new DataOutputStream(bytes);
  out.writeInt(data.length);out.writeInt(data[0][0].length);out.writeInt(elems.length);out.writeInt(elems[0].length);
  for(int i=0;i<data.length;i++)for(double v:data[i][0])out.writeDouble(v);
  for(int[] a:elems)for(int v:a)out.writeInt(v);out.close(); System.out.println("PAPER_DATA_"+name+" "+java.util.Base64.getEncoder().encodeToString(bytes.toByteArray()));
  System.out.println("EXPORTED "+name+" points="+data[0][0].length+" element rows="+elems.length);
  try {
   m.result().create("paperPlot","PlotGroup3D");m.result("paperPlot").create("f",flow?"Slice":"Surface");
   m.result("paperPlot").feature("f").set("expr",flow?"spf.U":"solid.disp");
   m.result("paperPlot").feature("f").set("unit",flow?"mm/s":"um");
   m.result().export().create("paperImage","paperPlot","Image");
   m.result().export("paperImage").set("imagetype","png");m.result().export("paperImage").set("pngfilename",ROOT+"work/paper-figures/"+name+"_comsol.png");
   m.result().export("paperImage").set("width",1600);m.result().export("paperImage").set("height",1200);
   m.result().export("paperImage").run();
  }catch(Exception e){System.out.println("IMAGE_EXPORT_NOTE "+e.getMessage());}
  ModelUtil.remove("M");
 }
 public static void main(String[] args) throws Exception {
  export("outputs/solid-fluid-validation/Gyroid","flow",true);
  export("outputs/tpmslab-comsol-validation/Gyroid_quality_fan","elastic",false);
 }
}
