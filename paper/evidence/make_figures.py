from pathlib import Path
import numpy as np, vtk, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from vtk.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray,vtk_to_numpy
ROOT=Path.cwd(); OUT=ROOT/'outputs/SoftwareX_LaTeX'; TMP=ROOT/'work/paper-figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
colors=['#cf793d','#398fc0','#64b9a0']
def grid(p,t):
 v=vtk.vtkPoints();v.SetData(numpy_to_vtk(p,deep=True));g=vtk.vtkUnstructuredGrid();g.SetPoints(v)
 c=vtk.vtkCellArray();c.SetCells(len(t),numpy_to_vtkIdTypeArray(np.c_[np.full(len(t),4),t].astype(np.int64).ravel(),deep=True));g.SetCells(vtk.VTK_TETRA,c);return g

def render(items,path):
 r=vtk.vtkRenderer();r.SetBackground(1,1,1)
 for g,color in items:
  m=vtk.vtkDataSetMapper();m.SetInputData(g);m.ScalarVisibilityOff();a=vtk.vtkActor();a.SetMapper(m);a.GetProperty().SetColor(*matplotlib.colors.to_rgb(color));a.GetProperty().SetInterpolationToFlat();r.AddActor(a)
 cam=r.GetActiveCamera();cam.SetPosition(11,-15,10);cam.SetFocalPoint(2.5,2.5,2.5);cam.SetViewUp(0,0,1);cam.ParallelProjectionOn();r.ResetCamera();cam.Zoom(1.18)
 w=vtk.vtkRenderWindow();w.SetOffScreenRendering(1);w.SetSize(800,740);w.AddRenderer(r);w.Render();f=vtk.vtkWindowToImageFilter();f.SetInput(w);f.Update();wr=vtk.vtkPNGWriter();wr.SetFileName(str(path));wr.SetInputConnection(f.GetOutputPort());wr.Write();w.Finalize()

fig,axs=plt.subplots(2,3,figsize=(10,6.7))
for col,fam in enumerate(['Gyroid','Primitive_Schwartz','Diamond']):
 d=np.load(ROOT/'outputs/solid-fluid-validation'/fam/'mesh.npz');p=d['points_mm'];t=d['tetrahedra'];dom=d['domain_ids'];phase=d['phase_ids']
 for row in range(2):
  items=[]
  for i,di in enumerate(np.unique(dom[phase==(1 if row==0 else 2)])):
   items.append((grid(p,t[dom==di]),colors[0] if row==0 else colors[i+1]))
  path=TMP/f'{fam}_{row}.png';render(items,path);axs[row,col].imshow(plt.imread(path));axs[row,col].axis('off');axs[row,col].set_title(f"({chr(97+row*3+col)}) {fam.replace('_Schwartz','')} — {'solid' if row==0 else 'pore fluid'}",fontsize=11,pad=0)
fig.subplots_adjust(left=.01,right=.99,bottom=.055,top=.95,wspace=.0,hspace=.07);fig.text(.5,.015,'5 mm cube • one cell per axis • target density 0.25 → 0.50 along z • n = 12',ha='center',fontsize=10)
fig.savefig(OUT/'fig2_geometry.png',dpi=300,facecolor='white');plt.close(fig)
# Actual tetrahedron-plane cross sections
from matplotlib.collections import PolyCollection
fig,axs=plt.subplots(1,2,figsize=(9,4.2));d=np.load(ROOT/'outputs/solid-fluid-validation/Gyroid/mesh.npz');p=d['points_mm'];t=d['tetrahedra'];dom=d['domain_ids']
polys=[];cs=[]
for di in np.unique(dom):
 g=grid(p,t[dom==di]);plane=vtk.vtkPlane();plane.SetOrigin(0,2.47,0);plane.SetNormal(0,1,0);cut=vtk.vtkCutter();cut.SetCutFunction(plane);cut.SetInputData(g);cut.Update();o=cut.GetOutput();pts=vtk_to_numpy(o.GetPoints().GetData());cells=o.GetPolys();cells.InitTraversal();ids=vtk.vtkIdList()
 while cells.GetNextCell(ids):
  polys.append(pts[[ids.GetId(j) for j in range(ids.GetNumberOfIds())]][:,[0,2]]);cs.append(colors[int(di)-1])
for ax in axs:
 ax.add_collection(PolyCollection(polys,facecolors=cs,edgecolors='#333333',linewidths=.22));ax.set_aspect('equal');ax.set_xlabel('x (mm)');ax.set_ylabel('z (mm)')
axs[0].set(xlim=(0,5),ylim=(0,5),title='(a) Gyroid phase partition at y = 2.47 mm');axs[1].set(xlim=(1.7,2.7),ylim=(1.7,2.7),title='(b) Shared-interface detail');fig.tight_layout();fig.savefig(OUT/'fig3_interface.png',dpi=300);plt.close(fig)
# Vector workflow
fig,ax=plt.subplots(figsize=(9,3.7));ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off')
boxes=[(.15,2.25,'Configure implicit field\nFamily • density profile\nBox • sampling resolution'),(3.55,2.25,'Generate both phases\nClip background tetrahedra\nShare interface connectivity'),(6.95,2.25,'Verify and export\nVolumes • domain labels\nNASTRAN + NPZ + JSON'),(3.55,.15,'Python / CLI / local UI\nOne configuration\nReproducible artifacts'),(6.95,.15,'Optional COMSOL bridge\nImport → select → solve\nSave → reopen → verify')]
for x,y,s in boxes:
 ax.add_patch(FancyBboxPatch((x,y),2.85,1.45,boxstyle='round,pad=.06',facecolor='#eff5f9',edgecolor='#42667e'));ax.text(x+1.425,y+.725,s,ha='center',va='center',fontsize=10,linespacing=1.5)
for a,b in [((3.05,3),(3.45,3)),((6.45,3),(6.85,3)),((4.975,1.7),(4.975,2.1)),((8.375,2.1),(8.375,1.7))]:ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#42667e',lw=1.5))
fig.tight_layout();fig.savefig(OUT/'fig1_workflow.pdf',bbox_inches='tight');fig.savefig(OUT/'fig1_workflow.png',dpi=180,bbox_inches='tight');plt.close(fig)
print('Geometry, interface and workflow figures saved.')
