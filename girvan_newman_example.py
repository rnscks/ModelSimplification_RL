from src.model_3d.cad_model import AssemblyFactory, Assembly
from src.model_3d.model_util import GirvanNewman

assembly = AssemblyFactory.create_assembly("GearMotorPump.stp")
cl = GirvanNewman().cluster(assembly)        

ma: Assembly = AssemblyFactory.create_merged_assembly(assembly, cl, "r")    
colors = ["red", "blue", "yellow", "purple", "green", "orange", "pink", "brown", "gray", "black"]   
import pyvista as pv    
p = pv.Plotter()    

for idx, part in enumerate(ma.part_model_list):
    p.add_mesh(part.vista_mesh, color=colors[idx % len(colors)])

p.show()   

