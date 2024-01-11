from model_3d.cad_model import Assembly, PartModel, AssemblyFactory 
from model_3d.cad_model import ViewDocument
import pyvista as pv
import time


assembly = AssemblyFactory.create_assembly("ButterflyValve.stp")

time_start = time.time()    
for part_model in assembly.part_model_list:
    part_model.simplify(0.85)
print(time.time() - time_start) 

view_document  = ViewDocument()
assembly.add_to_view_document(view_document)
view_document.display()