from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox    
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.gp import gp_Pnt  

from src.model_3d.cad_model import PartModel, ViewDocument

brep_shape: TopoDS_Shape = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), gp_Pnt(1, 1, 1)).Shape()
part_model = PartModel(
    part_name="part model for unit test",
    brep_shape=brep_shape,
    part_index=0
)   

view_document = ViewDocument()  
part_model.add_to_view_document(view_document)
view_document.display()
