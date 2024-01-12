from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox    
from OCC.Core.Bnd import Bnd_Box    
from OCC.Core.BRepBndLib import brepbndlib  
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.gp import gp_Pnt

from src.model_3d.cad_model import PartModel



brep_shape: TopoDS_Shape = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), gp_Pnt(1, 1, 1)).Shape()    

part_model = PartModel(brep_shape=brep_shape)    
