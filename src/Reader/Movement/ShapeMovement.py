import WorkBanch.util as util

from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Pnt, gp_Ax1, gp_Dir
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.TopoDS import TopoDS_Shape

from OCC.Display.SimpleGui import init_display

import math

class ShapeMovement:
    def __init__(self, shape: TopoDS_Shape, move: tuple[float, float, float], rotate: tuple[float, float, float, float]) -> None:
        self.Trans = gp_Trsf()
        self.Shape = shape
        self.Move(move)
        self.Rotate(rotate)
        pass
    
    def Move(self, move: tuple[float, float, float]) -> None:
        # shape move
        self.Trans.SetTranslation(gp_Vec(move[0], move[1], move[2]))
        moveLocation = TopLoc_Location(self.Trans)
        
        self.Shape.Move(moveLocation)
        return
    
    def Rotate(self, rotate: tuple[float, float, float, float]) -> None:
        axis = gp_Ax1(gp_Pnt(self.Shape.Location().Transformation().TranslationPart()), gp_Dir(rotate[0], rotate[1], rotate[2]))
        angle = math.radians(rotate[3])
        
        rotation = gp_Trsf()
        rotation.SetRotation(axis, angle)
        
        
        self.Shape.Move(TopLoc_Location(rotation))
        
        return
        
        
if (__name__ == "__main__"):
    a,b,c,d = init_display()
    
    box =  BRepPrimAPI_MakeBox(gp_Pnt(0,0,0), gp_Pnt(1,1,1)).Shape()
    a.DisplayShape(box)
    sm = ShapeMovement(box, (1,1,1), (1,0,0,45))
    a.DisplayShape(box)
    a.FitAll()
    b()
    pass