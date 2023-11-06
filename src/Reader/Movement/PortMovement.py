import WorkBanch.util as util

from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Pnt, gp_Ax1, gp_Dir
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Display.SimpleGui import init_display
from ShapeMovement import ShapeMovement
import math

class RoutingShapeMovement(ShapeMovement):
    def __init__(self, shape: TopoDS_Shape, ports: list[gp_Pnt] ,move: tuple[float, float, float], rotate: tuple[float, float, float, float]) -> None:
        self.Shape = shape
        self.Ports: list[gp_Pnt] = ports
        self.PortsMove(move)
        self.PortsRotate(rotate)

    def PortsMove(self, move: tuple[float, float, float]) -> None:
        for port in self.Ports:
            port.Translate(gp_Vec(move[0], move[1], move[2]))
        return
    def PortsRotate(self, rotate: tuple[float, float, float, float]) -> None:
       axis = gp_Ax1(gp_Pnt(self.Shape.Location().Transformation().TranslationPart()), gp_Dir(rotate[0], rotate[1], rotate[2]))
       angle = math.radians(rotate[3])
       
       rotation = gp_Trsf()
       rotation.SetRotation(axis, angle)
 
       for port in self.Ports:
           port.Transform(rotation)
           
if (__name__ == "__main__"):
    a,b,c,d = init_display()
    
    box =  BRepPrimAPI_MakeBox(gp_Pnt(0,0,0), gp_Pnt(1,1,1)).Shape()
    a.DisplayShape(box)
    sm = RoutingShapeMovement(box,[gp_Pnt(0,0,0)], (1,1,1), (1,0,0,45))
    s = BRepPrimAPI_MakeSphere(sm.Ports[0], 0.1).Shape()
    a.DisplayShape(box)
    a.DisplayShape(s)
    a.FitAll()
    b()
    pass