from OCC.Core.gp import gp_Pnt
from OCC.Display.OCCViewer import Viewer3d

from STLReader import STPReader
from Movement.PortMovement import RoutingShapeMovement

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere

from OCC.Display.SimpleGui import init_display
from OCC.Core.Quantity import (
    Quantity_Color,
    Quantity_TOC_RGB,
    Quantity_NOC_WHITE
)

class RoutingSTPReader(STPReader):
    def __init__(self, fileName: str, ports: list[gp_Pnt] ,move: tuple[float, float, float] = (0,0,0), rotate: tuple[float, float, float, float] = (1,0,0,0)) -> None:
        super().__init__(fileName, move, rotate)
        self.Ports = ports
        rm = RoutingShapeMovement(self.STPShape, self.Ports, move, rotate)
        pass
    
    def DisplayPortsShpae(self, display: Viewer3d, _transparency: float = 0.5, _color: str = "red") -> None:
        for port in self.Ports:
            s = BRepPrimAPI_MakeSphere(port, 1).Shape()
            display.DisplayShape(s, transparency= _transparency, color= _color)
        
    
if (__name__ == "__main__"):
    a,b,c,d = init_display()
    
    a.View.SetBackgroundColor(Quantity_TOC_RGB, 0, 0, 0)
    a.hide_triedron()

    a.View.SetBgGradientColors(
        Quantity_Color(Quantity_NOC_WHITE),
        Quantity_Color(Quantity_NOC_WHITE),
        2,
        True,
    )
    
    stp = RoutingSTPReader("AB6M-M1P-G.stp", [gp_Pnt(0,0,0)], (20, 0, 0), (1, 0, 0, 80))
    stp.DisplaySTPShape(a, _color = "black")
    stp.DisplayPortsShpae(a, _color = "black")
    a.FitAll()
    b()
    