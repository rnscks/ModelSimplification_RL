from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import TopoDS_Compound
# for example code
from OCC.Display.SimpleGui import init_display

class FileReader:
    @staticmethod  
    def read_stp_file(file_name:str) -> TopoDS_Compound:
        stpReader = STEPControl_Reader()
        stpReader.ReadFile(file_name)
        stpReader.TransferRoots()
        
        return stpReader.Shape()
    
if __name__ == "__main__":
    a,b,c,d = init_display()
    brep_shape: TopoDS_Compound = FileReader.read_stp_file("AirCompressor.stp")
    a.DisplayShape(brep_shape, update=True)
    a.FitAll()
    b()