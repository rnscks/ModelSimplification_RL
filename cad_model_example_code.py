from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox    
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.gp import gp_Pnt  

from src.model_3d.cad_model import PartModel, ViewDocument, Assembly, AssemblyFactory
from src.model_3d.model_util import RegionGrowing   


def visual_part_model_example() -> None:
    """
    부품 모델 예제를 시각화하는 함수입니다.

    Return:
        None
    """
    brep_shape: TopoDS_Shape = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), gp_Pnt(1, 1, 1)).Shape()
    part_model = PartModel(
        part_name="part model for example code",
        brep_shape=brep_shape,
        part_index=0
    )   

    view_document = ViewDocument()  
    part_model.add_to_view_document(view_document)
    view_document.display()
    return

def visual_assembly_example() -> None:
    """
    어셈블리 시각화하는 예제 함수입니다.

    Return:
        None
    """
    assembly: Assembly = AssemblyFactory.create_assembly("AirCompressor.stp")
    view_document = ViewDocument()  
    assembly.add_to_view_document(view_document)
    view_document.display()
    return

def visual_merged_assembly_example() -> None:
    """
    병합된 어셈블리를 시각화하는 예제입니다.

    Return:
        None

    Parameters:
        None
    """
    assembly: Assembly = AssemblyFactory.create_assembly("AirCompressor.stp")
    cluster_list = RegionGrowing().cluster(assembly, 0)    
    merged_assembly = AssemblyFactory.create_merged_assembly(assembly, cluster_list, "Merged AirCompressor")    
    view_document = ViewDocument()
    merged_assembly.add_to_view_document(view_document) 
    view_document.display()
    
visual_merged_assembly_example()