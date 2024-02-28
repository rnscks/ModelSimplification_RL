from src.model_3d.cad_model import AssemblyFactory, ViewDocument    
from src.model_3d.model_util import RegionGrowing, ChamferDistance, PointToMeshDistance


def region_growing_example() -> None:
    """
    RG 예제를 실행하는 함수입니다.
    
    Return:
        None

    Parameters:
        없음
    """
    air_compressor = AssemblyFactory.create_assembly("AirCompressor.stp") 
    cluster_list: list[int] = RegionGrowing(growing_ratio=0.5).cluster(air_compressor)
    colors = ["red", "blue", "yellow", "purple", "green", "orange", "pink", "brown", "gray", "black"]
    for cluster_index, cluster in enumerate(cluster_list):
        for part_index in cluster:
            air_compressor.part_model_list[part_index].color = colors[cluster_index]    
    view_document = ViewDocument()  
    air_compressor.add_to_view_document(view_document)
    view_document.display()
    return

def chamfer_distance_example() -> None:
    """
    Chamfer Distance를 계산하는 예제 함수입니다.

    Return:
        None

    Parameters:
        None
    """
    air_compressor = AssemblyFactory.create_assembly("AirCompressor.stp")
    model1 = air_compressor.part_model_list[0]
    model2 = air_compressor.part_model_list[1]
    print(ChamferDistance().evaluate(model1, model2))   
    return
    
def point_to_mesh_distance_example() -> None:
    """
    PMD를 계산하는 예제 함수입니다.

    Return:
        None
    """
    air_compressor = AssemblyFactory.create_assembly("AirCompressor.stp")
    model1 = air_compressor.part_model_list[0]
    model2 = air_compressor.part_model_list[1]
    print(PointToMeshDistance().evaluate(model1, model2))
    return

