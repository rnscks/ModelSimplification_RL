from src.model_3d.cad_model import AssemblyFactory, ViewDocument    
from src.model_3d.model_util import RegionGrowing, ChamferDistance, PointToMeshDistance


def region_growing_example() -> None:
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
    air_compressor = AssemblyFactory.create_assembly("AirCompressor.stp")
    model1 = air_compressor.part_model_list[0]
    model2 = air_compressor.part_model_list[1]
    print(ChamferDistance().evaluate(model1, model2))   
    return
    
def point_to_mesh_distance_example() -> None:   
    air_compressor = AssemblyFactory.create_assembly("AirCompressor.stp")
    model1 = air_compressor.part_model_list[0]
    model2 = air_compressor.part_model_list[1]
    print(PointToMeshDistance().evaluate(model1, model2))
    return

