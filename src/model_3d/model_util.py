from abc import ABC, abstractmethod

from pytorch3d import loss
from pytorch3d.structures import Pointclouds, Meshes

from model_3d.cad_model import MetaModel, Assembly, PartModel
# for example code
from model_3d.cad_model import AssemblyFactory, ViewDocument
from model_3d.cad_model import MetaModel


class Evaluator(ABC):
    def __init__(self) -> None:
        super().__init__()  
    
    
    @abstractmethod
    def evaluate(self, model: MetaModel) -> float:
        pass    

class ChamferDistance(Evaluator):
    def __init__(self) -> None:
        super().__init__()  

  
    def evaluate(self, model: PartModel, other_model: PartModel) -> float:
        p1: Pointclouds = model.torch_point_cloud
        p2: Pointclouds = other_model.torch_point_cloud
        
        chamfer_distance_loss, loss_normal = loss.chamfer_distance(p1, p2, 
                                                        point_reduction="mean", 
                                                        single_directional=False)
        
        return chamfer_distance_loss.item()

class PointToMeshDistance(Evaluator):   
    def __init__(self) -> None:
        super().__init__()
        
    def evaluate(self, model: PartModel, other_model: PartModel) -> float:
        pmd1 = loss.point_mesh_distance.point_mesh_face_distance(model.torch_mesh, 
                                                                 other_model.torch_point_cloud).item()
        
        pmd2 = loss.point_mesh_distance.point_mesh_face_distance(other_model.torch_mesh, 
                                                                 model.torch_point_cloud).item()   
        return (pmd1 + pmd2) * 0.5

class ChamferDistanceAssembly(Evaluator):   
    def __init__(self) -> None:
        super().__init__()  
        
        
    def evaluate(self, assembly: Assembly, other_assembly: Assembly) -> float:
        if len(assembly.part_model_list) != len(other_assembly.part_model_list):    
            raise ValueError("assembly and other_assembly must have same length")   
        
        sum_of_chamfer_distance: float = 0.0    
        for part_index in range(len(assembly.part_model_list)):
            part_model = assembly.part_model_list[part_index]
            other_part_model = other_assembly.part_model_list[part_index]   
            
            chamfer_distance: float = ChamferDistance().evaluate(part_model, other_part_model)    
            sum_of_chamfer_distance += chamfer_distance

        return sum_of_chamfer_distance

class Cluster(ABC):
    def __init__(self) -> None:
        super().__init__()  
        
        
    @abstractmethod
    def cluster(self, assembly: Assembly) -> list[list[int]]:
        pass    
    
class RegionGrowing(Cluster):
    def __init__(self, growing_ratio: float = 0.5) -> None:
        super().__init__()  
        self.growing_ratio: float = growing_ratio  
        self.closed_list: list[int] = []    
        self.cluster_list: list[list[int]] = []
        
        
    def cluster(self, assembly: Assembly) -> list[list[int]]:
        cluster: list[int] = []
        
        for part_model in assembly.part_model_list:
            part_index: int = part_model.part_index    
            seed_volume: float = assembly.part_model_list[part_index].get_volume()
            self.growing(part_index, assembly, cluster, seed_volume)
            
            if len(cluster) == 0:   
                continue
            
            input_list = []
            input_list.extend(cluster)
            self.cluster_list.append(input_list)
            cluster.clear() 
        self.set_part_model_color(assembly)
        return self.cluster_list
    
    def growing(self, part_index: int, assembly: Assembly, cluster: list[int], seed_number: float) -> None:
        if part_index in self.closed_list:  
            return None 

        cluster.append(part_index)
        self.closed_list.append(part_index)

        neighbor_index_list: list[int] = assembly.conectivity_dict[part_index]

        for neighbor_index in neighbor_index_list:
            neighbor_part: PartModel = assembly.part_model_list[neighbor_index]    
            
            if self.growing_ratio * neighbor_part.get_volume() < seed_number:
                continue
            
            self.growing(neighbor_index, assembly, cluster, seed_number)
        
        return None
    
    def set_part_model_color(self, assembly: Assembly) -> None:
        colors = ["red", "blue", "yellow", "purple", "green", "orange", "pink", "brown", "gray", "black"]
        for cluster_index, cluster in enumerate(self.cluster_list):
            for part_index in cluster:
                assembly.part_model_list[part_index].color = colors[cluster_index]    
        return None
    
        
if __name__ == "__main__":
    def region_growing_example():
        air_compressor = AssemblyFactory.create_assembly("AirCompressor.stp") 
        cluster_list: list[int] = RegionGrowing(growing_ratio=0.5).cluster(air_compressor)
        colors = ["red", "blue", "yellow", "purple", "green", "orange", "pink", "brown", "gray", "black"]
        for cluster_index, cluster in enumerate(cluster_list):
            for part_index in cluster:
                air_compressor.part_model_list[part_index].color = colors[cluster_index]    
        view_document = ViewDocument()  
        air_compressor.add_to_view_document(view_document)
        view_document.display()

    def chamfer_distance_example():
        air_compressor = AssemblyFactory.create_assembly("AirCompressor.stp")
        model1 = air_compressor.part_model_list[0]
        model2 = air_compressor.part_model_list[1]
        print(ChamferDistance().evaluate(model1, model2))   
        
    def point_to_mesh_distance_example():   
        air_compressor = AssemblyFactory.create_assembly("AirCompressor.stp")
        model1 = air_compressor.part_model_list[0]
        model2 = air_compressor.part_model_list[1]
        print(PointToMeshDistance().evaluate(model1, model2))
