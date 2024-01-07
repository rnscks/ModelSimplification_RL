import gymnasium as gym
from typing import Optional

from model_3d.cad_model import AssemblyFactory, ViewDocument, Assembly, PartModel
from model_3d.model_util import ChamferDistance, PointToMeshDistance, RegionGrowing, ChamferDistanceAssembly

class MeshSimplificationEnv(gym.Env):
    def __init__(self, 
                 stp_file_path: str = "AirCompressor.stp", 
                 growing_ratio: float = 0.5) -> None:
        super().__init__()
        self.stp_file_path: str = stp_file_path  
        self.growing_ratio: float = growing_ratio   
        self.original_assembly: Assembly = AssemblyFactory.create_assembly(self.stp_file_path)  
        self.merged_assembly: Optional[Assembly] = self.get_merged_assembly(self.original_assembly)


    def step(self, action: int) -> tuple[object, float, bool, dict]:
        pass
    
    def reset(self) -> object:
        self.merged_assembly = self.get_merged_assembly(self.original_assembly) 
        return
    
    def render(self, mode: str = 'human') -> None:
        raise NotImplementedError()
    
    def close(self) -> None:
        raise NotImplementedError()
    
    def get_merged_assembly(self, original_assembly: Assembly) -> Assembly:
        cluster_list = RegionGrowing(growing_ratio=self.growing_ratio).cluster(original_assembly)
        merged_assembly = \
            AssemblyFactory.create_merged_assembly(original_assembly, 
                                                   cluster_list, 
                                                   f"{original_assembly.part_name}_merged")
        return merged_assembly
    
if __name__ == "__main__":
    air_compressor: Assembly = AssemblyFactory.create_assembly("AirCompressor.stp")
    cluster_list = RegionGrowing(growing_ratio=0.5).cluster(air_compressor)
    merged_air_compressor = AssemblyFactory.create_merged_assembly(air_compressor, cluster_list, "AirCompressor")
    view_document = ViewDocument()
    merged_air_compressor.add_to_view_document(view_document)
    view_document.display() 