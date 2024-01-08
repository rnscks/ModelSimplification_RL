import numpy as np
from typing import Tuple    

from model_3d.model_util import ChamferDistanceAssembly, ChamferDistance, RegionGrowing
from model_3d.cad_model import Assembly, PartModel, AssemblyFactory

class ListMethodChamferDistanceObservation:
    def __init__(self, original_assembly: Assembly, simplified_assembly: Assembly) -> None:
        self.original_assembly: Assembly = original_assembly    
        self.simplified_assembly: Assembly = simplified_assembly
        self.chamfer_distance: ChamferDistance = ChamferDistance()
        self.chamfer_distance_assembly: ChamferDistanceAssembly = ChamferDistanceAssembly() 
        self.max_part_number: int = len(self.original_assembly.part_model_list)
        self.observation: np.ndarray = np.zeros(30 * 3 + 3, dtype=np.float32)
    
    
    def get_observation(self, current_action: tuple[int, float]) -> np.ndarray:   
        self.observation = np.zeros(30 * 3 + 3, dtype=np.float32)
        
        for i in range(len(self.simplified_assembly.part_model_list)):
            simplified_part_model = self.simplified_assembly.part_model_list[i]
            original_part_model = self.original_assembly.part_model_list[i] 
            
            self.observation[i * 3] = simplified_part_model.vista_mesh.n_faces_strict
            self.observation[i * 3 + 1] = self.chamfer_distance.evaluate(simplified_part_model, original_part_model)    

        self.observation[current_action[0] * 3 + 2] = current_action[1]
        self.observation[-1] = self.chamfer_distance_assembly.evaluate(self.simplified_assembly, self.original_assembly)    
        self.observation[-2] = self.simplified_assembly.get_face_number()   
        self.observation[-3] = self.max_part_number

        return self.observation


class ListMethodSimplificationAgent:
    def __init__(self, original_assembly: Assembly, simplified_assembly: Assembly) -> None:
        self.original_assembly: Assembly = original_assembly    
        self.simplified_assembly: Assembly = simplified_assembly 
        self.region_growing: RegionGrowing = RegionGrowing()   
        self.observation: ListMethodChamferDistanceObservation = ListMethodChamferDistanceObservation(self.original_assembly, self.simplified_assembly)  
        self.max_part_number: int = len(self.original_assembly.part_model_list)
        self.observation_step: int = 0      
        
    def action(self, action: tuple[np.int32, np.float32, np.int32, np.float32]) -> Tuple[bool, bool, int]:     
        is_out_of_range_decimation: bool = False      
        is_out_of_range_clustering: bool = False
        
        if not self.out_of_range(action[0]):
            is_out_of_range_decimation = True
            dfaces = self.simplified_assembly.part_model_list[action[0]].simplify(action[1]) 
            
        if not self.out_of_range(action[2]):
            is_out_of_range_clustering = True
            original_merged_assembly, merged_assembly = self.get_merged_assembly(action[3])
            self.original_assembly = original_merged_assembly
            self.simplified_assembly = merged_assembly  
        
        return is_out_of_range_decimation, is_out_of_range_clustering, dfaces
    
    def get_merged_assembly(self, growing_ratio: float) -> Assembly:
        cluster_list = RegionGrowing(growing_ratio=growing_ratio).cluster(self.original_assembly)
        original_merged_assembly = \
            AssemblyFactory.create_merged_assembly(assembly = self.original_assembly,   
                                                cluster_list = cluster_list, 
                                                assembly_name = f"merged {self.original_assembly.part_name}")   
            
        merged_assembly = \
            AssemblyFactory.create_merged_assembly(assembly = self.simplified_assembly, 
                                                cluster_list = cluster_list, 
                                                assembly_name = f"merged {self.original_assembly.part_name}")
        
        return original_merged_assembly, merged_assembly
    
    def out_of_range(self, part_index: int):
        if part_index < 0 or part_index >= self.max_part_number:
            return True
        return False    
    
    def get_observation(self, current_action: float) -> np.ndarray:
        return self.observation.get_observation(current_action) 
    
    def get_reward(self, 
                    dfaces: int = 0,
                    is_out_of_range_decimation: bool = False, 
                    is_out_of_range_clustering: bool = False,
                    terminated: bool = False) -> float:  
        reward: float = 0.0 
        if is_out_of_range_decimation:
            reward -= 0.1
        if is_out_of_range_clustering:
            reward -= 0.1
            
        for part_model in self.simplified_assembly.part_model_list:
            if part_model.vista_mesh.n_faces_strict == 0:
                reward -= 1.0
        dfaces_ratio: float = dfaces / self.original_assembly.get_face_number()
        reward += dfaces_ratio
        if terminated:
            chamfer_distance: float = ChamferDistanceAssembly().evaluate(self.original_assembly, self.simplified_assembly)  
            if chamfer_distance == 0:   
                chamfer_distance = 0.00001
            reward += (1 / chamfer_distance)
        
        return reward
    
