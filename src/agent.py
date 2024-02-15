import numpy as np
from typing import Tuple    
import pyvista as pv

from src.model_3d.model_util import ChamferDistance, RegionGrowing
from src.model_3d.cad_model import Assembly, PartModel, AssemblyFactory

class LMSObservation:
    def __init__(self, original_assembly: Assembly, simplified_assembly: Assembly) -> None:
        self.original_assembly: Assembly = original_assembly    
        self.simplified_assembly: Assembly = simplified_assembly
        self.chamfer_distance: ChamferDistance = ChamferDistance()
        self.chamfer_distance_assembly: ChamferDistance = ChamferDistance() 
        self.max_part_number: int = len(self.original_assembly.part_model_list)
        self.observation: np.ndarray = np.zeros(30 * 3 + 3, dtype=np.float32)
    
    
    def get_observation(self) -> np.ndarray:   
        self.observation = np.zeros(30 * 2 + 1, dtype=np.float32)
        total_chamfer_distance = self.chamfer_distance_assembly.evaluate(self.simplified_assembly, self.original_assembly)    
        
        for i in range(len(self.simplified_assembly.part_model_list)):
            simplified_part_model = self.simplified_assembly.part_model_list[i]
            original_part_model = self.original_assembly.part_model_list[i]             
            if original_part_model.vista_mesh.n_faces_strict != 0:
                self.observation[i * 2] = simplified_part_model.vista_mesh.n_faces_strict / original_part_model.vista_mesh.n_faces_strict
            if total_chamfer_distance != 0:
                self.observation[i * 2 + 1] = self.chamfer_distance.evaluate(simplified_part_model, original_part_model) / total_chamfer_distance 
    
        self.observation[-1] = self.simplified_assembly.get_face_number() / self.original_assembly.get_face_number()  

        return self.observation


class LMSAgent:
    def __init__(self, original_assembly: Assembly, simplified_assembly: Assembly) -> None:
        self.original_assembly: Assembly = original_assembly    
        self.simplified_assembly: Assembly = simplified_assembly 
        self.region_growing: RegionGrowing = RegionGrowing()   
        self.observation: LMSObservation = LMSObservation(self.original_assembly, self.simplified_assembly)  
        self.max_part_number: int = len(self.original_assembly.part_model_list)
        self.observation_step: int = 0      
        
    def action(self, decimation_index: int, decimate_ratio: float) -> None:     
        self.simplified_assembly.part_model_list[decimation_index].vista_mesh.n_faces_strict
        self.simplified_assembly.part_model_list[decimation_index].simplify(decimate_ratio) 
        return 
        
    def out_of_range(self, part_index: int):
        if part_index < 0 or part_index >= self.max_part_number:
            return True
        return False    
    
    def get_observation(self) -> np.ndarray:
        return self.observation.get_observation() 
    
    def is_breakage_part(self, part_index: int) -> bool:
        part_mesh = self.simplified_assembly.part_model_list[part_index].vista_mesh
        edges = part_mesh.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
        is_not_closed = edges.n_points != 0
        
        return is_not_closed    
    
    def get_reward(self, 
                    decimation_index: int = 0,
                    terminated: bool = False) -> float:  
        
        reward: float = 0.0 
        
        if self.is_breakage_part(decimation_index):
            reward -= 0.1
        else:
            reward += 0.1

        if terminated:
            for i in range(len(self.simplified_assembly.part_model_list)):  
                part_model = self.simplified_assembly.part_model_list[i]
                original_part_model = self.original_assembly.part_model_list[i] 
                chamfer_distance: float = ChamferDistance().evaluate(part_model, original_part_model)
                
                if chamfer_distance < 10:   
                    chamfer_distance = 10.0
                
                reward += (1 / chamfer_distance)
            
            for part_model in self.simplified_assembly.part_model_list:
                if part_model.is_open() or part_model.vista_mesh.n_faces_strict == 0:
                    reward -= 1.0

        return reward
    

class LMSCAgent:
    def __init__(self, original_assembly: Assembly, simplified_assembly: Assembly) -> None:
        self.original_assembly: Assembly = original_assembly    
        self.simplified_assembly: Assembly = simplified_assembly  
        self.observation_step: int = 0      
        
        
    def is_breakage_part(self, part_index: int) -> bool:        
        part_mesh = self.simplified_assembly.part_model_list[part_index].vista_mesh
        if part_mesh is None or part_mesh.n_faces_strict == 0:
            return True

        edges = part_mesh.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
        is_not_closed = edges.n_points != 0
        
        return is_not_closed    

    def action(self, 
                decimation_index: float, 
                decimate_ratio: float, 
                clustering_index: float, 
                growing_ratio: float) -> None:
        reward: float = 0.0 
        decimation_index = self.quantize_action(decimation_index)
        for i in range(len(self.simplified_assembly.part_model_list)):
            if self.is_breakage_part(i):
                reward -= 0.1
        
        if self.is_breakage_part(decimation_index):
            reward -= 0.1

        self.simplified_assembly.part_model_list[decimation_index].simplify(decimate_ratio)

        clustering_index = self.quantize_action(clustering_index)   

        region_growing = RegionGrowing(growing_ratio)
        cluster_list: list[list[int]] = region_growing.cluster(self.simplified_assembly, clustering_index)    
        self.simplified_assembly = AssemblyFactory.create_merged_assembly(self.simplified_assembly, cluster_list, "Merged AirCompressor")   
        self.original_assembly = AssemblyFactory.create_merged_assembly(self.original_assembly, cluster_list, "Merged AirCompressor")       
        return reward
            
    def get_observation(self) -> np.ndarray:
        observation: LMSObservation = LMSObservation(self.original_assembly, self.simplified_assembly)  
        return observation.get_observation() 
    
    def get_reward(self, 
                    terminated: bool = False) -> float:          
        reward: float = 0.0 

        if terminated:
            for i in range(len(self.simplified_assembly.part_model_list)):  
                part_model = self.simplified_assembly.part_model_list[i]
                original_part_model = self.original_assembly.part_model_list[i] 
                chamfer_distance: float = ChamferDistance().evaluate(part_model, original_part_model)
                
                if chamfer_distance < 10:   
                    chamfer_distance = 10.0
                
                reward += (1 / chamfer_distance)
            
            for part_model in self.simplified_assembly.part_model_list:
                if part_model is None or part_model.vista_mesh.n_faces_strict == 0:
                    reward -= 1.0

        return reward
    
    def quantize_action(self, continuous_action):
        part_number = len(self.simplified_assembly.part_model_list)
        discrete_action = int(round(continuous_action * (part_number - 1)))
        discrete_action = max(0, min(discrete_action, part_number- 1))
        return discrete_action