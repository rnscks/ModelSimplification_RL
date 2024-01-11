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
    
    
    def get_observation(self) -> np.ndarray:   
        self.observation = np.zeros(30 * 2 + 1, dtype=np.float32)
        total_chamfer_distance = self.chamfer_distance_assembly.evaluate(self.simplified_assembly, self.original_assembly)    
        
        for i in range(len(self.simplified_assembly.part_model_list)):
            simplified_part_model = self.simplified_assembly.part_model_list[i]
            original_part_model = self.original_assembly.part_model_list[i] 
            self.observation[i * 3] = simplified_part_model.vista_mesh.n_faces_strict / original_part_model.vista_mesh.n_faces_strict   
            self.observation[i * 3 + 1] = self.chamfer_distance.evaluate(simplified_part_model, original_part_model) / total_chamfer_distance 
        
        self.observation[-1] = self.simplified_assembly.get_face_number() / self.original_assembly.get_face_number()  

        return self.observation


class ListMethodSimplificationAgent:
    def __init__(self, original_assembly: Assembly, simplified_assembly: Assembly) -> None:
        self.original_assembly: Assembly = original_assembly    
        self.simplified_assembly: Assembly = simplified_assembly 
        self.region_growing: RegionGrowing = RegionGrowing()   
        self.observation: ListMethodChamferDistanceObservation = ListMethodChamferDistanceObservation(self.original_assembly, self.simplified_assembly)  
        self.max_part_number: int = len(self.original_assembly.part_model_list)
        self.observation_step: int = 0      
        
    def action(self, decimation_index: int, decimate_ratio: float) -> Tuple[bool, int]:     
        dfaces: int = 0
        faces = self.simplified_assembly.part_model_list[decimation_index].vista_mesh.n_faces_strict
        dfaces = self.simplified_assembly.part_model_list[decimation_index].simplify(decimate_ratio) 
        if faces:
            dfaces /= faces
        else:
            dfaces = 0.0   

        return dfaces 
        
    def out_of_range(self, part_index: int):
        if part_index < 0 or part_index >= self.max_part_number:
            return True
        return False    
    
    def get_observation(self) -> np.ndarray:
            
        return self.observation.get_observation() 
    
    def get_reward(self, 
                    decimation_index: int = 0,
                    dfaces_ratio: int = 0,
                    terminated: bool = False) -> float:  
        reward: float = 0.0 

        if self.simplified_assembly.part_model_list[decimation_index].vista_mesh.n_faces_strict == 0:
            reward -= 0.1
        # reward += dfaces_ratio

        if terminated:
            for i in range(len(self.simplified_assembly.part_model_list)):  
                part_model = self.simplified_assembly.part_model_list[i]
                original_part_model = self.original_assembly.part_model_list[i] 
                chamfer_distance: float = ChamferDistance().evaluate(part_model, original_part_model)
                
                if chamfer_distance < 10:   
                    chamfer_distance = 10.0
                
                reward += (1 / chamfer_distance)
            
            for part_model in self.simplified_assembly.part_model_list:
                if part_model.vista_mesh.n_faces_strict == 0:
                    reward -= 1.0

        return reward
    
