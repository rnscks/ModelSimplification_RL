import numpy as np

from model_3d.model_util import ChamferDistanceAssembly, ChamferDistance 
from model_3d.cad_model import Assembly, PartModel

class ChamferDistanceObservation:
    def __init__(self) -> None:
        self.chamfer_distance: ChamferDistance = ChamferDistance()  
    
    
    def get_observation(self, original_part: PartModel, simplified_part: PartModel) -> np.ndarray:   
        observation = np.zeros(2)
        observation[0] = self.chamfer_distance.evaluate(original_part, simplified_part)
        observation[1] = simplified_part.vista_mesh.n_faces_strict
        
        return observation


class SimplificationAgent:
    def __init__(self, original_assembly: Assembly, simplified_assembly: Assembly) -> None:
        self.original_assembly: Assembly = original_assembly    
        self.simplified_assembly: Assembly = simplified_assembly 
        self.observation: ChamferDistanceObservation = ChamferDistanceObservation()  
        self.max_part_number: int = len(self.original_assembly.part_model_list)
        self.observation_step: int = 0      
        self.action_step: int = 0   
        
    def action(self, action: np.float32) -> None:        
        self.simplified_assembly.part_model_list[self.action_step].simplify(action)
        self.action_step = (self.action_step + 1) % self.max_part_number    
        return None
    
    def get_observation(self) -> np.ndarray:
        original_part: PartModel = self.original_assembly.part_model_list[self.observation_step]
        simplified_part: PartModel = self.simplified_assembly.part_model_list[self.observation_step]
        self.observation_step = (self.observation_step + 1) % self.max_part_number 

        return self.observation.get_observation(original_part, simplified_part) 
    
    def get_reward(self) -> float:  
        reward: float = 0.0 
        for part_model in self.simplified_assembly.part_model_list:
            if part_model.vista_mesh.n_faces_strict == 0:
                reward -= 1.0
            
        return reward
    
    def get_last_reward(self) -> float:
        chamfer_distance: float = ChamferDistanceAssembly().evaluate(self.original_assembly, self.simplified_assembly)  
        if chamfer_distance == 0:   
            chamfer_distance = 0.00001
        original_face_number: int = self.original_assembly.get_face_number()
        simplified_face_number: int = self.simplified_assembly.get_face_number()
        
        return (1 / chamfer_distance) 