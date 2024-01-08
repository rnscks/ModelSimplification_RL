import numpy as np

from model_3d.model_util import ChamferDistanceAssembly, ChamferDistance 
from model_3d.cad_model import Assembly, PartModel

class CircularQueueChamferDistanceObservation:
    def __init__(self, original_assembly: Assembly, simplified_assembly: Assembly) -> None:
        self.original_assembly: Assembly = original_assembly    
        self.simplified_assembly: Assembly = simplified_assembly
        self.chamfer_distance: ChamferDistance = ChamferDistance()
        self.chamfer_distance_assembly: ChamferDistanceAssembly = ChamferDistanceAssembly()
        self.max_part_number: int = len(self.original_assembly.part_model_list)
        self.observation: np.ndarray = np.zeros(10)
    
    
    def get_observation(self, time_step: int, current_action: float) -> np.ndarray:   
        back_original_part = self.original_assembly.part_model_list[time_step - 1]
        original_part = self.original_assembly.part_model_list[time_step]
        front_original_part = self.original_assembly.part_model_list[(time_step + 1) % self.max_part_number]   
        
        back_simplified_part = self.simplified_assembly.part_model_list[time_step - 1]  
        simplified_part = self.simplified_assembly.part_model_list[time_step]
        front_simplified_part = self.simplified_assembly.part_model_list[(time_step + 1) % self.max_part_number]
        
        self.observation[0] = self.chamfer_distance.evaluate(back_original_part, back_simplified_part)  
        self.observation[1] = back_simplified_part.vista_mesh.n_faces_strict
        self.observation[2] = self.chamfer_distance.evaluate(original_part, simplified_part)
        self.observation[3] = simplified_part.vista_mesh.n_faces_strict 
        self.observation[4] = self.chamfer_distance.evaluate(front_original_part, front_simplified_part)
        self.observation[5] = front_simplified_part.vista_mesh.n_faces_strict   
        self.observation[6] = self.simplified_assembly.get_face_number() 
        self.observation[7] = self.chamfer_distance_assembly.evaluate(self.original_assembly, self.simplified_assembly)       
        self.observation[8] = current_action
        self.observation[9] = time_step
        
        return self.observation


class CircularQueueSimplificationAgent:
    def __init__(self, original_assembly: Assembly, simplified_assembly: Assembly) -> None:
        self.original_assembly: Assembly = original_assembly    
        self.simplified_assembly: Assembly = simplified_assembly 
        self.observation: CircularQueueChamferDistanceObservation = CircularQueueChamferDistanceObservation(self.original_assembly, self.simplified_assembly)  
        self.max_part_number: int = len(self.original_assembly.part_model_list)
        self.observation_step: int = 0      
        self.action_step: int = 0   
        
    def action(self, action: np.float32) -> None:        
        self.simplified_assembly.part_model_list[self.action_step].simplify(action)
        self.action_step = (self.action_step + 1) % self.max_part_number    
        return None
    
    def get_observation(self, current_action: float) -> np.ndarray:
        if self.action_step == 0:   
            self.observation_step = self.max_part_number - 1
        else:
            self.observation_step = self.action_step - 1    

        return self.observation.get_observation(self.observation_step, current_action) 
    
    def get_reward(self, terminated: bool) -> float:  
        reward: float = 0.0 
        for part_model in self.simplified_assembly.part_model_list:
            if part_model.vista_mesh.n_faces_strict == 0:
                reward -= 1.0
                
        if terminated:
            chamfer_distance: float = ChamferDistanceAssembly().evaluate(self.original_assembly, self.simplified_assembly)  
            if chamfer_distance == 0:   
                chamfer_distance = 0.00001
            reward += (1 / chamfer_distance)
            
        return reward
    
