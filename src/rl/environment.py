import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, List

from src.mesh.metrics import ChamferDistance, METRIC
from src.mesh.model import Assembly
from src.rl.agent import SimplificationAgent
from src.rl.task_buffer import TaskBuffer, Task
from src.rl.agent import GRAPH
from src.rl.util import profile_time


class TrainEnv(gym.Env):
    metadata = {"render_modes": [None]}
    def __init__(self, 
                task_dirs: List[str],   
                max_time_step: int = 50,
                target_sim_boundary: Tuple[float, float] = (0.5, 0.9)) -> None:
        super(TrainEnv, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'pointcloud': spaces.Box(low=-1, high=1, shape=(GRAPH.MAX_NODES.value, 1024, 3), dtype=np.float32),
            'node': spaces.Box(low=-1, high=1, shape=(GRAPH.MAX_NODES.value, GRAPH.NODE_DIM.value), dtype=np.float32),
            'edge_index': spaces.Box(low=0, high=GRAPH.MAX_NODES.value-1, shape=(GRAPH.MAX_EDGES.value, 2), dtype=np.int64),
            'edge_attr': spaces.Box(low=-1, high=1, shape=(GRAPH.MAX_EDGES.value, 1), dtype=np.float32)})
        self.task_buffer: TaskBuffer = TaskBuffer(task_dirs)
        self.max_time_step: int = max_time_step
        self.simplyfied_assembly: Assembly = Assembly()
        self.original_assembly: Assembly = Assembly()
        self.target_sim_boundary: Tuple[float, float] = target_sim_boundary 
        self.n_episode: int = 0
        return


    def step(self, action: float) -> Tuple[object, float, bool, dict]:
        terminated: bool = False
        # 기본 음의 보상
        reward: float = -0.1
        self.time_step += 1
        decimation_index, decimation_ratio, cluster_ratio = action
        self.simplyfied_assembly, self.original_assembly = self.agent.action(
            simplified_assembly=self.simplyfied_assembly,
            original_assembly=self.original_assembly,
            decimation_index=decimation_index, 
            decimate_ratio=decimation_ratio,
            cluster_ratio=cluster_ratio)
        
        obs = self.agent.get_observation(self.simplyfied_assembly, self.original_assembly)
        total_n_face = self.simplyfied_assembly.n_faces()
        original_n_face = self.original_assembly.n_faces()
        
        for part in self.simplyfied_assembly:
            if isinstance(part.mesh, type(None)) or part.n_faces() == 0:
                print("Fail!!(Episode): Part is None")
                terminated = True
                break
        
        if (total_n_face/original_n_face) <= (1 - self.target_sim_ratio):
            terminated = True
            print("Success!!(Episode): Simplified")
        
        if self.time_step >= self.max_time_step:
            terminated = True   
            print("Failed!!(Episode): Time Over")
        
        reward += self._get_reward(decimation_index, terminated)
        return obs, reward, terminated, False, {}
    
    def reset(self, 
            seed=None, 
            options=None):
        
        self.n_episode += 1
        assembly = self.task_buffer.cur_assembly()  
        
        self.original_assembly = Assembly()
        self.simplyfied_assembly = Assembly()
        self.original_assembly.copy_from(assembly) 
        self.simplyfied_assembly.copy_from(assembly)
        
        self.target_sim_ratio = np.random.uniform(self.target_sim_boundary[0], self.target_sim_boundary[1])  
        
        self.agent = SimplificationAgent(self.action_space)  
        self.time_step: int = 0
        
        obs = self.agent.get_observation(self.simplyfied_assembly, self.original_assembly)
        return obs, {}
    
    def _get_reward(self, decimation_index: float, terminated: bool) -> float:
        reward: float = 0.0
        decimation_index = self._quantize_action(decimation_index, len(self.simplyfied_assembly))
        
        # 단순화 유도 보상
        remain_face_ratio: float = (self.max_time_step - self.time_step)/self.max_time_step
        reward -= remain_face_ratio

        target_sim_ratio = (self.target_sim_ratio*self.original_assembly.n_faces()) - (self.original_assembly.n_faces() - self.simplyfied_assembly.n_faces())
        target_sim_ratio /= self.original_assembly.n_faces()
        reward -= target_sim_ratio
        
        # 형상 유사도 보상
        max_cd = 0.6
        min_cd = 0  
        cd: float = ChamferDistance().evaluate(self.simplyfied_assembly, self.original_assembly)
        if cd == METRIC.BROKEN:
            return -10.0
        cd = (cd - min_cd) / (max_cd - min_cd)  
        reward -= cd
        
        total_n_face = self.simplyfied_assembly.n_faces()
        original_n_face = self.original_assembly.n_faces()
        
        if total_n_face/original_n_face < (1 - self.target_sim_ratio):
            reward += 10.0
        if self.time_step >= self.max_time_step:
            reward -= 10.0
            
        if terminated:
            cd_norm = cd
            epsilon = 1e-6
            reward += np.log(1 + 1/(cd_norm + epsilon))    
            
            n_face = self.simplyfied_assembly.n_faces() 
            target_n_face = self.original_assembly.n_faces() - (self.target_sim_ratio * self.original_assembly.n_faces())   
            reward += (n_face - target_n_face) / target_n_face  
            
        return reward
    
    def _quantize_action(self, continuous_action, max_range: int = 10):
        discrete_action = int(round(continuous_action * (max_range - 1)))
        discrete_action = max(0, min(discrete_action, max_range- 1))
        return discrete_action
    
    def render(self, mode: str = 'human') -> None:
        raise NotImplementedError()
    
    def close(self) -> None:
        raise NotImplementedError()
    
    
class TestEnv(gym.Env):
    metadata = {"render_modes": [None]}
    def __init__(self, 
                test_model: Assembly,   
                max_time_step: int = 50,
                target_sim_ratio: float = 0.98) -> None:
        super(TestEnv, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'node': spaces.Box(low=-1, high=1, shape=(GRAPH.MAX_NODES.value, GRAPH.NODE_DIM.value), dtype=np.float32),
            'edge_index': spaces.Box(low=0, high=GRAPH.MAX_NODES.value-1, shape=(GRAPH.MAX_EDGES.value, 2), dtype=np.int64),
            'edge_attr': spaces.Box(low=-1, high=1, shape=(GRAPH.MAX_EDGES.value, 1), dtype=np.float32)})
        self.test_model: Assembly = test_model  
        self.max_time_step: int = max_time_step
        self.simplyfied_assembly: Assembly = Assembly()
        self.original_assembly: Assembly = Assembly()
        self.target_sim_ratio: float = target_sim_ratio
        self.ret_assembly: Assembly = Assembly()    
        self.n_episode: int = 0
        return


    def step(self, action: float) -> Tuple[object, float, bool, dict]:
        terminated: bool = False
        # 기본 음의 보상
        reward: float = -0.1
        self.time_step += 1
        decimation_index, decimation_ratio, cluster_ratio = action
        self.simplyfied_assembly, self.original_assembly = self.agent.action(
            simplified_assembly=self.simplyfied_assembly,
            original_assembly=self.original_assembly,
            decimation_index=decimation_index, 
            decimate_ratio=decimation_ratio,
            cluster_ratio=cluster_ratio)
        
        obs = self.agent.get_observation(self.simplyfied_assembly, self.original_assembly)
        total_n_face = self.simplyfied_assembly.n_faces()
        original_n_face = self.original_assembly.n_faces()
        
        for part in self.simplyfied_assembly:
            if isinstance(part.mesh, type(None)) or part.n_faces() == 0:
                print("Fail!!(Episode): Part is None")
                terminated = True
                break
        
        if (total_n_face/original_n_face) <= (1 - self.target_sim_ratio):
            terminated = True
            print("Success!!(Episode): Simplified")
        
        if self.time_step >= self.max_time_step:
            terminated = True   
            print("Failed!!(Episode): Time Over")
        
        reward += self._get_reward(decimation_index, terminated)
        
        if terminated:  
            self.ret_assembly.copy_from(self.simplyfied_assembly)
        
        return obs, reward, terminated, False, {}
    
    def reset(self, 
            seed=None, 
            options=None):
        self.n_episode += 1
        assembly = self.test_model
        
        self.original_assembly = Assembly()
        self.simplyfied_assembly = Assembly()
        self.original_assembly.copy_from(assembly) 
        self.simplyfied_assembly.copy_from(assembly)
        
        self.agent = SimplificationAgent(self.action_space)  
        self.time_step: int = 0
        
        obs = self.agent.get_observation(self.simplyfied_assembly, self.original_assembly)
        return obs, {}
    
    def _get_reward(self, decimation_index: float, terminated: bool) -> float:
        reward: float = 0.0
        decimation_index = self._quantize_action(decimation_index, len(self.simplyfied_assembly))
        
        # 단순화 유도 보상
        remain_face_ratio: float = (self.max_time_step - self.time_step)/self.max_time_step
        reward -= remain_face_ratio

        target_sim_ratio = (self.target_sim_ratio*self.original_assembly.n_faces()) - (self.original_assembly.n_faces() - self.simplyfied_assembly.n_faces())
        target_sim_ratio /= self.original_assembly.n_faces()
        reward -= target_sim_ratio
        
        # 형상 유사도 보상
        max_cd = 0.6
        min_cd = 0  
        cd: float = ChamferDistance().evaluate(self.simplyfied_assembly, self.original_assembly)
        if cd == METRIC.BROKEN:
            return -10.0
        cd = (cd - min_cd) / (max_cd - min_cd)  
        reward -= cd
        
        total_n_face = self.simplyfied_assembly.n_faces()
        original_n_face = self.original_assembly.n_faces()
        
        if total_n_face/original_n_face < (1 - self.target_sim_ratio):
            reward += 10.0
        if self.time_step >= self.max_time_step:
            reward -= 10.0
            
        if terminated:
            cd_norm = cd
            epsilon = 1e-6
            reward += np.log(1 + 1/(cd_norm + epsilon))    
        return reward
    
    def _quantize_action(self, continuous_action, max_range: int = 10):
        discrete_action = int(round(continuous_action * (max_range - 1)))
        discrete_action = max(0, min(discrete_action, max_range- 1))
        return discrete_action
    
    def render(self, mode: str = 'human') -> None:
        raise NotImplementedError()
    
    def close(self) -> None:
        raise NotImplementedError()