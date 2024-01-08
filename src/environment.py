import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import numpy as np  
from typing import Optional, Tuple

from model_3d.cad_model import AssemblyFactory, ViewDocument, Assembly, PartModel
from model_3d.model_util import ChamferDistance, PointToMeshDistance, RegionGrowing, ChamferDistanceAssembly
from agent import ListMethodSimplificationAgent  
from torch.utils.tensorboard import SummaryWriter


class ListMethodSimplificationEnv(gym.Env):
    metadata = {"render_modes": [None]}
    def __init__(self, stp_file_path: str = "ButterflyValve.stp") -> None:
        super(ListMethodSimplificationEnv, self).__init__()
        self.action_space = spaces.Tuple(
            spaces.Discrete(30), spaces.Box(low=0.0, high=0.9, shape=(1,), dtype=float),
            spaces.Discrete(30), spaces.Box(low=0.0, high=0.9, shape=(1,), dtype=float),    
        )
        self.observation_space = spaces.Box(low=0.1, high=0.9, shape=(30 * 3 + 3,), dtype=float)
        
        self.stp_file_path: str = stp_file_path  
        
        self.original_assembly: Assembly = AssemblyFactory.create_assembly(self.stp_file_path)  
        self.simplyfied_assembly: Optional[Assembly] = AssemblyFactory.create_assembly(self.stp_file_path)  
        self.agent = ListMethodSimplificationAgent()    


    def step(self, action: float) -> Tuple[object, float, bool, dict]:
        is_out_of_range_dicimation, is_out_of_range_clustering, dfaces = self.agent.action(action)
        
        print(f"action: {action[0]}")
        print(f"face: {self.agent.simplified_assembly.get_face_number()}")
        
        self.current_action = action[0]
        observation: np.ndarry = self.agent.get_observation(self.current_action)
        
        
        terminated: bool = False
        if self.agent.simplified_assembly.get_face_number() <= 700:
            terminated = True

        reward: float = 0.0    
        reward += self.agent.get_reward(dfaces, is_out_of_range_dicimation, is_out_of_range_clustering, terminated)   
        print(f"reward: {reward}")
        
        return observation, reward, terminated, False, {} 
    
    def reset(self, seed=None, options=None):
        self.original_assembly: Assembly = AssemblyFactory.create_assembly(self.stp_file_path)  
        self.simplyfied_assembly: Optional[Assembly] = AssemblyFactory.create_assembly(self.stp_file_path)  
        self.agent = ListMethodSimplificationAgent()    
        
        return self.agent.get_observation(self.current_action), {}
    
    def render(self, mode: str = 'human') -> None:
        raise NotImplementedError()
    
    def close(self) -> None:
        raise NotImplementedError()
    
    def get_merged_assembly(self, original_assembly: Assembly, growing_ratio: float) -> Assembly:
        cluster_list = RegionGrowing(growing_ratio=growing_ratio).cluster(original_assembly)
        merged_assembly = \
            AssemblyFactory.create_merged_assembly(assembly = original_assembly, 
                                                cluster_list = cluster_list, 
                                                assembly_name = f"merged {original_assembly.part_name}")
        return merged_assembly
    
    
class IndexingMeshSimplificationEnv(gym.Env):
    metadata = {"render_modes": [None]}
    def __init__(self, 
                stp_file_path: str = "ButterflyValve.stp", 
                growing_ratio: float = 0.5) -> None:
        super(ListMethodSimplificationEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=0.0, high=0.9, shape=(1,), dtype=float)
        self.observation_space = gym.spaces.Box(low=0.1, high=0.9, shape=(10,), dtype=float)
        self.stp_file_path: str = stp_file_path  
        self.growing_ratio: float = growing_ratio   
        original_assembly: Assembly = AssemblyFactory.create_assembly(self.stp_file_path)  
        self.merged_assembly: Optional[Assembly] = self.get_merged_assembly(original_assembly)
        self.simplyfied_assembly: Optional[Assembly] = Assembly().copy_from_assembly(self.merged_assembly)   
        self.agent = ListMethodSimplificationAgent(original_assembly = original_assembly, 
                                        simplified_assembly = self.simplyfied_assembly)    
        self.current_action: float = 0.0


    def step(self, action: float) -> Tuple[object, float, bool, dict]:
        self.agent.action(action[0])
        print(f"action: {action[0]}")
        print(f"face: {self.agent.simplified_assembly.get_face_number()}")
        
        self.current_action = action[0]
        observation: np.ndarry = self.agent.get_observation(self.current_action)
        
        
        terminated: bool = False
        if self.agent.simplified_assembly.get_face_number() <= 700:
            terminated = True

        reward: float = 0.0    
        reward += self.agent.get_reward(terminated)   
        print(f"reward: {reward}")
        
        return observation, reward, terminated, False, {} 
    
    def reset(self, seed=None, options=None):
        original_assembly: Assembly = AssemblyFactory.create_assembly(self.stp_file_path)  
        self.merged_assembly: Optional[Assembly] = self.get_merged_assembly(original_assembly)
        
        simplified_assmebly = Assembly()
        simplified_assmebly.copy_from_assembly(self.merged_assembly)   
        self.simplyfied_assembly: Optional[Assembly] = simplified_assmebly
        self.agent = ListMethodSimplificationAgent(original_assembly = self.merged_assembly, 
                                        simplified_assembly = self.simplyfied_assembly)    
        return self.agent.get_observation(self.current_action), {}
    
    def render(self, mode: str = 'human') -> None:
        raise NotImplementedError()
    
    def close(self) -> None:
        raise NotImplementedError()
    
    def get_merged_assembly(self, original_assembly: Assembly) -> Assembly:
        cluster_list = RegionGrowing(growing_ratio=self.growing_ratio).cluster(original_assembly)
        merged_assembly = \
            AssemblyFactory.create_merged_assembly(assembly = original_assembly, 
                                                cluster_list = cluster_list, 
                                                assembly_name = f"merged {original_assembly.part_name}")
        return merged_assembly
    
if __name__ == "__main__":
    def merged_assembly_example():
        air_compressor: Assembly = AssemblyFactory.create_assembly("AirCompressor.stp")
        cluster_list = RegionGrowing(growing_ratio=0.5).cluster(air_compressor)
        merged_air_compressor = AssemblyFactory.create_merged_assembly(air_compressor, cluster_list, "AirCompressor")
        view_document = ViewDocument()
        merged_air_compressor.add_to_view_document(view_document)
        view_document.display() 
        return

    def list_method_mesh_simplification_env_example():
        env = ListMethodSimplificationEnv()
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_model_simplification_tensorboard/")    
        model.learn(total_timesteps=100000, tb_log_name="first_run")  
        model.save("ppo_model_simplification2")
        return
    circular_queue_mesh_simplification_env_example()
    # 관찰:  파트 수, 현재 파트 인덱스
    # 행동: 파트 수, 단순화 비율
