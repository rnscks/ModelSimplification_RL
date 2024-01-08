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
        self.action_space = spaces.Box(low=0.0, high=0.9, shape=(2,), dtype=float)
        self.observation_space = spaces.Box(low=0.1, high=0.9, shape=(30 * 3 + 3,), dtype=float)
        
        self.stp_file_path: str = stp_file_path  
        
        self.original_assembly: Assembly = AssemblyFactory.create_assembly(self.stp_file_path)  
        self.simplyfied_assembly: Optional[Assembly] = AssemblyFactory.create_assembly(self.stp_file_path)  
        self.agent = ListMethodSimplificationAgent(self.original_assembly, self.simplyfied_assembly)    


    def step(self, action: float) -> Tuple[object, float, bool, dict]:
        is_out_of_range_dicimation, dfaces = self.agent.action(action)
        decimation_index = int(action[0] * 10)
        decimation_ratio = action[1]    
        
        observation: np.ndarry = self.agent.get_observation(decimation_index, decimation_ratio)

        terminated: bool = False
        if self.agent.simplified_assembly.get_face_number() <= 700:
            terminated = True

        reward: float = 0.0    
        reward += self.agent.get_reward(dfaces, is_out_of_range_dicimation, terminated)   

        return observation, reward, terminated, False, {} 
    
    def reset(self, seed=None, options=None):
        self.original_assembly: Assembly = AssemblyFactory.create_assembly(self.stp_file_path)  
        self.simplyfied_assembly: Optional[Assembly] = AssemblyFactory.create_assembly(self.stp_file_path)  
        self.agent = ListMethodSimplificationAgent(self.original_assembly, self.simplyfied_assembly)
        
        return self.agent.get_observation(0, 0), {}
    
    def render(self, mode: str = 'human') -> None:
        raise NotImplementedError()
    
    def close(self) -> None:
        raise NotImplementedError()
    
        
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
    list_method_mesh_simplification_env_example()
    # 관찰:  파트 수, 현재 파트 인덱스
    # 행동: 파트 수, 단순화 비율
