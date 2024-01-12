import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import numpy as np  
from typing import Optional, Tuple

import time
from src.model_3d.cad_model import AssemblyFactory, ViewDocument, Assembly
from src.model_3d.model_util import RegionGrowing
from src.agent import LMSAgent, LMSCAgent
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt 

class LMSEnv(gym.Env):
    metadata = {"render_modes": [None]}
    def __init__(self, stp_file_path: str = "ButterflyValve.stp", part_number: int = 14) -> None:
        super(LMSEnv, self).__init__()
        self.action_space = spaces.Box(low=0.01, high=1.0, shape=(2,), dtype=float)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(30 * 2 + 1, ), dtype=float)
        
        self.stp_file_path: str = stp_file_path  
        
        self.original_assembly: Assembly = AssemblyFactory.create_assembly(self.stp_file_path)  
        self.simplyfied_assembly: Optional[Assembly] = AssemblyFactory.create_assembly(self.stp_file_path)  
        self.agent = LMSAgent(self.original_assembly, self.simplyfied_assembly)  
        self.time_step: int = 0
        self.part_number: int = part_number 


    def step(self, action: float) -> Tuple[object, float, bool, dict]:
        decimation_index = self.quantize_action(action[0])  
        decimation_ratio = action[1]
        dfaces_ratio = self.agent.action(decimation_index, decimation_ratio)
        
        self.time_step += 1
        reward: float = 0.0  
        observation: np.ndarry = self.agent.get_observation()
        terminated: bool = False
        
        if self.agent.simplified_assembly.get_face_number() <= 2000:
            terminated = True
            reward += 1.0  
            
        if self.time_step >= 1000:
            terminated = True   
            reward -= 1.0
    
        reward += self.agent.get_reward(decimation_index, dfaces_ratio, terminated)   
        reward += 1 / self.time_step
        return observation, reward, terminated, False, {} 
    
    def reset(self, seed=None, options=None):
        self.original_assembly: Assembly = AssemblyFactory.create_assembly(self.stp_file_path)  
        self.simplyfied_assembly: Optional[Assembly] = AssemblyFactory.create_assembly(self.stp_file_path)  
        self.agent = LMSAgent(self.original_assembly, self.simplyfied_assembly)
        self.time_step = 0
        return self.agent.get_observation(), {}
    
    def render(self, mode: str = 'human') -> None:
        raise NotImplementedError()
    
    def close(self) -> None:
        raise NotImplementedError()
    
    def quantize_action(self, continuous_action):
        # 0 ~ 1 범위의 값을 0 ~ 13 범위로 양자화
        discrete_action = int(round(continuous_action * (self.part_number - 1)))
        # 결과가 0 ~ 13 범위 내에 있는지 확인
        discrete_action = max(0, min(discrete_action, self.part_number- 1))
        return discrete_action
    
class LMSCEnv(gym.Env):
    metadata = {"render_modes": [None]}
    def __init__(self, stp_file_path: str = "ControlValve.stp") -> None:
        super(LMSCEnv, self).__init__()
        self.action_space = spaces.Box(low=0.01, high=1.0, shape=(4,), dtype=float)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(30 * 2 + 1, ), dtype=float)
        
        self.stp_file_path: str = stp_file_path  

        original_assembly: Assembly = AssemblyFactory.create_assembly(self.stp_file_path)  
        simplyfied_assembly: Optional[Assembly] = AssemblyFactory.create_assembly(self.stp_file_path)  
        self.agent = LMSCAgent(original_assembly, simplyfied_assembly)  
        self.time_step: int = 0


    def step(self, action: float) -> Tuple[object, float, bool, dict]:
        
        decimation_index = action[0]
        decimation_ratio = action[1]
        cluster_index = action[2]
        cluster_ratio = action[3]
        self.agent.action(decimation_index, 
                            decimation_ratio,
                            cluster_index,
                            cluster_ratio)        
                    
        self.time_step += 1
        reward: float = 0.0  
        observation: np.ndarry = self.agent.get_observation()
        terminated: bool = False
        print(f"face: {self.agent.simplified_assembly.get_face_number()}")
        if self.agent.simplified_assembly.get_face_number() <= 4000:
            terminated = True
            reward += 1.0  
            
        if self.time_step >= 1000:
            terminated = True   
            reward -= 1.0
        reward += self.agent.get_reward(terminated)   
        if terminated:
            print("terminated!!")
        return observation, reward, terminated, False, {} 
    
    def reset(self, seed=None, options=None):
        self.original_assembly: Assembly = AssemblyFactory.create_assembly(self.stp_file_path)  
        self.simplyfied_assembly: Optional[Assembly] = AssemblyFactory.create_assembly(self.stp_file_path)  
        self.agent = LMSCAgent(self.original_assembly, self.simplyfied_assembly)
        self.time_step = 0
        return self.agent.get_observation(), {}
    
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
        env = LMSEnv()
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_model_simplification_tensorboard/")    
        model.learn(total_timesteps=100000, tb_log_name="first_run_")  
        model.save("ppo_model_simplification2_")
        return
    
    def load_list_method_mesh_simplification_env_example():
        env = LMSEnv()
        model = PPO.load("ppo_model_simplification2_")
        model.set_env(env)
        model._total_timesteps = 0
        model.learn(total_timesteps=100000, tb_log_name="first_run_", reset_num_timesteps=False)
        model.save("ppo_model_simplification2_")
        return
    
    def quantize_action(continuous_action, num_intervals=14):
        # 0 ~ 1 범위의 값을 0 ~ 13 범위로 양자화
        discrete_action = int(round(continuous_action * (num_intervals - 1)))
        # 결과가 0 ~ 13 범위 내에 있는지 확인
        discrete_action = max(0, min(discrete_action, num_intervals - 1))
        return discrete_action
    
    def list_method_mesh_simplification_env_example():
        env = LMSCEnv()
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_model_simplification_tensorboard/")    
        model.learn(total_timesteps=100000, tb_log_name="first_run_c")  
        model.save("ppo_model_simplification2_c")
        return
    
    def test_list_method_mesh_simplification_env_example():
        env = LMSEnv()
        model = PPO.load("ppo_model_simplification2_")
        stp_file_path = "ButterflyValve.stp"
        original_assembly: Assembly = AssemblyFactory.create_assembly(stp_file_path)  
        simplyfied_assembly: Optional[Assembly] = AssemblyFactory.create_assembly(stp_file_path)  
        agent = LMSAgent(original_assembly, simplyfied_assembly)  
        start_time = time.time()
        while True:
            action, _ = model.predict(agent.get_observation())
            decimation_index = quantize_action(action[0], 14)
            actions.append(action[1])
            decimation_ratio = action[1]    
            agent.action(decimation_index, decimation_ratio)
            print(agent.simplified_assembly.get_face_number(), action[1])
            if agent.simplified_assembly.get_face_number() <= 5000:
                break           
        print("time: ", time.time() - start_time)
        view_document = ViewDocument()
        agent.simplified_assembly.add_to_view_document(view_document)
        view_document.display()
        plt.hist(actions, bins=14, alpha=0.5, color='blue', edgecolor='black')
        plt.show()
        return
    list_method_mesh_simplification_env_example()

