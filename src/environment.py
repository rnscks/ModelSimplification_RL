import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import numpy as np  
from typing import Optional, Tuple

import time
from src.model_3d.cad_model import AssemblyFactory, ViewDocument, Assembly
from src.model_3d.model_util import RegionGrowing, GirvanNewman
from src.agent import LMSAgent, LMSCAgent
import matplotlib.pyplot as plt 

class LMSEnv(gym.Env):
    metadata = {"render_modes": [None]}
    def __init__(self, 
                stp_file_path: str = "ButterflyValve.stp", 
                part_number: int = 14) -> None:
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
        """
        환경에서 한 스텝을 진행합니다.

        Parameters:
            action (float): 에이전트의 액션

        Returns:
            Tuple[object, float, bool, dict]: 다음 관측값, 보상, 종료 여부, 추가 정보
        """
        decimation_index = self.quantize_action(action[0])  
        decimation_ratio = action[1]
        self.agent.action(decimation_index, decimation_ratio)
        
        self.time_step += 1
        reward: float = 0.0  
        observation: np.ndarry = self.agent.get_observation()
        terminated: bool = False
        total_face_number = self.agent.simplified_assembly.get_face_number()
        
        if total_face_number <= 7000:
            print("success!!")
            terminated = True
            reward += 1.0  
            
        if self.time_step >= 50:
            print("failed!!")
            terminated = True   
            reward -= 1.0
    
        reward += self.agent.get_reward(decimation_index, terminated)   
        return observation, reward, terminated, False, {}
    def reset(self, 
            seed=None, 
            options=None):
        """
        환경을 초기화합니다.

        Parameters:
            seed (Any): 난수 시드 (기본값: None)
            options (Any): 옵션 (기본값: None)

        Returns:
            Tuple[object, dict]: 초기 관측값, 추가 정보
        """
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
        """
        연속적인 액션을 양자화합니다.

        Parameters:
            continuous_action (float): 연속적인 액션 값

        Returns:
            int: 양자화된 액션 값
        """
        # 0 ~ 1 범위의 값을 0 ~ 13 범위로 양자화
        discrete_action = int(round(continuous_action * (self.part_number - 1)))
        # 결과가 0 ~ 13 범위 내에 있는지 확인
        discrete_action = max(0, min(discrete_action, self.part_number- 1))
        return discrete_action
    
class LMSCEnv(gym.Env):
    metadata = {"render_modes": [None]}
    def __init__(self, 
                stp_file_path: str = "ControlValve.stp") -> None:
        super(LMSCEnv, self).__init__()
        self.action_space = spaces.Box(low=0.01, high=1.0, shape=(4,), dtype=float)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(30 * 2 + 1, ), dtype=float)
        
        self.stp_file_path: str = stp_file_path  

        original_assembly: Assembly = AssemblyFactory.create_assembly(self.stp_file_path)  
        simplyfied_assembly: Optional[Assembly] = AssemblyFactory.create_assembly(self.stp_file_path)  
        self.agent = LMSCAgent(original_assembly, simplyfied_assembly)  
        self.time_step: int = 0

    
    def step(self, action: float) -> Tuple[object, float, bool, dict]:
        """
        환경에서 한 스텝을 진행합니다.

        Parameters:
            action (float): 에이전트의 액션

        Return:
            observation (object): 관측값
            reward (float): 보상값
            terminated (bool): 종료 여부
            info (dict): 추가 정보
        """
        decimation_index = action[0]
        decimation_ratio = action[1]
        cluster_index = action[2]
        cluster_ratio = action[3]
        reward: float = 0.0  

        reward += self.agent.action(decimation_index, 
                            decimation_ratio,
                            cluster_index,
                            cluster_ratio)        
        self.time_step += 1
        observation: np.ndarry = self.agent.get_observation()
        terminated: bool = False
        
        total_face = self.agent.simplified_assembly.get_face_number()
        print(total_face)
        if total_face <= 5000:
            terminated = True
            reward += 1.0  
            reward += 1
            
        if self.time_step >= 100:
            terminated = True   
            reward -= 1.0
            
        reward += self.agent.get_reward(terminated)   
        if terminated:
            print("terminated!!")
        return observation, reward, terminated, False, {} 
    
    def reset(self, 
            seed=None, 
            options=None):
        """
        환경을 초기화합니다.

        Parameters:
            seed (optional): 난수 시드 값
            options (optional): 옵션

        Return:
            observation (object): 초기 관측값
            info (dict): 추가 정보
        """
        self.original_assembly: Assembly = AssemblyFactory.create_assembly(self.stp_file_path)  
        self.simplyfied_assembly: Optional[Assembly] = AssemblyFactory.create_assembly(self.stp_file_path)  
        self.agent = LMSCAgent(self.original_assembly, self.simplyfied_assembly)
        self.time_step = 0
        return self.agent.get_observation(), {}
    
    def render(self, mode: str = 'human') -> None:
        raise NotImplementedError()
    
    def close(self) -> None:
        raise NotImplementedError()
    

class LMSEnvWithCluster(gym.Env):
    metadata = {"render_modes": [None]}
    def __init__(self, stp_file_path: str = "ButterflyValve.stp") -> None:
        super(LMSEnvWithCluster, self).__init__()
        self.action_space = spaces.Box(low=0.01, high=1.0, shape=(2,), dtype=float)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(30 * 2 + 1, ), dtype=float)
        self.stp_file_path: str = stp_file_path  
        
        cl = GirvanNewman().cluster(AssemblyFactory.create_assembly(stp_file_path))  
        self.original_assembly: Assembly = AssemblyFactory.create_merged_assembly(AssemblyFactory.create_assembly(stp_file_path), cl, "r")  
        self.simplyfied_assembly: Optional[Assembly] = AssemblyFactory.create_merged_assembly(AssemblyFactory.create_assembly(stp_file_path), cl, "cr")  
        self.agent = LMSAgent(self.original_assembly, self.simplyfied_assembly)  
        self.time_step: int = 0
        self.part_number: int = len(self.original_assembly.part_model_list) 


    def step(self, action: float) -> Tuple[object, float, bool, dict]:
        """
        환경에서 한 스텝을 진행합니다.

        Parameters:
            action (float): 에이전트의 액션

        Return:
            observation (object): 관측값
            reward (float): 보상값
            terminated (bool): 종료 여부
            info (dict): 추가 정보
        """
        decimation_index = self.quantize_action(action[0])  
        decimation_ratio = action[1]
        self.agent.action(decimation_index, decimation_ratio)
        
        self.time_step += 1
        reward: float = 0.0  
        observation: np.ndarry = self.agent.get_observation()
        terminated: bool = False
        total_face_number = self.agent.simplified_assembly.get_face_number()
        
        if total_face_number <= 7000:
            print("success!!")
            terminated = True
            reward += 10.0
            
        if self.time_step >= 50:
            print("failed!!")
            terminated = True   
            reward -= 10.0
    
        reward += self.agent.get_reward(decimation_index, terminated)   
        return observation, reward, terminated, False, {} 
    
    def reset(self, 
            seed=None, 
            options=None):
        """
        환경을 초기화합니다.

        Parameters:
            seed (optional): 시드 값
            options (optional): 옵션

        Return:
            observation (object): 초기 관측값
            info (dict): 추가 정보
        """
        cl = GirvanNewman().cluster(AssemblyFactory.create_assembly(self.stp_file_path))  
        
        self.original_assembly: Assembly = AssemblyFactory.create_merged_assembly(AssemblyFactory.create_assembly(self.stp_file_path), cl, "r")  
        self.simplyfied_assembly: Optional[Assembly] = AssemblyFactory.create_merged_assembly(AssemblyFactory.create_assembly(self.stp_file_path), cl, "cr")          
        self.agent = LMSAgent(self.original_assembly, self.simplyfied_assembly)
        self.time_step = 0
        return self.agent.get_observation(), {}
    
    def render(self, mode: str = 'human') -> None:
        raise NotImplementedError()
    
    def close(self) -> None:
        raise NotImplementedError()
    
    def quantize_action(self, continuous_action):
        """
        액션을 양자화합니다.

        Parameters:
            continuous_action: 연속적인 액션 값

        Return:
            discrete_action: 양자화된 액션 값
        """
        # 0 ~ 1 범위의 값을 0 ~ 13 범위로 양자화
        discrete_action = int(round(continuous_action * (self.part_number - 1)))
        # 결과가 0 ~ 13 범위 내에 있는지 확인
        discrete_action = max(0, min(discrete_action, self.part_number- 1))
        return discrete_action