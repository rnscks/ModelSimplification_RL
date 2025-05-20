from stable_baselines3 import PPO  
import gymnasium.spaces as spaces
from typing import List
import os

from src.mesh.model import Assembly
from src.rl.environment import TestEnv

def test(
    rl_file_name: str,
    step_dirs: List[str] = ["test_set"], 
    target_sim_ratio: float = 0.95, 
    max_time_step: int = 30) -> Assembly:
    model = PPO.load(rl_file_name)  
    
    for assembly_dir in step_dirs:
        test_model: Assembly = Assembly.load(assembly_dir)
        test_env = TestEnv(
            test_model,
            max_time_step=max_time_step,
            target_sim_ratio=target_sim_ratio)
        obs, _ = test_env.reset()    
        done = False
        
        for _ in range(max_time_step):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info, _ = test_env.step(action)    
        
            
            if done:
                print("Episode finished")
                return test_env.ret_assembly
    return


if __name__ == "__main__":
    test(
        rl_file_name="GCN_BASIC_STEP4000",
        step_dirs=[
            "data/set4/21_assembly67"],
        target_sim_ratio=0.95,
        max_time_step=50)  