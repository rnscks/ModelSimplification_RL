from stable_baselines3 import PPO   
from src.environment import LMSCEnv

def list_method_mesh_simplification_env_example():
    env = LMSCEnv()
    model = PPO("MlpPolicy", env, verbose=1)    
    model.learn(total_timesteps=100000)  
    model.save("ppo_model_simplification2_c")
    return


list_method_mesh_simplification_env_example()