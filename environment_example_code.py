from stable_baselines3 import PPO   
from typing import Optional, Tuple  
import time
from src.model_3d.cad_model import AssemblyFactory, ViewDocument, Assembly
from src.model_3d.model_util import ChamferDistance, RegionGrowing  
from src.environment import LMSCEnv, LMSEnv, LMSEnvWithCluster
from src.agent import LMSAgent, LMSCAgent   
from torch.utils.tensorboard import SummaryWriter



def list_method_mesh_simplification_env_example():
    env = LMSEnv()
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_model_simplification_tensorboard/")    
    model.learn(total_timesteps=100000, tb_log_name="first_run_")  
    model.save("ppo_model_simplification_grivan_newman")
    return

def list_method_mesh_simplification_with_cluster_env_example():
    env = LMSEnvWithCluster()
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_model_simplification_tensorboard/")    
    model.learn(total_timesteps=100000, tb_log_name="first_run_")  
    model.save("ppo_model_simplification_grivan_newman")
    return

def list_method_mesh_simplification_cluster_env_example():
    env = LMSCEnv()
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_model_simplification_tensorboard/")    
    model.learn(total_timesteps=100000, tb_log_name="second_run_")  
    model.save("ppo_model_simplification2_c")
    return

def quantize_action(continuous_action, part_number):
    discrete_action = int(round(continuous_action * (part_number - 1)))
    discrete_action = max(0, min(discrete_action, part_number- 1))
    return discrete_action

def test_list_method_mesh_simplification_env_example():
    env = LMSEnv()
    model = PPO.load("ppo_model_simplification")
    stp_file_path = "ButterflyValve.stp"
    original_assembly: Assembly = AssemblyFactory.create_assembly(stp_file_path)  
    simplyfied_assembly: Optional[Assembly] = AssemblyFactory.create_assembly(stp_file_path)  
    agent = LMSAgent(original_assembly, simplyfied_assembly)  
    start_time = time.time()
    while True:
        action, _ = model.predict(agent.get_observation())
        print(action)
        decimation_index = quantize_action(action[0], 14)
        decimation_ratio = action[1]    
        agent.action(decimation_index, decimation_ratio)
        #print(agent.simplified_assembly.get_face_number(), action[1])
        if agent.simplified_assembly.get_face_number() <= 9000:
            break           
    print("time: ", time.time() - start_time)
    cd = ChamferDistance().evaluate(agent.simplified_assembly, agent.original_assembly)    
    print("cd: ", cd)
    print("number of faces: ", agent.simplified_assembly.get_face_number()) 
    view_document = ViewDocument()
    agent.simplified_assembly.add_to_view_document(view_document)
    view_document.display()
    return

def simplfication_example():
    stp_file_path = "ButterflyValve.stp"
    original_assembly: Assembly = AssemblyFactory.create_assembly(stp_file_path) 
    simplified_assembly: Optional[Assembly] = AssemblyFactory.create_assembly(stp_file_path)    
    start_time = time.time() 
    for part in simplified_assembly.part_model_list:
        part.simplify(0.24) 

    print(f"time: {time.time() - start_time}")  
    cd = ChamferDistance().evaluate(simplified_assembly, original_assembly)   
    print(f"cd: {cd}")  
    print(f"number of faces: {simplified_assembly.get_face_number()}")    
    
    view_document = ViewDocument()
    simplified_assembly.add_to_view_document(view_document)
    view_document.display()
    pass

if __name__ == "__main__":
    model = PPO.load("ppo_model_simplification2")
    list_method_mesh_simplification_env_example()