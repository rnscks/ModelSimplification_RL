from stable_baselines3 import PPO   
from typing import Optional, Tuple  
import time
from src.model_3d.cad_model import AssemblyFactory, ViewDocument, Assembly
from src.model_3d.model_util import ChamferDistance, GirvanNewman
from src.environment import LMSCEnv, LMSEnv, LMSEnvWithCluster
from src.agent import LMSAgent, LMSCAgent   
from src.model_3d.file_system import FileReader 
#from torch.utils.tensorboard import SummaryWriter



def list_method_mesh_simplification_env_example():
    """
    메시 단순화 환경 예제를 실행하는 함수입니다.

    Returns:
        None

    Parameters:
        None
    """
    env = LMSEnv()
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_model_simplification_tensorboard/")    
    model.learn(total_timesteps=100000, tb_log_name="first_run_")  
    model.save("ppo_model_simplification_grivan_newman")
    return

def list_method_mesh_simplification_with_cluster_env_example():
    """
    클러스터를 진행하는 환경에서 메시 단순화를 위한 리스트 방법 예제를 실행합니다.

    Return:
        None

    Parameters:
        None
    """
    env = LMSEnvWithCluster()
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_model_simplification_tensorboard/")    
    model.learn(total_timesteps=50000, tb_log_name="first_run_")  
    model.save("ppo_model_simplification_grivan_newman")
    return

def list_method_mesh_simplification_cluster_env_example():
    """
    메시 단순화와 클러스터를 반복적으로 진행하는 환경 예제에 대한 리스트 메서드입니다.

    Return:
        None

    Parameters:
        None
    """
    env = LMSCEnv()
    
    model = PPO("MlpPolicy", env, verbose=1)    
    model.learn(total_timesteps=50000)  
    model.save("ppo_model_simplification_with_markov_cluster")
    return

def quantize_action(continuous_action, part_number):
    """
    주어진 연속적인 액션을 이산화된 액션으로 변환하는 함수입니다.

    Parameters:
        continuous_action (float): 연속적인 액션 값
        part_number (int): 액션을 나눌 구간 수

    Return:
        int: 이산화된 액션 값
    """
    discrete_action = int(round(continuous_action * (part_number - 1)))
    discrete_action = max(0, min(discrete_action, part_number- 1))
    return discrete_action

def test_list_method_mesh_simplification_env_example():
    """
    메시 단순화 모델을 테스트하는 예제 함수입니다.

    Return:
        None

    Parameters:
        None
    """
    
    env = LMSEnvWithCluster()
    model = FileReader.read_rl_model("ppo_model_simplification_grivan_newman", "ppo")
    model.set_env(env)  
    stp_file_path = "ButterflyValve.stp"
    cluster_list = GirvanNewman().cluster(AssemblyFactory.create_assembly(stp_file_path))   
    original_assembly: Assembly = AssemblyFactory.create_merged_assembly(
        AssemblyFactory.create_assembly(stp_file_path), 
        cluster_list, 
        "butterfly_valve")  
    simplyfied_assembly: Assembly = AssemblyFactory.create_merged_assembly(
        AssemblyFactory.create_assembly(stp_file_path), 
        cluster_list, 
        "butterfly_valve")  
    
    agent = LMSAgent(original_assembly, simplyfied_assembly)  
    model_length = len(agent.original_assembly.part_model_list)
    print(f"model length: {model_length}")
    start_time = time.time()
    step = 0
    
    while True:
        step += 1
        action, _ = model.predict(agent.get_observation(), deterministic=True)
        decimation_index = quantize_action(action[0], 8)
        print(decimation_index, action[1])
        decimation_ratio = action[1]    
        agent.action(decimation_index, decimation_ratio)
        #print(agent.simplified_assembly.get_face_number(), action[1])
        if agent.simplified_assembly.get_face_number() <= 9000:
            break           
        if step == 100:
            break   

    print("time: ", time.time() - start_time)
    print("step: ", step)   
    cd = ChamferDistance().evaluate(agent.simplified_assembly, agent.original_assembly)    
    print("cd: ", cd)
    print("number of faces: ", agent.simplified_assembly.get_face_number()) 
    view_document = ViewDocument()
    agent.simplified_assembly.add_to_view_document(view_document)
    view_document.display()
    return

def simplfication_example():
    """
    동일한 퍼센트로 모델 단순화를 진행하는 함수입니다.

    Return:
        None

    Parameters:
        None
    """
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


## TEST FUNCTION
def test_list_method_mesh_simplification_env_iterable_cluster_example():
    """
    메시 단순화 모델을 테스트하는 예제 함수입니다.

    Return:
        None
    
    Parameters:
        None
    """
    model = FileReader.read_rl_model("ppo_model_simplification_grivan_newman", "ppo")
    # model.set_env(env)  
    
    stp_file_path = "ControlValve.stp"
    original_assembly: Assembly = AssemblyFactory.create_assembly(
        stp_file_path=stp_file_path) 
    
    simplyfied_assembly: Assembly = AssemblyFactory.create_assembly(
        stp_file_path=stp_file_path)  
    
    agent = LMSCAgent(original_assembly, simplyfied_assembly)
    model_length = len(agent.original_assembly.part_model_list)
    print(f"model length: {model_length}")
    start_time = time.time()
    step = 0
    
    while True:
        step += 1
        action, _ = model.predict(agent.get_observation(), deterministic=True)
        decimation_index = quantize_action(action[0], 8)
        print(decimation_index, action[1])
        decimation_ratio = action[1]    
        agent.action(decimation_index, decimation_ratio)
        # print(agent.simplified_assembly.get_face_number(), action[1])
        if agent.simplified_assembly.get_face_number() <= 9000:
            break           
        if step == 100:
            break   

    print("time: ", time.time() - start_time)
    print("step: ", step)   
    cd = ChamferDistance().evaluate(agent.simplified_assembly, agent.original_assembly)    
    print("cd: ", cd)
    print("number of faces: ", agent.simplified_assembly.get_face_number()) 
    view_document = ViewDocument()
    agent.simplified_assembly.add_to_view_document(view_document)
    view_document.display()
    return


if __name__ == "__main__":
    ## 학습 코드
    list_method_mesh_simplification_cluster_env_example()
    ## 테스트 코드
    # test_list_method_mesh_simplification_env_iterable_cluster_example()