from stable_baselines3 import PPO   
from stable_baselines3.common.env_checker import check_env
from typing import List
import os

from src.rl.environment import TrainEnv 
from src.rl.agent.agent import SimplificationAgent
from src.rl.agent.observation import BasicObservation, PointNetObservation
from src.rl.feature_extractor import GNN_PointNetExtractor, GNNExtractor



RL_MODEL_DIR = 'models'
def curriculum_train(
    agent: SimplificationAgent, 
    task_set: List[str] = ["test_set"], 
    rl_file_name: str = "",
    save_file_name: str = "TEST_GCN_MODELSIMPLIIIFCATION",
    max_time_step: int = 30,
    total_timesteps:int =20000 ) -> None:
    env = TrainEnv(
        agent,
        task_set, 
        max_time_step=max_time_step)       
    check_env(env)  
    
    if rl_file_name == "":
        policy_kwargs = dict(
            features_extractor_class=GNNExtractor,
            features_extractor_kwargs=dict(features_dim=128)
        )

        model = PPO(
            "MultiInputPolicy", 
            env, 
            policy_kwargs=policy_kwargs,
            verbose=1, 
            tensorboard_log='./logs/',
            n_steps=2048,
            batch_size=64,
            n_epochs=10,)
    else:
        rl_file_path = os.path.join('models', rl_file_name) 
        model = PPO.load(
            rl_file_path, 
            env=env, 
            tensorboard_log='./logs/')
    model.learn(total_timesteps=total_timesteps, tb_log_name='run')  
    # models 디렉토리에 모델을 저장합니다.
    file_path = os.path.join(RL_MODEL_DIR, save_file_name)
    model.save(file_path)
    return


if __name__ == "__main__":
    agent1 = SimplificationAgent(
        observation=BasicObservation(
            cd_option=False, 
            vl_option=False))
    agent2 = SimplificationAgent(
        observation=PointNetObservation(
            cd_option=True, 
            vl_option=False))   
    agent3 = SimplificationAgent(
        observation=PointNetObservation(
            cd_option=False, 
            vl_option=True))
    agent4 = SimplificationAgent(
        observation=BasicObservation(
            cd_option=True, 
            vl_option=True))

    agents = [agent1, agent2, agent3, agent4]
    for i, agent in enumerate(agents):
        curriculum_train(
            agent=agent,
            task_set=[
                "data/set1",
                "data/set2",
                "data/set3",
                "data/set4"],
            rl_file_name="",
            save_file_name=f"PPO_MCL_TEST_MODEL{i}",
            max_time_step=50,
            total_timesteps=20000)