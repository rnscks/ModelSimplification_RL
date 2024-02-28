from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import TopoDS_Compound
# for example code
from OCC.Display.SimpleGui import init_display
from stable_baselines3 import PPO, DQN
import os
from stable_baselines3.common.base_class import BaseAlgorithm

BERP_DIR_NAME = "brep_models"
RL_MODEL_DIR_NAME = "rl_models"

class FileReader:
    @classmethod
    def read_rl_model(cls, file_name: str, model_name: str) -> BaseAlgorithm:
        file_path = os.path.join(RL_MODEL_DIR_NAME, file_name)     
        model = dict(ppo=PPO, dqn=DQN) 
        return model[model_name].load(file_path)
    
    @classmethod  
    def read_stp_file(cls, file_name:str) -> TopoDS_Compound:
        """
        파일 이름을 입력받아 STP 파일을 읽고, TopoDS_Compound 객체를 반환합니다.

        Parameters:
            file_name (str): 읽을 STP 파일의 이름

        Return:
            TopoDS_Compound: STP 파일에서 읽은 3D 모델의 TopoDS_Compound 객체
        """
        stpReader = STEPControl_Reader()
        if os.path.exists(BERP_DIR_NAME) == False:
            os.mkdir(BERP_DIR_NAME)
            raise FileNotFoundError(f"{BERP_DIR_NAME} 폴더가 없어서 새로 만들었습니다.")
        
        file_path = os.path.join(BERP_DIR_NAME, file_name)  
        stpReader.ReadFile(file_path)
        stpReader.TransferRoots()
        
        return stpReader.Shape()
    
if __name__ == "__main__":
    a,b,c,d = init_display()
    brep_shape: TopoDS_Compound = FileReader.read_stp_file("AirCompressor.stp")
    a.DisplayShape(brep_shape, update=True)
    a.FitAll()
    b()