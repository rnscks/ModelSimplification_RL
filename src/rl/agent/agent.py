from itertools import product
from enum import Enum
from typing import Tuple, List
import gymnasium as gym
import numpy as np

from src.graph.cluster import HierarchicalCluster
from src.graph.preprocess import AdjGraph, GRAPH
from src.mesh.metrics import ChamferDistance, METRIC
from src.mesh.model import Assembly, PartModel
from src.rl.util import profile_time
from src.rl.agent.action import SimplificationAction, MergeAction
from src.rl.agent.observation import Observation

class SimplificationAgent:
    def __init__(self, 
                 observation: Observation) -> None:
        self.observation: Observation = observation
        self.simplification_action = SimplificationAction()
        self.cluster_action = MergeAction()    
        
        self.observation_space: gym.spaces = observation.observation_space
        self.action_space: gym.spaces.Box = self.simplification_action.action_space
        return
    
    
    def action(self,
                simplified_assembly: Assembly,
                original_assembly: Assembly,
                decimation_index: float, 
                decimate_ratio: float, 
                cluster_ratio: float = 3.0) -> Tuple[Assembly, Assembly]:
        simplified_assembly = self.simplification_action.action(simplified_assembly, decimation_index, decimate_ratio)
        
        original_matrix: np.ndarray = AdjGraph().graph_process(original_assembly)
        matrix: np.ndarray = AdjGraph().graph_process(simplified_assembly)
        for i, j in product(range(len(matrix)), range(len(matrix))):
            if original_matrix[i][j] != 0 and matrix[i][j] == 0:
                return False
        
        clusters = self.cluster_action.cal_cluster(simplified_assembly, cluster_ratio)
        simplified_assembly = self.cluster_action.action(simplified_assembly, clusters)
        original_assembly = self.cluster_action.action(original_assembly, clusters)
        return simplified_assembly, original_assembly
    
    def get_observation(self,
                        simplified_assembly: Assembly,
                        original_assembly: Assembly) -> dict:
        return self.observation.get_observation(simplified_assembly, original_assembly) 
