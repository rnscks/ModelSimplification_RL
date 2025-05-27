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


class Action:
    def __init__(self) -> None:
        self.action_space: gym.spaces.Box=gym.spaces.Box(low=0.01, high=0.9, shape=(3,), dtype=np.float32)        
        self.lb = self.action_space.low[0]
        self.ub = self.action_space.high[1]
    
    def quantize_action(self, continuous_action, max_range: int = 10):
        normalized_action = (continuous_action - self.lb) / self.ub
        discrete_action = int(round(normalized_action * (max_range - 1)))
        discrete_action = max(0, min(discrete_action, max_range - 1))
        return discrete_action
    
    def _rescale_action(self, continuous_action, target_lb: float, target_ub: float):
        normalized_action = (continuous_action - self.lb) / (self.ub - self.lb)
        scaled_action = target_lb + normalized_action * (target_ub - target_lb)
        return scaled_action
    
class SimplificationAction(Action):
    def __init__(self) -> None:
        super(SimplificationAction, self).__init__()
        return
    
    
    def action(self,
                assembly: Assembly,
                decimation_index: float,
                decimate_ratio: float) -> Assembly:
        decimation_index = self.quantize_action(decimation_index, len(assembly))    
        part = assembly[decimation_index]
        decimate_ratio = self._rescale_action(decimate_ratio, 0.01, 0.9)
        part.simplify(decimate_ratio)
        return assembly

class MergeAction(Action):
    def __init__(self) -> None:
        super(MergeAction, self).__init__()
        return
    
    def cal_cluster(self,
                    assembly: Assembly,
                    cluster_ratio: float) -> List[int]:
        matrix: np.ndarray = AdjGraph().graph_process(assembly)
        if len(matrix) == 1:
            return []
        cluster_ratio = np.clip(cluster_ratio, 0.1, 1.0)
        clusters: List[int] = HierarchicalCluster().cluster(matrix, cluster_ratio)
        return clusters
    
    def action(self,
                assembly: Assembly,
                clusters: List[int]) -> Assembly:  
        if clusters == []:
            return assembly
        parts: List[List[PartModel]] = [[] for _ in range(max(clusters) + 1)]   
        for idx, part in enumerate(assembly):
            parts[clusters[idx]].append(part)
            
        merged_assembly: Assembly = Assembly()
        for part_cluster in parts:
            if len(part_cluster) == 0:
                raise ValueError("Empty part cluster")  
            merged_part = part_cluster[0]
            for part in part_cluster[1:]:
                merged_part.merge_with(part)
            
            merged_assembly.parts.append(merged_part)
        merged_assembly.mesh = merged_assembly.merged_mesh()
        return merged_assembly


if __name__ == "__main__":
    def merge_action_example():
        merge_action = MergeAction()
        assembly = Assembly.load('data/set4/2_assembly51')
        assembly.simplify(0.1)
        
        print("Before Merge N Faces: ", assembly.n_faces())
        print("Before Merge N Parts: ", len(assembly))
        cluster = merge_action.cal_cluster(assembly, 0.5)
        assembly = merge_action.action(assembly, cluster)
        print("After Merge N Faces: ", assembly.n_faces())
        print("After Merge N Parts: ", len(assembly))
        assembly.display()


    def simplification_action_example():
        simplification_action = SimplificationAction()
        assembly = Assembly.load('data/set4/2_assembly51')
        assembly.simplify(0.5)
        
        print("Before Simplification N Faces: ", assembly.n_faces())
        print("Before Simplification N Parts: ", len(assembly))
        assembly = simplification_action.action(assembly, 15, 0.9)

        print("After Simplification N Faces: ", assembly.n_faces())
        print("After Simplification N Parts: ", len(assembly))
        assembly.display()