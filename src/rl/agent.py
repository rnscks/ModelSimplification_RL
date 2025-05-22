from itertools import product
from enum import Enum
from typing import Tuple, List
import gymnasium as gym
import numpy as np

from src.graph.cluster import HierarchicalCluster
from src.graph.preprocess import Preprocess, GRAPH
from src.mesh.metrics import ChamferDistance, METRIC
from src.mesh.model import Assembly, PartModel
from src.rl.util import profile_time

class Action:
    def __init__(self, action_space: gym.spaces.Box) -> None:
        self.action_space: gym.spaces.Box = action_space
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
    def __init__(self,
                 action_space: gym.spaces.Box=gym.spaces.Box(low=0.01, high=0.9, shape=(3,), dtype=float)) -> None:
        super(SimplificationAction, self).__init__(action_space)
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
    def __init__(self, 
                 action_space: gym.spaces.Box=gym.spaces.Box(low=0.01, high=0.9, shape=(3,), dtype=float)) -> None:
        super(MergeAction, self).__init__(action_space)
        return
    
    def cal_cluster(self,
                    assembly: Assembly,
                    cluster_ratio: float) -> List[int]:
        matrix: np.ndarray = Preprocess().graph_process(assembly)
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

class SimplificationAgent:
    def __init__(self, 
                 action_space: gym.spaces.Box) -> None:
        self.action_space: gym.spaces.Box = action_space
        self.simplification_action = SimplificationAction(action_space)
        self.cluster_action = MergeAction(action_space)       
        return
    
    
    def action(self,
                simplified_assembly: Assembly,
                original_assembly: Assembly,
                decimation_index: float, 
                decimate_ratio: float, 
                cluster_ratio: float = 3.0) -> Tuple[Assembly, Assembly]:
        simplified_assembly = self.simplification_action.action(simplified_assembly, decimation_index, decimate_ratio)
        
        original_matrix: np.ndarray = Preprocess().graph_process(original_assembly)
        matrix: np.ndarray = Preprocess().graph_process(simplified_assembly)
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
        node_features = self._calculate_node_features(simplified_assembly, original_assembly)
        adjacency_matrix = Preprocess().graph_process(simplified_assembly)  
        
        edge_index = []
        edge_attr = []
        for i, j in product(range(len(adjacency_matrix)), range(len(adjacency_matrix))):   
            if adjacency_matrix[i][j] != 0:
                edge_index.append([i, j])
                edge_attr.append(adjacency_matrix[i][j])
        
        edge_index = np.array(edge_index)
        edge_attr = np.array(edge_attr).reshape(-1, 1)
        
        padded_x = np.zeros((GRAPH.MAX_NODES.value, GRAPH.NODE_DIM.value))
        padded_x[:node_features.shape[0], :] = node_features
        padded_edge_index = np.zeros((GRAPH.MAX_EDGES.value, 2), dtype=np.int64)
        if len(edge_index) != 0:
            padded_edge_index[:edge_index.shape[0], :] = edge_index
        
        padded_edge_attr = np.zeros((GRAPH.MAX_EDGES.value, 1))    
        padded_edge_attr[:edge_attr.shape[0], :] = edge_attr
        
        obs = {
            'node': padded_x.astype(np.float32),
            'edge_index': padded_edge_index.astype(np.int64),
            'edge_attr': padded_edge_attr.astype(np.float32)}
        return obs
    
    def _calculate_node_features(self,
                                simplified_assembly: Assembly,
                                original_assembly: Assembly) -> np.ndarray:
        features = []
        for idx, part in enumerate(simplified_assembly):
            if part.n_faces() == 0:
                features.append([0] * GRAPH.NODE_DIM.value)   
                continue
            volume = part.volume()
            surface_area = part.area()
            polygon_count = part.n_faces() / original_assembly[idx].n_faces()
            features.append([volume, surface_area, polygon_count])
        
        features = np.array(features)
        features = (features - np.min(features, axis=0)) / (np.ptp(features, axis=0) + 1e-6)
        return features


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