from itertools import product
import gymnasium as gym
import numpy as np

from src.graph.preprocess import AdjGraph, GRAPH
from src.mesh.metrics import ChamferDistance, METRIC, VisualLoss
from src.mesh.model import Assembly


class Observation:
    def __init__(self) -> None:
        self.node_dim: int = 3
        self.edge_dim: int = 1
        self.max_edges: int = 400
        self.observation_space: gym.spaces = gym.spaces.Dict({
            'node': gym.spaces.Box(low=-1, high=1, shape=(GRAPH.MAX_NODES.value, self.node_dim), dtype=np.float32),
            'edge_index': gym.spaces.Box(low=0, high=self.max_edges, shape=(self.edge_dim, 2), dtype=np.int64),
            'edge_attr': gym.spaces.Box(low=0, high=1, shape=(GRAPH.MAX_EDGES.value, self.edge_dim), dtype=np.float32)
        })
    
    def get_observation(self,
                      simplified_assembly: Assembly,
                      original_assembly: Assembly) -> dict:
        raise NotImplementedError("This method should be overridden by subclasses.")    

class BasicObservation(Observation):
    def __init__(self, 
                 cd_option:bool=False,
                 vl_option:bool=False) -> None:
        super().__init__()
        self.cd_option:bool = cd_option  
        self.vl_option:bool = vl_option
        self.node_dim: int = 3
        if cd_option == True:
            self.node_dim += 1  
        if vl_option == True:   
            self.node_dim += 1  
        
        self.edge_dim: int = 1
        self.max_edges: int = 400
        self.max_nodes: int = 20
        self.observation_space: gym.spaces = gym.spaces.Dict({
            'node': gym.spaces.Box(low=-1, high=1, shape=(self.max_nodes, self.node_dim), dtype=np.float32),
            'edge_index': gym.spaces.Box(low=0, high=self.max_edges, shape=(self.max_edges, 2), dtype=np.int64),
            'edge_attr': gym.spaces.Box(low=0, high=1, shape=(self.max_edges, self.edge_dim), dtype=np.float32)
        })  


    def get_observation(self,
                        simplified_assembly: Assembly,
                        original_assembly: Assembly) -> dict:
        node_features = self._calculate_node_features(simplified_assembly, original_assembly)
        adjacency_matrix = AdjGraph().graph_process(simplified_assembly)  
        
        edge_index = []
        edge_attr = []
        for i, j in product(range(len(adjacency_matrix)), range(len(adjacency_matrix))):   
            if adjacency_matrix[i][j] != 0:
                edge_index.append([i, j])
                edge_attr.append(adjacency_matrix[i][j])
        
        edge_index = np.array(edge_index)
        edge_attr = np.array(edge_attr).reshape(-1, 1)
        
        padded_x = np.zeros((self.max_nodes, self.node_dim))
        padded_x[:node_features.shape[0], :] = node_features
        padded_edge_index = np.zeros((self.max_edges, 2), dtype=np.int64)
        if len(edge_index) != 0:
            padded_edge_index[:edge_index.shape[0], :] = edge_index
        
        padded_edge_attr = np.zeros((self.max_edges, 1))    
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
                features.append([0] * self.node_dim)   
                continue
            feature = []
            if self.cd_option == True:
                cd = ChamferDistance().evaluate(part, original_assembly[idx])
                if cd == METRIC.BROKEN or cd < 1e-4:
                    cd = 0.0
                feature.append(cd)  
            if self.vl_option == True:
                vl = VisualLoss().evaluate(part, original_assembly[idx])
                if vl == METRIC.BROKEN or vl < 1e-4:
                    vl = 0.0
                feature.append(vl)
            
            volume = part.volume()
            surface_area = part.area()
            polygon_count = part.n_faces() / original_assembly[idx].n_faces()
            feature.extend([volume, surface_area, polygon_count])
            features.append(feature)
        
        features = np.array(features)
        features = (features - np.min(features, axis=0)) / (np.ptp(features, axis=0) + 1e-6)
        return features

class PointNetObservation(Observation):
    def __init__(self, 
                 cd_option:bool=False,
                 vl_option:bool=False) -> None:
        super().__init__()
        self.cd_option:bool = cd_option  
        self.vl_option:bool = vl_option
        self.node_dim: int = 3
        if cd_option == True:
            self.node_dim += 1  
        if vl_option == True:   
            self.node_dim += 1  
        
        self.edge_dim: int = 1
        self.max_edges: int = 400
        self.max_nodes: int = 20
        self.observation_space: gym.spaces = gym.spaces.Dict({
            'pointcloud': gym.spaces.Box(low=-1, high=1, shape=(self.max_nodes, 1024, 3), dtype=np.float32),    
            'node': gym.spaces.Box(low=-1, high=1, shape=(self.max_nodes, self.node_dim), dtype=np.float32),
            'edge_index': gym.spaces.Box(low=0, high=self.max_edges, shape=(self.max_edges, 2), dtype=np.int64),
            'edge_attr': gym.spaces.Box(low=0, high=1, shape=(self.max_edges, self.edge_dim), dtype=np.float32)
        })  


    def get_observation(self,
                        simplified_assembly: Assembly,
                        original_assembly: Assembly) -> dict:
        node_features = self._calculate_node_features(simplified_assembly, original_assembly)
        adjacency_matrix = AdjGraph().graph_process(simplified_assembly)  
        
        edge_index = []
        edge_attr = []
        for i, j in product(range(len(adjacency_matrix)), range(len(adjacency_matrix))):   
            if adjacency_matrix[i][j] != 0:
                edge_index.append([i, j])
                edge_attr.append(adjacency_matrix[i][j])
        
        edge_index = np.array(edge_index)
        edge_attr = np.array(edge_attr).reshape(-1, 1)
        
        padded_x = np.zeros((self.max_nodes, self.node_dim))
        padded_x[:node_features.shape[0], :] = node_features
        padded_edge_index = np.zeros((self.max_edges, 2), dtype=np.int64)
        if len(edge_index) != 0:
            padded_edge_index[:edge_index.shape[0], :] = edge_index
        
        padded_edge_attr = np.zeros((self.max_edges, 1))    
        padded_edge_attr[:edge_attr.shape[0], :] = edge_attr
        
        point_cloud = np.zeros((self.max_nodes, 1024, 3))
        for idx, part in enumerate(simplified_assembly):    
            point_cloud[idx] = part.np_point_cloud()
        
        obs = {
            'pointcloud': point_cloud.astype(np.float32),
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
                features.append([0] * self.node_dim)   
                continue
            feature = []
            if self.cd_option == True:
                cd = ChamferDistance().evaluate(part, original_assembly[idx])
                if cd == METRIC.BROKEN or cd < 1e-4:
                    cd = 0.0
                feature.append(cd)  
            if self.vl_option == True:
                vl = VisualLoss().evaluate(part, original_assembly[idx])
                if vl == METRIC.BROKEN or vl < 1e-4:
                    vl = 0.0
                feature.append(vl)
            
            volume = part.volume()
            surface_area = part.area()
            polygon_count = part.n_faces() / original_assembly[idx].n_faces()
            feature.extend([volume, surface_area, polygon_count])
            features.append(feature)
        
        features = np.array(features)
        features = (features - np.min(features, axis=0)) / (np.ptp(features, axis=0) + 1e-6)
        return features
