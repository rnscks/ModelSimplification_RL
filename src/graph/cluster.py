from abc import ABC, abstractmethod
from typing import List, Tuple
import markov_clustering as mcl
import numpy as np
from enum import Enum
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

from src.mesh.model import Assembly, PartModel


class Cluster(ABC):
    def __init__(self) -> None:
        super().__init__()  
    
    @abstractmethod
    def cluster(self) -> List[List[int]]:
        pass

class MarkovCluster(Cluster):
    def __init__(self) -> None:
        pass
    
    
    def cluster(self, matrix: np.ndarray, inflation_value: float = 3.0) -> List[int]:
        result = mcl.run_mcl(matrix, inflation=inflation_value)
        clusters = mcl.get_clusters(result)
        clusters = [list(cluster) for cluster in clusters]
        cluster_idx: List[int] = [0 for _ in range(len(matrix))]
        for cluster in clusters:
            for idx in cluster:
                cluster_idx[idx] = clusters.index(cluster)
        return cluster_idx

class HierarchicalCluster(Cluster):
    def __init__(self) -> None:
        pass
    
    def cluster(self, matrix: np.ndarray, tau: float = 0.3) -> List[int]:
        condensed_distance_matrix = squareform(matrix)
        Z = linkage(condensed_distance_matrix, method='single')
        num_clusters: int = int(tau*len(matrix))
        clusters_idx = fcluster(Z, num_clusters, criterion='maxclust')
        clusters_idx = [i - 1 for i in clusters_idx]
        return clusters_idx

if __name__ == "__main__":
    from src.graph.preprocess import AdjGraph
    
    assembly = Assembly.load("src/data/assembly_models/set3/3_assembly41")
    matrix = AdjGraph().graph_process(assembly)
    clusters = HierarchicalCluster().cluster(matrix)
    print(clusters)
    clusters = MarkovCluster().cluster(matrix, inflation_value=5.0)
    print(clusters)
    
    