from typing import Tuple, List, Set, Dict
import numpy as np
from itertools import product
from enum import Enum

from src.mesh.model import Assembly, PartModel
from src.mesh.metrics import ConcatArea, METRIC


class GRAPH(Enum):   
    NODE_DIM = 3
    MAX_NODES = 20
    MAX_EDGES = 400
    EPSILON = 1e-4


class Preprocess:
    def __init__(self):
        pass
    
    def graph_process(self, assembly:Assembly) -> np.ndarray:
        matrix = np.zeros((len(assembly), len(assembly)))
        for i, j in product(range(len(assembly)), range(len(assembly))):
            if i == j: 
                continue
            matrix[i][j] = self._cal_edge_weight(assembly[i], assembly[j])
            matrix[j][i] = matrix[i][j]
            
        max_val = np.max(matrix[np.isfinite(matrix)]) + GRAPH.EPSILON.value
        matrix[np.isinf(matrix)] = max_val
        min_val = np.min(matrix)
        
        denom = max_val - min_val + 1e-8  # 0-division 방지용 epsilon
        matrix = (matrix - min_val) / denom
        return matrix
    
    def node_process(self, assembly:Assembly) -> np.ndarray:
        node_arr = []
        for part in assembly:
            node_arr.append(self._cal_node_feature(part))
        
        node_arr = np.array(node_arr, dtype=np.float32)
        node_arr = (node_arr - np.min(node_arr, axis=0)) / (np.ptp(node_arr, axis=0) + 1e-6)
        return node_arr

    def _cal_edge_weight(self, part:PartModel, other:PartModel):
        concat_area = ConcatArea().evaluate(part, other)  
        if concat_area == METRIC.BROKEN or concat_area < 1e-4:
            concat_area = np.inf
        return concat_area

    def _cal_node_feature(self, part: PartModel) -> List:
        surface_area: float = part.area()
        volume: float = part.volume()
        n_face: int = part.n_faces()
        return [surface_area, volume, n_face]


if __name__ == "__main__":
    assembly = Assembly.load('src/data/assembly_models/set3/13_assembly47')
    preprocess = Preprocess()
    matrix = preprocess.graph_process(assembly)
    node_arr = preprocess.node_process(assembly)    
    print(matrix)
    print(node_arr)