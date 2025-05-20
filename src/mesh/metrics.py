from abc import ABC, abstractmethod
import numpy as np
from pytorch3d import loss
from pytorch3d.structures import Pointclouds
from scipy.spatial import KDTree
from enum import Enum

from src.mesh.model import Entity

class METRIC(Enum):
    BROKEN = 1e4

class Metric(ABC):
    def __init__(self) -> None:
        super().__init__()  
    
    @abstractmethod
    def evaluate(self, model: Entity) -> float:
        pass

class ChamferDistance(Metric):
    def __init__(self) -> None:
        super().__init__()  
        
    def evaluate(self, 
                model: Entity, 
                other_model: Entity) -> float:
        p1: Pointclouds = model.torch_point_cloud()
        p2: Pointclouds = other_model.torch_point_cloud()
        
        if isinstance(p1, Pointclouds) == False or isinstance(p2, Pointclouds) == False:    
            return METRIC.BROKEN
        
        chamfer_distance_loss, loss_normal = loss.chamfer_distance(p1, p2, 
                                                        point_reduction="mean", 
                                                        single_directional=False)
        return chamfer_distance_loss.item()

class ConcatArea(Metric):
    def __init__(self) -> None:
        super().__init__()
        
    def evaluate(self, 
                model: Entity, 
                other_model: Entity,
                threshold: float = 1e-2) -> float:
        pnt_cloud1 = model.torch_point_cloud()
        pnt_cloud2 = other_model.torch_point_cloud()
        if isinstance(pnt_cloud1, Pointclouds) == False or isinstance(pnt_cloud2, Pointclouds) == False:    
            return METRIC.BROKEN
        pnt_cloud1 = pnt_cloud1.points_packed().detach().cpu().numpy()
        pnt_cloud2 = pnt_cloud2.points_packed().detach().cpu().numpy()
        close_points = self._get_close_points(pnt_cloud1, pnt_cloud2, threshold)   
        
        return self._get_concat_area_weight(close_points)  
    
    def _get_close_points(self, pnt_cloud1, pnt_cloud2, threshold=1e-2):
        tree = KDTree(pnt_cloud2)
        close_points_mask1 = tree.query_ball_point(pnt_cloud1, r=threshold)
        close_points1 = np.array([pnt_cloud1[i] for i in range(len(pnt_cloud1)) if close_points_mask1[i]])
        
        tree = KDTree(pnt_cloud1)
        close_points_mask2 = tree.query_ball_point(pnt_cloud2, r=threshold)
        close_points2 = np.array([pnt_cloud2[i] for i in range(len(pnt_cloud2)) if close_points_mask2[i]])
        close_points = np.vstack((close_points1, close_points2))
        
        return close_points

    def _get_concat_area_weight(self, points) -> float:
        bounding_box_min = np.min(points, axis=0)
        bounding_box_max = np.max(points, axis=0)
        extents = bounding_box_max - bounding_box_min
        return np.sum(extents)