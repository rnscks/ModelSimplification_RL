from abc import ABC, abstractmethod
import numpy as np
from pytorch3d import loss
from pytorch3d.structures import Pointclouds
from scipy.spatial import KDTree
from enum import Enum
import pyvista as pv
import numpy as np
from typing import List, Tuple, Dict
from src.mesh.model import Entity

class METRIC(Enum):
    BROKEN = 1e4

class Metric(ABC):
    def __init__(self) -> None:
        super().__init__()  
    
    @abstractmethod
    def evaluate(self, model: Entity, other_model: Entity) -> float:
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
    
class VisualLoss(Metric):
    def __init__(self, window_size: Tuple[int, int] = (400, 400)):  
        self.plotter = pv.Plotter(off_screen=True, window_size=window_size) 
        self.positions: List[Tuple] = [
            (2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0), (0, 0, 2), (0, 0, -2),
            (2, 2, 2), (-2, 2, 2), (2, -2, 2), (2, 2, -2), (-2, -2, 2), (-2, 2, -2), (2, -2, -2), (-2, -2, -2)
        ]
        self.viewup: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {
            (2, 0, 0) : (0, 1, 0),
            (-2, 0, 0) : (0, 1, 0), 
            (0, 2, 0) : (0, 0, 1),  
            (0, -2, 0) : (0, 0, -1), 
            (0, 0, 2) : (0, 1, 0),
            (0, 0, -2) : (0, 1, 0), 
            (2, 2, 2) : (0, 1, 0),  
            (-2, 2, 2) : (0, 1, 0), 
            (2, -2, 2) : (0, 1, 0), 
            (-2, -2, 2) : (0, 1, 0), 
            (2, 2, -2) : (0, 1, 0),
            (-2, 2, -2) : (0, 1, 0), 
            (2, -2, -2) : (0, 1, 0), 
            (-2, -2, -2) : (0, 1,  0)}
        self.plotter.camera.zoom(1.0)   
        # self.plotter.enable_parallel_projection()   
        # self.plotter.parallel_scale = 1.0
        
        
    def evaluate(self, model: Entity, other_model: Entity) -> float:   
        depth_maps = []
        min_depth = -5.0
        max_depth = 0.0
        mesh: pv.PolyData = model.mesh
        other_mesh: pv.PolyData = other_model.mesh
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        center = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
        
        for position in self.positions:
            # 카메라의 위치, 초점, ViewUp 방향을 설정
            self.plotter.set_position((position[0] + center[0], position[1] + center[1], position[2] + center[2]))
            self.plotter.set_focus(center)
            self.plotter.set_viewup(self.viewup[position])
            
            # 첫번째 메쉬에 대한 깊이맵을 생성 후 초기화
            self.plotter.add_mesh(mesh)
            self.plotter.show(auto_close=False) 
            depth_map: np.ndarray = self.plotter.get_image_depth(fill_value=min_depth)
            self.plotter.clear()
            
            # 두번째 메쉬에 대한 깊이맵을 생성 후 초기화
            self.plotter.add_mesh(other_mesh)  
            self.plotter.show(auto_close=False) 
            other_depth_map: np.ndarray = self.plotter.get_image_depth(fill_value=min_depth)    
            self.plotter.clear()
            
            if depth_map.max() > max_depth or other_depth_map.max() > max_depth:    
                raise ValueError("Depth map is out of range")   
            if depth_map.min() < min_depth or other_depth_map.min() < min_depth:
                raise ValueError("Depth map is out of range")   
            
            # 깊이맵을 정규화
            depth_map = (depth_map - min_depth) / (max_depth - min_depth)
            other_depth_map = (other_depth_map - min_depth) / (max_depth - min_depth)
            
            # 깊이맵의 차이를 계산(MSE의 Error에 해당하는 값)
            error_depth_map = depth_map - other_depth_map
            depth_maps.append(error_depth_map)
        depth_maps = np.array(depth_maps)   
        return np.mean((depth_maps) ** 2)