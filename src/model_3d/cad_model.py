import os

from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound, TopoDS_Iterator
from OCC.Core.Bnd import Bnd_Box    
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse   
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.TopAbs import TopAbs_OUT
import pyvista as pv
import torch
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.ops import sample_points_from_meshes

from src.model_3d.tessellator.brep_convertor import ShapeToMeshConvertor
from src.model_3d.file_system import FileReader    

class MetaModel(ABC):
    def __init__(self, part_name: Optional[str] = None):        
        self.part_name = part_name
        self.vista_mesh: Optional[pv.PolyData] = None   
        self.color: str = "red"
        self.tranparency: float = 1.0
        self.is_visible: bool = True    

    @abstractmethod
    def get_volume(self) -> float:
        """
        추상 메서드입니다.
        모델의 부피를 계산하여 반환합니다.
        """
        pass
    
    def __str__(self) -> str:
        return self.part_name

class ViewDocument:
    def __init__(self) -> None:
        self.model_list: list[MetaModel] = []   
        
    def add_model(self, model: MetaModel) -> None:
        """
        ViewDocument에 모델을 추가합니다.
        
        Parameters:
            model (MetaModel): 추가할 모델 객체
        
        Returns:
            None
        """
        self.model_list.append(model)
        return  
    
    def display(self) -> None:
        """
        ViewDocument에 있는 모델들을 화면에 표시합니다.
        
        Returns:
            None
        """
        plotter = pv.Plotter()
        for model in self.model_list:
            if model.vista_mesh is None or model.is_visible is False:
                continue
            if model.vista_mesh.n_faces_strict == 0:
                continue
            plotter.add_mesh(model.vista_mesh, color = model.color, opacity = model.tranparency)    
            
        plotter.show()
        return  

class PartModel(MetaModel):
    def __init__(self, 
                part_name: str = "part model", 
                brep_shape: Optional[TopoDS_Shape] = None, 
                vista_mesh: Optional[pv.PolyData] = None, 
                bnd_box: Optional[Bnd_Box] = None, 
                part_index: Optional[int] = None) -> None:
        super().__init__(part_name)
        self.brep_shape: Optional[TopoDS_Shape] = brep_shape
        self.part_index: Optional[int] = part_index
        self.torch_mesh: Optional[Meshes] = None 
        self.torch_point_cloud: Optional[Pointclouds] = None
        
        if bnd_box is None and isinstance(brep_shape, TopoDS_Shape):  
            self.bnd_box = Bnd_Box()
            brepbndlib.Add(brep_shape, self.bnd_box)
        else:
            self.bnd_box: Optional[Bnd_Box] = bnd_box
        
        if vista_mesh is None and isinstance(brep_shape, TopoDS_Shape):
            self.vista_mesh = ShapeToMeshConvertor.convert_to_pyvista_mesh(brep_shape)
        else:
            self.vista_mesh: Optional[pv.PolyData] = vista_mesh
            
        if isinstance(self.vista_mesh, pv.PolyData) and self.vista_mesh.n_faces_strict != 0:
            self.vista_mesh.clean()
            self.vista_mesh = self.vista_mesh.triangulate()
            points = self.vista_mesh.points
            faces = self.vista_mesh.faces  
            
            torch_points = torch.tensor(points, dtype=torch.float32)
            torch_faces = torch.tensor(faces, dtype=torch.int64)    
            
            torch_points = torch_points.view(1, -1, 3)
            torch_faces = torch_faces.reshape(-1, 4)[:, 1:4]
            torch_faces = torch_faces.view(1, -1, 3)
            
            self.torch_mesh = Meshes(torch_points, torch_faces)  
            self.torch_point_cloud = Pointclouds(torch_points)  
            if not self.torch_mesh.isempty():
                sampled_points = sample_points_from_meshes(self.torch_mesh, 1000)  
                sampled_points = sampled_points.view(1, -1, 3)  
                self.torch_point_cloud = Pointclouds(sampled_points)
        return
            
        
    def simplify(self, simplified_ratio: float) -> None: 
        """
        모델을 단순화합니다.

        Parameters:
            simplified_ratio (float): 모델을 얼마나 단순화할지를 나타내는 비율입니다.

        Return:
            None
        """
        if self.vista_mesh is None:
            return
        if self.vista_mesh.n_faces_strict == 0:
            return
        self.vista_mesh = self.vista_mesh.clean()
        self.vista_mesh = self.vista_mesh.triangulate()
        self.vista_mesh = self.vista_mesh.decimate(simplified_ratio)
        if self.vista_mesh.n_faces_strict != 0:
            self.__init_torch_property()  
        return

    def add_to_view_document(self, view_document: ViewDocument) -> None:   
        """
        PartModel을 ViewDocument에 추가합니다.
        
        Parameters:
            view_document (ViewDocument): 추가할 ViewDocument 객체
        
        Returns:
            None
        """
        view_document.add_model(self)
        return
        
    def copy_from(self, other: 'PartModel') -> None: 
        """
        다른 PartModel에서 속성을 복사합니다.

        Parameters:
            other (PartModel): 복사할 다른 PartModel 객체

        Return:
            None
        """
        if other.brep_shape is None:
            raise ValueError("copy from: other.brep_shape is None")    
        if other.torch_point_cloud is None or other.torch_mesh is None:
            other.__init_torch_property()
        
        self.brep_shape = other.brep_shape
        self.vista_mesh = pv.PolyData()
        self.vista_mesh.deep_copy(other.vista_mesh)
        self.bnd_box = other.bnd_box    
        self.__init_torch_property()
        self.part_index = other.part_index  
        self.color = other.color
        return
    
    def get_volume(self) -> float:
        """
        PartModel의 부피를 계산하여 반환합니다.
        
        Returns:
            float: PartModel의 부피
        """
        if self.vista_mesh is None:
            return 0.0  
        elif self.vista_mesh.n_faces_strict == 0:
            return 0.0  
        return self.vista_mesh.volume
    
    def is_neighbor(self, other: 'PartModel') -> bool:
        """
        주어진 다른 PartModel과 이웃인지 여부를 확인합니다.

        Parameters:
            other (PartModel): 다른 PartModel 객체

        Return:
            bool: 이웃인 경우 True, 그렇지 않은 경우 False
        """
        if self.bnd_box is None or other.bnd_box is None:
            return False
        
        return not self.bnd_box.IsOut(other.bnd_box) == TopAbs_OUT
    
    def __init_torch_property(self) -> None:
        """
        토치 속성을 초기화하는 메서드입니다.

        Parameters:
            없음

        Return:
            없음
        """
        points = self.vista_mesh.points
        faces = self.vista_mesh.faces  
        
        torch_points = torch.tensor(points, dtype=torch.float32)
        torch_faces = torch.tensor(faces, dtype=torch.int64)    
        torch_faces = torch_faces.reshape(-1, 4)[:, 1:4]
        torch_points = torch_points.view(1, -1, 3)
        torch_faces = torch_faces.view(1, -1, 3)
        self.torch_mesh = Meshes(torch_points, torch_faces)  
        if self.torch_mesh.isempty():
            self.torch_point_cloud = Pointclouds(torch_points)  
            return
        sampled_points = sample_points_from_meshes(self.torch_mesh, 1000)  
        sampled_points = sampled_points.view(1, -1, 3)  
        self.torch_point_cloud = Pointclouds(sampled_points)
        return
    
    def is_open(self) -> bool:
        """
        해당 3D 모델이 열려 있는지 여부를 확인합니다.

        Return:
            - bool: 3D 모델이 열려 있으면 True, 그렇지 않으면 False를 반환합니다.
        """
        edges = self.vista_mesh.extract_feature_edges(boundary_edges=True,
                                                        feature_edges=False,
                                                        manifold_edges=False,
                                                        non_manifold_edges=False)
        if edges.n_points == 0:
            return False    
        return True
    
    
class Assembly(MetaModel):
    def __init__(self, assemply_name: Optional[str] = None) -> None:
        super().__init__(assemply_name)
        self.is_visible = False 

        self.part_model_list: Optional[List[PartModel]] = None
        self.conectivity_dict: Optional[Dict[int, List[int]]] = None 
        return
    
    def get_face_number(self) -> int:
        """
        Return:
            정수형으로 표현된 모델의 총 면의 수를 반환합니다.
        """
        if self.part_model_list is None:
            return 0
        
        sum_of_face_number: int = 0
        for part in self.part_model_list:
            if isinstance(part.vista_mesh, pv.PolyData):
                sum_of_face_number += part.vista_mesh.n_faces_strict

        return sum_of_face_number
    
    def get_volume(self) -> float:
        """
        3D CAD 모델의 총 부피를 계산하여 반환합니다.

        Returns:
            float: 3D CAD 모델의 총 부피
        """
        if self.part_model_list is None:
            return 0.0
        
        sum_of_volume: float = 0.0 
        for part in self.part_model_list:
            sum_of_volume += part.get_volume()
            
        return sum_of_volume
    
    def add_to_view_document(self, view_document: ViewDocument) -> None:
        """
        Assembly를 ViewDocument에 추가합니다.
        
        Parameters:
            view_document (ViewDocument): 추가할 ViewDocument 객체
        
        Returns:
            None
        """
        for part in self.part_model_list:
            view_document.add_model(part)
        return  

    def copy_from_assembly(self, other: 'Assembly') -> None:
        """
        다른 어셈블리에서 모델을 복사합니다.

        Parameters:
            other ('Assembly'): 복사할 어셈블리 객체

        Return:
            None
        """
        self.part_model_list = []   
        for other_part_model in other.part_model_list:
            part_model = PartModel()
            part_model.copy_from(other_part_model)    
            self.part_model_list.append(part_model)
            
        self.conectivity_dict = other.conectivity_dict
        self.part_name = other.part_name
        return

class AssemblyFactory:
    @classmethod
    def create_assembly(cls, stp_file_path: str) -> Assembly:
        """
        STP 파일로부터 Assembly 객체를 생성합니다.
        
        Parameters:
            stp_file_path (str): STP 파일 경로
        
        Returns:
            Assembly: 생성된 Assembly 객체
        """
        assembly = Assembly()   
        
        stp_file_path = os.path.basename(stp_file_path)
        assembly_name, _ = os.path.splitext(stp_file_path)
        assembly.part_name = assembly_name 
                
        part_model_list: list[PartModel] = []
        
        brep_compound: TopoDS_Compound = FileReader.read_stp_file(stp_file_path)
        shape_iter: TopoDS_Iterator = TopoDS_Iterator(brep_compound)
        
        part_index: int = 0
        while (shape_iter.More()):
            brep_shape: TopoDS_Shape = shape_iter.Value()
            if brep_shape.IsNull():
                continue

            part_model_list.append(cls.create_part_model(brep_shape = brep_shape, 
                                                        part_index = part_index))
            shape_iter.Next()
            part_index += 1
            
        assembly.part_model_list = part_model_list
        
        connectivity_dict: dict[int, list[int]] = \
            cls.create_part_connectivity_dict(part_model_list)
        assembly.conectivity_dict = connectivity_dict   
        
        return assembly
    
    @classmethod
    def create_part_model(cls, brep_shape: TopoDS_Shape, part_index: int) -> PartModel:
        """
        BRep 모델로부터 PartModel 객체를 생성합니다.
        
        Parameters:
            brep_shape (TopoDS_Shape): BRep 모델
            part_index (int): PartModel의 인덱스
        
        Returns:
            PartModel: 생성된 PartModel 객체
        """
        part_model = PartModel(part_index = part_index)
        part_model.copy_from(brep_shape)
        return part_model
    
    @classmethod
    def create_part_connectivity_dict(cls, part_model_list: List[PartModel]) -> Dict[int, List[int]]:
        """
        PartModel들 간의 연결성을 나타내는 딕셔너리를 생성합니다.
        
        Parameters:
            part_model_list (List[PartModel]): PartModel 리스트
        
        Returns:
            Dict[int, List[int]]: PartModel들 간의 연결성을 나타내는 딕셔너리
        """
        connectivity_dict: Dict[int, List[int]] = {}
        for i, part_model in enumerate(part_model_list):
            connectivity_dict[i] = []
            for j, other_part_model in enumerate(part_model_list):
                if i != j and part_model.is_neighbor(other_part_model):
                    connectivity_dict[i].append(j)
        return connectivity_dict
    
    @classmethod
    def create_merged_assembly(cls, 
                            assembly: Assembly, 
                            cluster_list: List[List[int]], 
                            assembly_name: str) -> Assembly:
        """
        연결성을 표현하는 dict을 통해 assembly를 병합하고 반환
        
        Parameters:
            part_model_list (List[PartModel]): PartModel 리스트
        
        Returns:
            merged_assembly (Assembly): 병합된 Assembly 객체
        """
        if cluster_list is None:   
            raise ValueError("cluster_list is None")    
        if cluster_list == [] or cluster_list == [[]]:
            raise ValueError("cluster_list is empty")   
        if assembly.part_model_list is None:    
            raise ValueError("assembly.part_model_list is None")    
        
        merged_assembly = Assembly()   
        merged_assembly.part_name = assembly_name
        
        # merge part model  
        part_model_list: list[PartModel] = []
        for cluster_index, cluster in enumerate(cluster_list):
            if len(cluster) == 1:
                part_model: PartModel = PartModel()
                part_model.copy_from(assembly.part_model_list[cluster[0]])  
                part_model_list.append(part_model)
                continue
            
            merged_part_model = cls.merge_part_model(assembly, cluster, cluster_index)
            part_model_list.append(merged_part_model)  
            
        merged_assembly.part_model_list = part_model_list 
        # reindxing
        for part_index, part_model in enumerate(part_model_list):
            part_model.part_index = part_index
        
        # create connectivity dict
        connectivity_dict: dict[int, list[int]] = \
            cls.create_part_connectivity_dict(part_model_list)
        merged_assembly.conectivity_dict = connectivity_dict   

        return merged_assembly

    @classmethod
    def merge_part_model(cls, assembly: Assembly, 
                        cluster: List[int], 
                        cluster_index: int) -> PartModel:        
        """
        연결성을 표현하는 list을 통해 assembly의 part를 병합하고 반환
        
        Parameters:
            assembly (Assembly): 파트가 포함된 Assembly 객체
            cluster: List[int]: 연결성을 표현하는 list 
            cluster_index: int: 병합을 진행하는 cluster의 인덱스
        
        Returns:
            merged_part (PartModel): 병합된 PartModel 객체
        """
        merged_part_model = PartModel()
        merged_part_model.part_name = "merged_part"
        color: str = assembly.part_model_list[cluster[0]].color
        fused_brep_shape: TopoDS_Shape = TopoDS_Shape() 
        fused_vista_mesh: pv.PolyData = pv.PolyData()
        
        for part_index in cluster:
            part_model: PartModel = assembly.part_model_list[part_index]
            
            if fused_brep_shape.IsNull():
                fused_brep_shape = part_model.brep_shape
                if part_model.vista_mesh is not None and isinstance(part_model.vista_mesh, pv.PolyData):  
                    fused_vista_mesh = part_model.vista_mesh
            else:
                fused_brep_shape = BRepAlgoAPI_Fuse(fused_brep_shape, part_model.brep_shape).Shape()
                
                if part_model.vista_mesh is not None and isinstance(part_model.vista_mesh, pv.PolyData):
                    try:
                        fused_vista_mesh += part_model.vista_mesh
                    except AttributeError:
                        continue

        return cls.create_part_model(brep_shape = fused_brep_shape,
                                    vista_mesh = fused_vista_mesh, 
                                    part_name = "merged_part", 
                                    part_index = cluster_index,
                                    color = color)
    
    @classmethod
    def create_part_model(cls, brep_shape: TopoDS_Shape, 
                        vista_mesh: Optional[pv.PolyData] = None, 
                        part_name: str = "part", 
                        part_index: Optional[int] = None, 
                        color: str = "red") -> PartModel: 
            """
            클래스 메서드로서 파트 모델을 생성합니다.

            Parameters:
                brep_shape (TopoDS_Shape): BREP 형태의 모델
                vista_mesh (Optional[pv.PolyData]): PyVista의 PolyData 형태의 메쉬 (기본값: None)
                part_name (str): 파트의 이름 (기본값: "part")
                part_index (Optional[int]): 파트의 인덱스 (기본값: None)
                color (str): 파트의 색상 (기본값: "red")

            Returns:
                PartModel: 생성된 파트 모델 객체
            """
            bnd_box = Bnd_Box()
            brepbndlib.Add(brep_shape, bnd_box) 
            
            if vista_mesh is None:
                mesh: pv.PolyData = ShapeToMeshConvertor.convert_to_pyvista_mesh(brep_shape)
            else:
                mesh: pv.PolyData = pv.PolyData(vista_mesh.points, vista_mesh.faces)

            part_model = PartModel(part_name = part_name,
                            vista_mesh = mesh, 
                            brep_shape = brep_shape,
                            bnd_box = bnd_box,
                            part_index = part_index)
            part_model.color = color
            return part_model
        
    @classmethod
    def create_part_connectivity_dict(cls, part_model_list: List[PartModel]) -> Dict[int, int]:
        """
        주어진 PartModel 리스트를 기반으로 부품 연결성 딕셔너리를 생성합니다.

        Parameters:
            part_model_list (List[PartModel]): 부품 모델 리스트

        Return:
            Dict[int, int]: 부품 인덱스를 키로, 연결된 이웃 부품 인덱스의 리스트를 값으로 갖는 딕셔너리
        """
        conectivity_dict: dict[int, list[int]] = {}
        
        for part_model in part_model_list:   
            for part_model_neighbor in part_model_list:
                if part_model == part_model_neighbor:
                    continue
                
                if part_model.is_neighbor(part_model_neighbor):
                    part_index = part_model.part_index
                    neighbor_index = part_model_neighbor.part_index
                    
                    if part_index not in conectivity_dict:
                        conectivity_dict[part_index] = []

                    conectivity_dict[part_index].append(neighbor_index)
                    
        return conectivity_dict