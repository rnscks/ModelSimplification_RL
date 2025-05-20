import os
from typing import Optional, List
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.ops import sample_points_from_meshes
import pyvista as pv
import torch
from abc import ABC

POINTS_PER_PART = 1280

class Entity(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.mesh: Optional[pv.PolyData] = None 
        self.point_cloud: Optional[Pointclouds] = None
        self.color: str = "skyblue"
        self.transparency: float = 1.0  
    
    
    def display(self) -> None:  
        if self.mesh == None:
            raise ValueError("Mesh is None")
        if self.mesh.n_faces_strict == 0:
            raise ValueError("Mesh is empty")   
        mesh:pv.PolyData = self.mesh.copy() 
        mesh.plot(color=self.color, opacity=self.transparency)   
        return
    
    def torch_mesh(self) -> Meshes:
        torch_mesh = None
        if self.mesh == None:
            return torch_mesh
        if self.mesh.n_faces_strict == 0:
            return torch_mesh
        
        points = self.mesh.points
        faces = self.mesh.faces
        
        torch_points = torch.tensor(points, dtype=torch.float32)
        torch_faces = torch.tensor(faces, dtype=torch.int64)    
        torch_faces = torch_faces.reshape(-1, 4)[:, 1:4]
        torch_points = torch_points.view(1, -1, 3)
        torch_faces = torch_faces.view(1, -1, 3)
        return Meshes(torch_points, torch_faces)  
    
    def torch_point_cloud(self) -> Pointclouds:
        if self.point_cloud != None:
            return self.point_cloud
        torch_mesh = self.torch_mesh()
        if torch_mesh == None:  
            return None 
        sampled_points = sample_points_from_meshes(torch_mesh, POINTS_PER_PART)
        sampled_points = sampled_points.view(1, -1, 3)
        point_cloud = Pointclouds(sampled_points)
        self.point_cloud = point_cloud  
        return point_cloud

class PartModel(Entity):
    def __init__(self, mesh:pv.PolyData=pv.PolyData()) -> None:
        super().__init__()
        self.mesh = mesh
        
        
    def n_faces(self) -> int:
        return self.mesh.n_faces_strict
    
    def volume(self) -> float:
        return self.mesh.volume
    
    def area(self) -> float:
        return self.mesh.area
    
    def merge_with(self, other: 'PartModel') -> None:
        if isinstance(other, PartModel) == False:
            raise ValueError("병합 대상이 PartModel 아닙니다.")  
        if self.mesh == None or other.mesh == None: 
            return
        vista_mesh = self.mesh    
        other_vista_mesh = other.mesh 
        merged_mesh = vista_mesh + other_vista_mesh
        merged_mesh = merged_mesh.clean()   
        merged_mesh = merged_mesh.triangulate() 
        self.mesh = merged_mesh
        self.point_cloud = None
        return
    
    def simplify(self, decimation_ratio: float) -> None: 
        if self.mesh is None:
            return
        if self.mesh.n_faces_strict == 0:
            return
        self.mesh = self.mesh.decimate(decimation_ratio)  
        # self.mesh = self.mesh.clean() 
        # self.mesh = self.mesh.triangulate()   
        self.point_cloud = None
        return

    def copy_from(self, other: 'PartModel') -> None:
        if isinstance(other, PartModel) == False:    
            raise ValueError("카피 대상이 PartModel이 아닙니다.")   
        if other.mesh == None:  
            raise ValueError("카피 대상이 None입니다.") 
        
        self.mesh = other.mesh.copy(deep=True)   
        self.point_cloud = None
        self.color = other.color
        self.transparency = other.transparency  
        return 
    
    def copy(self) -> 'PartModel':
        copied_part = PartModel()   
        copied_part.copy_from(self) 
        return copied_part  
    
    def __hash__(self):
        return super().__hash__()

class Assembly(Entity):
    def __init__(self) -> None:
        super().__init__()
        self.parts: List[PartModel] = []
        return
    
    
    def n_faces(self) -> int:
        if self.parts == []:
            return 0
        
        total_n_faces: int = 0
        for part in self.parts:
            total_n_faces += part.mesh.n_faces_strict
        return total_n_faces
    
    def volume(self) -> float:
        if self.parts == []:
            return 0.0
        
        total_volume: float = 0.0
        for part in self.parts:
            total_volume += part.mesh.volume
        return total_volume
    
    def area(self) -> float:
        if self.parts == []:
            return 0.0
        
        total_area: float = 0.0
        for part in self.parts:
            total_area += part.mesh.area
        return total_area
    
    def merged_mesh(self) -> Optional[pv.PolyData]: 
        if self.parts == []:
            return None
        
        merged_mesh = self.parts[0].mesh.copy()
        for part in self.parts[1:]:
            merged_mesh += part.mesh.copy()
        
        # merged_mesh = merged_mesh.clean()   
        # merged_mesh = merged_mesh.triangulate() 
        return merged_mesh  
    
    def merged_assembly(self) -> 'Assembly':    
        merged_assembly = Assembly()
        merged_assembly.parts.append(PartModel(mesh=self.merged_mesh()))
        merged_assembly.mesh = merged_assembly.merged_mesh()
        return merged_assembly  
    
    def simplify(self, decimation_ratio: float) -> None:    
        for part in self:
            part.simplify(decimation_ratio)
        self.mesh = self.merged_mesh()
        return
    
    def copy_from(self, other: 'Assembly') -> None:
        if isinstance(other, Assembly) == False:    
            raise ValueError("카피 대상이 PartAssembly가 아닙니다.")    
        
        self.parts = []
        for number, part in enumerate(other):
            copied_part = PartModel()
            copied_part.copy_from(part)
            self.parts.append(copied_part)
        self.mesh = self.merged_mesh()  
        return
    
    def copy(self) -> 'Assembly':   
        copied_assembly = Assembly()
        copied_assembly.copy_from(self)
        return copied_assembly  
    
    def index(self, part: PartModel) -> int:    
        for idx, part_model in enumerate(self.parts):
            if part_model == part:
                return idx
        raise ValueError("해당 파트가 어셈블리에 존재하지 않습니다.")
    
    def __getitem__(self, index: int) -> PartModel:
        return self.parts[index]
    
    def __iter__(self):
        for part in self.parts:
            yield part
    
    def __len__(self) -> int:
        return len(self.parts)
    
    def __str__(self) -> str:
        return f"Number of parts: {len(self.parts)}\nFaces: {self.n_faces()}\nVolume: {self.volume()}\nArea: {self.area()}"

    def save(self, assembly_dir: str) -> None:
        mesh_list: List[pv.PolyData] = []
            
        for part in self:
            mesh_list.append(part.mesh)
        
        os.mkdir(assembly_dir)  
        for idx, poly_data in enumerate(mesh_list):
            file_name = f"part{idx}.stl"
            file_path = os.path.join(assembly_dir, file_name)
            poly_data.save(file_path)
        return
    
    @classmethod
    def load(cls, assembly_dir: str) -> 'Assembly':
        assembly = Assembly()   
        
        dir_list = os.listdir(assembly_dir) 
        for file in dir_list:
            if file.endswith('.stl'):
                file_path = os.path.join(assembly_dir, file)   
                part_model = PartModel(mesh=pv.read(file_path))   
                assembly.parts.append(part_model)
        assembly.mesh = assembly.merged_mesh()  
        return assembly
    
if __name__ == "__main__":
    def test_deep_copy_assembly(original_assembly: Assembly):
        print("[TEST] Assembly Deep Copy Validation")
        copied_assembly = original_assembly.copy()
        assert copied_assembly is not original_assembly, "Assembly 객체 자체가 동일"

        for i, (orig_part, copied_part) in enumerate(zip(original_assembly, copied_assembly)):
            assert copied_part is not orig_part, f"Part {i} 객체가 원본과 동일"
            if orig_part.mesh is not None and copied_part.mesh is not None:
                assert copied_part.mesh is not orig_part.mesh, f"Part {i}의 mesh 객체가 동일"
            else:
                print(f"Part {i}는 mesh가 None")
                
        print("복사본에 대해 simplify 수행 중...")
        copied_assembly.simplify(decimation_ratio=0.5)

        for i, (orig_part, copied_part) in enumerate(zip(original_assembly, copied_assembly)):
            if orig_part.mesh and copied_part.mesh:
                n_orig = orig_part.n_faces()
                n_copy = copied_part.n_faces()
                assert n_orig != n_copy, f"Part {i}: simplify 이후에도 face 수가 같음"
            else:
                print(f"Part {i}의 메시가 없음 → face 수 비교 생략")

        print("✅ 복사 동작이 정상적으로 확인")
    
    test_deep_copy_assembly(Assembly.load("data/set4/7_assembly78"))