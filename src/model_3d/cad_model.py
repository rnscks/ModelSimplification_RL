import os
from abc import ABC, abstractmethod
from typing import Optional 
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound, TopoDS_Iterator
from OCC.Core.Bnd import Bnd_Box    
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.TopAbs import TopAbs_OUT
import pyvista as pv

from file_system import FileReader    
from tessellator.brep_convertor import ShapeToMeshConvertor


class MetaModel(ABC):
    def __init__(self, part_name: Optional[str] = None):        
        self.part_name = part_name
        self.brep_shape: Optional[TopoDS_Shape] = None  
        self.mesh: Optional[pv.PolyData] = None 
        self.color: str = "red"
        self.tranparency: float = 0.0
        self.is_visible: bool = True    


    @abstractmethod
    def get_volume(self) -> float:
        pass
    
    def __str__(self) -> str:
        return self.part_name

class ViewDocument:
    def __init__(self) -> None:
        self.model_list: list[MetaModel] = []   
        
    
    def add_model(self, model: MetaModel) -> None:
        self.model_list.append(model)
        return  
    
    def display(self) -> None:
        plotter = pv.Plotter()
        for model in self.model_list:
            plotter.add_mesh(model.mesh)    
            
        plotter.show()
        return  

class PartModel(MetaModel):
    def __init__(self, part_name: str = "part model", brep_shape: Optional[TopoDS_Shape] = None, mesh: Optional[pv.PolyData] = None, bnd_box: Optional[Bnd_Box] = None) -> None:   
        super().__init__(part_name)
        self.brep_shape = brep_shape
        self.mesh = mesh
        self.bnd_box: Optional[Bnd_Box] = bnd_box
        
        
    def add_to_view_document(self, view_document: ViewDocument) -> None:   
        view_document.add_model(self)
        return
        
    def copy_from(self, other: 'PartModel') -> None:    
        self.brep_shape = other.brep_shape
        self.mesh = other.mesh
        return
    
    def get_volume(self) -> float:
        return self.mesh.volume
    
    def is_neighbor(self, other: 'PartModel') -> bool:  
        if self.bnd_box is None or other.bnd_box is None:
            return False
        
        return self.bnd_box.IsOut(other.bnd_box) == TopAbs_OUT
    
    def __eq__(self, other: 'PartModel') -> bool:
        return self.brep_shape.IsEqual(other.brep_shape)       

class Assembly(MetaModel):
    def __init__(self, assemply_name: Optional[str] = None) -> None:
        super().__init__(assemply_name)
        self.part_model_list: Optional[list[PartModel]] = None
        self.conectivity_dict: Optional[dict[int, int]] = None 
        return
    
    def get_volume(self) -> float:  
        if self.part_model_list is None:
            return 0.0
        
        sum_of_volume: float = 0.0 
        for part in self.part_model_list:
            sum_of_volume += part.get_volume()
            
        return sum_of_volume
    
    def add_to_view_document(self, view_document: ViewDocument) -> None:
        for part in self.part_model_list:
            view_document.add_model(part)
        return  

class AssemblyFactory:

    @classmethod
    def create_assembly(cls, stp_file_path: str) -> Assembly:
        assembly = Assembly()   
        
        stp_file_path = os.path.basename(stp_file_path)
        assembly_name, _ = os.path.splitext(stp_file_path)
        assembly.part_name = assembly_name 
                
        part_model_list: list[PartModel] = []
        
        brep_compound: TopoDS_Compound = FileReader.read_stp_file(stp_file_path)
        shape_iter: TopoDS_Iterator = TopoDS_Iterator(brep_compound)
    
        while (shape_iter.More()):
            brep_shape: TopoDS_Shape = shape_iter.Value()
            if brep_shape.IsNull():
                continue

            mesh: pv.PolyData = ShapeToMeshConvertor.convert_to_pyvista_mesh(brep_shape)
            part_model_list.append(cls.create_part_model(mesh, brep_shape))
            shape_iter.Next()
            
        assembly.part_model_list = part_model_list
        
        connectivity_dict: dict[int, int] = cls.create_part_connectivity_dict(part_model_list)
        assembly.conectivity_dict = connectivity_dict   
        
        return assembly
    
    @classmethod
    def create_part_model(cls, mesh: pv.PolyData, brep_shape: TopoDS_Shape, part_name: str = "part") -> PartModel: 
        bnd_box = Bnd_Box()
        brepbndlib.Add(brep_shape, bnd_box) 
        
        return PartModel(part_name= part_name,
                        mesh = mesh, 
                        brep_shape = brep_shape,
                        bnd_box = bnd_box)
        
    @classmethod
    def create_part_connectivity_dict(cls, part_model_list: list[PartModel]) -> dict[int, int]:
        conectivity_dict: dict[int, int] = {}
        for part_index, part_model in enumerate(part_model_list):   
            for neighbor_index, part_model_neighbor in enumerate(part_model_list):
                if part_model == part_model_neighbor:
                    continue
                
                if part_model.is_neighbor(part_model_neighbor):
                    conectivity_dict[part_index] = neighbor_index
        return conectivity_dict


if __name__ == "__main__":
    assembly: Assembly = AssemblyFactory.create_assembly("AirCompressor.stp")
    view_document = ViewDocument()
    
    assembly.add_to_view_document(view_document)
    view_document.display()