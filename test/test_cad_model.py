import unittest

from pytorch3d.structures import Meshes, Pointclouds
import pyvista as pv
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib  
from OCC.Core.gp import gp_Pnt
from OCC.Core.TopoDS import TopoDS_Shape

from src.model_3d.tessellator.brep_convertor import ShapeToMeshConvertor   
from src.model_3d.cad_model import PartModel, Assembly, AssemblyFactory
from src.model_3d.model_util import RegionGrowing


class PartModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.brep_shape: TopoDS_Shape = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), gp_Pnt(1, 1, 1)).Shape()
        self.vista_mesh: pv.PolyData = ShapeToMeshConvertor.convert_to_pyvista_mesh(self.brep_shape)    
        self.bnd_box: Bnd_Box = Bnd_Box()
        brepbndlib.Add(self.brep_shape, self.bnd_box)   
        self.part_index: int = 0

        self.part_model = PartModel(
            part_name="part model for unit test",
            brep_shape=self.brep_shape,
            vista_mesh=self.vista_mesh,
            bnd_box=self.bnd_box,
            part_index=self.part_index
        )
        
    def test_meta_model_init(self) -> None: 
        self.assertEqual(self.part_model.part_name, "part model for unit test")
        self.assertEqual(self.part_model.color, "red")
        self.assertEqual(self.part_model.tranparency, 1.0) 
        self.assertEqual(self.part_model.is_visible, True)
        return  

    def test_simplify(self) -> None:
        simplified_ratio = 0.1

        self.part_model.simplify(simplified_ratio)
        
        self.assertIsNotNone(self.part_model.vista_mesh)
        self.assertIsNotNone(self.part_model.torch_mesh)   
        self.assertIsNotNone(self.part_model.torch_point_cloud) 
        
        self.assertGreater(self.part_model.vista_mesh.n_faces_strict, 0)   
        self.assertGreater(self.part_model.vista_mesh.n_points, 0)  
        self.assertGreater(self.part_model.torch_mesh.num_faces_per_mesh(), 0)   
        self.assertGreater(self.part_model.torch_mesh.num_verts_per_mesh(), 0)  
        return   
        
    def test_init_torch_property(self) -> None:
        self.assertIsNotNone(self.part_model.torch_mesh)
        self.assertIsNotNone(self.part_model.torch_point_cloud)
        
        self.assertIsInstance(self.part_model.torch_mesh, Meshes)
        self.assertIsInstance(self.part_model.torch_point_cloud, Pointclouds) 
        return

    def test_copy_from(self) -> None:
        other_part_model = PartModel()
        other_part_model.copy_from(self.part_model)
                
        self.assertTrue(other_part_model.brep_shape.IsEqual(self.part_model.brep_shape))    
        self.assertTrue(other_part_model.vista_mesh.n_faces_strict == self.part_model.vista_mesh.n_faces_strict)    
        self.assertTrue(other_part_model.vista_mesh.n_points == self.part_model.vista_mesh.n_points)    
        self.assertTrue(other_part_model.part_index == self.part_model.part_index)  
        self.assertTrue(other_part_model.torch_mesh.num_faces_per_mesh() == self.part_model.torch_mesh.num_faces_per_mesh())        
        self.assertTrue(other_part_model.torch_mesh.num_verts_per_mesh() == self.part_model.torch_mesh.num_verts_per_mesh())    
        return

    def test_get_volume(self) -> None:
        volume = self.part_model.get_volume()
        self.assertIsInstance(volume, float)
        return

    def test_is_neighbor_out_case(self) -> None:
        # description: 경계 밖에 있는 경우
        other_brep_shape: TopoDS_Shape =BRepPrimAPI_MakeBox(
            gp_Pnt(2, 2, 2), 
            gp_Pnt(3, 3, 3)).Shape()    
        
        other_vista_mesh: pv.PolyData = ShapeToMeshConvertor.convert_to_pyvista_mesh(other_brep_shape)  
        other_bnd_box = Bnd_Box()  
        brepbndlib.Add(other_brep_shape, other_bnd_box)
        other_part_index: int = 1
        
        other_part_model = PartModel(
            part_name="other part model for unit test",
            brep_shape=other_brep_shape,
            vista_mesh=other_vista_mesh,
            bnd_box=other_bnd_box,
            part_index=other_part_index
        )   
        
        
        is_neighbor = self.part_model.is_neighbor(other_part_model)
        self.assertFalse(is_neighbor, bool)
        return
    
    def test_is_neighbor_in_case(self) -> None: 
        # description: 경계 안에 있는 경우
        other_brep_shape: TopoDS_Shape =BRepPrimAPI_MakeBox(
            gp_Pnt(0.5, 0.5, 0.5), 
            gp_Pnt(1.5, 1.5, 1.5)).Shape()    
        
        other_vista_mesh: pv.PolyData = ShapeToMeshConvertor.convert_to_pyvista_mesh(other_brep_shape)  
        other_bnd_box = Bnd_Box()  
        brepbndlib.Add(other_brep_shape, other_bnd_box)
        other_part_index: int = 1
        
        other_part_model = PartModel(
            part_name="other part model for unit test",
            brep_shape=other_brep_shape,
            vista_mesh=other_vista_mesh,
            bnd_box=other_bnd_box,
            part_index=other_part_index
        )   
        
        
        is_neighbor = self.part_model.is_neighbor(other_part_model)
        self.assertTrue(is_neighbor, bool)
        return  

class AssemblyFactoryTests(unittest.TestCase):
    def test_create_assembly(self) -> None:
        assembly = AssemblyFactory.create_assembly("AirCompressor.stp")

        self.assertIsInstance(assembly, Assembly)
        self.assertEqual(len(assembly.part_model_list), 8)
        
        for part_model in assembly.part_model_list:
            self.assertIsInstance(part_model, PartModel)    
            self.assertIsNotNone(part_model.brep_shape)
            self.assertIsNotNone(part_model.vista_mesh) 
            self.assertIsNotNone(part_model.bnd_box)
            self.assertIsNotNone(part_model.part_index)
            self.assertIsNotNone(part_model.torch_mesh)
            self.assertIsNotNone(part_model.torch_point_cloud)
        return  
    
    def test_create_merged_assembly(self) -> None:
        assembly: Assembly = AssemblyFactory.create_assembly("AirCompressor.stp")
        cluster_list: list[list[int]] = RegionGrowing().cluster(assembly, 0)
        merged_assembly: Assembly =\
            AssemblyFactory.create_merged_assembly(
                assembly = assembly, 
                cluster_list=cluster_list, 
                assembly_name="Merged AirCompressor")
            
        for part_model in merged_assembly.part_model_list:
            self.assertIsInstance(part_model, PartModel)    
            self.assertIsNotNone(part_model.brep_shape)
            self.assertIsNotNone(part_model.vista_mesh) 
            self.assertIsNotNone(part_model.bnd_box)
            self.assertIsNotNone(part_model.part_index)
            self.assertIsNotNone(part_model.torch_mesh)
            self.assertIsNotNone(part_model.torch_point_cloud)
        return

class AssemblyTests(unittest.TestCase): 
    def setUp(self) -> None:
        self.test_assembly: Assembly = AssemblyFactory.create_assembly("AirCompressor.stp")
    
    
    def test_part_model(self) -> None:
        for part_model in self.test_assembly.part_model_list:
            self.assertIsInstance(part_model, PartModel)
            self.assertIsNotNone(part_model.brep_shape) 
            self.assertIsNotNone(part_model.vista_mesh)
            self.assertIsNotNone(part_model.bnd_box)
            self.assertIsNotNone(part_model.part_index)
            self.assertIsNotNone(part_model.torch_mesh)
            self.assertIsNotNone(part_model.torch_point_cloud)
        
        return
    
    def test_copy_from(self) -> None:
        copy_assembly = Assembly()
        copy_assembly.copy_from_assembly(self.test_assembly)
        
        for part_model in copy_assembly.part_model_list:
            self.assertIsInstance(part_model, PartModel)
            self.assertIsNotNone(part_model.brep_shape)
            self.assertIsNotNone(part_model.vista_mesh)
            self.assertIsNotNone(part_model.bnd_box)
            self.assertIsNotNone(part_model.part_index)
            self.assertIsNotNone(part_model.torch_mesh)
            self.assertIsNotNone(part_model.torch_point_cloud)

        return
        

if __name__ == '__main__':
    unittest.main()