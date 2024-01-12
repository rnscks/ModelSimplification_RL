import unittest

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

from src.model_3d.cad_model import PartModel, AssemblyFactory, Assembly
from src.model_3d.model_util import (
    ChamferDistance,
    PointToMeshDistance,
    RegionGrowing,
)


class ChamferDistanceTests(unittest.TestCase):
    def test_evaluate_part(self) -> None:
        part_model = PartModel(brep_shape=BRepPrimAPI_MakeBox(1, 1, 1).Shape())    
        other_part_model = PartModel(brep_shape=BRepPrimAPI_MakeBox(1, 1, 1).Shape())  
        
        chamfer_distance = ChamferDistance()
        distance = chamfer_distance.evaluate(part_model, other_part_model) 
        self.assertIsInstance(distance, float)  
        return
        
    def test_evaluate_assembly(self) -> None:
        assembly: Assembly = AssemblyFactory.create_assembly("AirCompressor.stp")
        other_assembly: Assembly = AssemblyFactory.create_assembly("AirCompressor.stp") 
        
        chamfer_distance = ChamferDistance()
        distance = chamfer_distance.evaluate(assembly, other_assembly)
        self.assertIsInstance(distance, float)
        return


class PointToMeshDistanceTests(unittest.TestCase):
    def test_evaluate(self):
        part_model = PartModel(brep_shape=BRepPrimAPI_MakeBox(1, 1, 1).Shape())    
        other_part_model = PartModel(brep_shape=BRepPrimAPI_MakeBox(1, 1, 1).Shape())  
        point_to_mesh_distance = PointToMeshDistance()
        result = point_to_mesh_distance.evaluate(part_model, other_part_model)
        self.assertIsInstance(result, float)


class RegionGrowingTests(unittest.TestCase):
    def test_cluster(self):
        assembly: Assembly = AssemblyFactory.create_assembly("AirCompressor.stp")   
        while True:
            if len(assembly.part_model_list) == 1:
                break
            cluster_list = RegionGrowing().cluster(assembly)    
            self.assertIsInstance(cluster_list, list)
            assembly = AssemblyFactory.create_merged_assembly(assembly, cluster_list, "Merged AirCompressor")
            self.assertIsInstance(assembly, Assembly)
            
        
        return

if __name__ == '__main__':
    unittest.main()