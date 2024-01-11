import unittest
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Pnt
from OCC.Core.TopoDS import TopoDS_Shape
import numpy as np

from src.model_3d.tessellator.brep_convertor import ShapeToMeshConvertor    

class ShapeToMeshConvertorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.box_shape: TopoDS_Shape = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), gp_Pnt(1, 1, 1)).Shape() 
        self.vista_mesh = ShapeToMeshConvertor.convert_to_pyvista_mesh(self.box_shape)  
        return


    def test_number_of_properties_in_mesh(self) -> None:
        # Test 내용: Vertex, Face를 개수를 이용하여 가장 기본적인 박스 Mesh 테스트
        self.assertTrue(self.vista_mesh.n_points == 24)
        self.assertTrue(self.vista_mesh.n_faces_strict == 12)
        return
    
    def test_check_corner_points(self) -> None:
        corner_points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0]
        ]
        
        mesh_vertices = self.vista_mesh.points.tolist()
        
        for corner_point in corner_points:
            self.assertTrue(corner_point in mesh_vertices)  
        
        for mesh_vertex in mesh_vertices:
            self.assertTrue(mesh_vertex in corner_points)   
        
        return



if __name__ == '__main__':
    unittest.main()