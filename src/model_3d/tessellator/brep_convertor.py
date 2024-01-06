from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool_Triangulation
from OCC.Core.BRepTools import breptools_Clean
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Iterator, TopoDS_Shape
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.Graphic3d import Graphic3d_Vec3d
from OCC.Core.Prs3d import prs3d
from OCC.Core.Precision import precision_Confusion
from OCC.Core.gp import gp_Pnt
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
import pyvista as pv    
import open3d as o3d


class ShapeToMeshConvertor:
    
    @classmethod    
    def convert_to_pyvista_mesh(cls, brep_shape: TopoDS_Shape) -> pv.PolyData:
        cls.init_brep_mesh(brep_shape)
        explorer = TopExp_Explorer()
        explorer.Init(brep_shape, TopAbs_FACE)
        
        mesh_faces = []
        mesh_vertices = []

        while (explorer.More()):
            face = explorer.Current()
            cls.init_triangle_mesh(mesh_vertices, mesh_faces, face)
            explorer.Next()
            
        return pv.PolyData(mesh_vertices, mesh_faces)  
    
    @classmethod    
    def init_brep_mesh(cls, shape: TopoDS_Shape) -> BRepMesh_IncrementalMesh:    
        def calculate_angle_deflection() -> float:
            # init basic value
            angle_deflection_max, angle_deflection_min = 0.8, 0.2
            quality = 5.0

            # calculate angle deflection
            angle_deflection_gap = (angle_deflection_max - angle_deflection_min) / 10
            angle_deflection = \
                max(angle_deflection_max - (quality * angle_deflection_gap), angle_deflection_min)
            return angle_deflection

        def calculate_line_deflection(bnd_box: Bnd_Box) -> float:
            # calculate linear deflection
            gvec1 = Graphic3d_Vec3d(*bnd_box.CornerMin().Coord())
            gvec2 = Graphic3d_Vec3d(*bnd_box.CornerMax().Coord())
            deflection = prs3d.GetDeflection(gvec1, gvec2, 0.001)

            line_deflaction = max(deflection, precision_Confusion())

            return line_deflaction
        bnd_box = Bnd_Box()
        brepbndlib.Add(shape, bnd_box)
        
        angle_deflection = calculate_angle_deflection()
        line_deflaction = calculate_line_deflection(bnd_box)
        breptools_Clean(shape)
        bmesh = BRepMesh_IncrementalMesh(
            shape, line_deflaction, False, angle_deflection, False)
        return bmesh

    @classmethod    
    def init_triangle_mesh(cls, mesh_vertices, mesh_faces, face) -> None:
        before_vertices_number = len(mesh_vertices)
        loc = TopLoc_Location()

        poly = BRep_Tool_Triangulation(face, loc)
        if (poly is None):
            return
        
        node_numbers = poly.NbNodes()
        nodes = poly.InternalNodes()

        for node_number in range(node_numbers):
            pnt = nodes.Value(node_number).Transformed(loc.Transformation())
            mesh_vertices.append(pnt.Coord())

        triangles = poly.InternalTriangles()
        mesh_triangle_indicies = \
            list(range(before_vertices_number, len(mesh_vertices) + before_vertices_number))
            
        for triangle_number in range(triangles.Lower(), triangles.Upper() + 1):
            triangle_indicies = triangles.Value(triangle_number).Get()
            if len(triangle_indicies) == 0:
                continue
            
            triangle_index = \
                [mesh_triangle_indicies[index - 1] for index in triangle_indicies]
            mesh_faces.append((3, *triangle_index))
        return


if __name__ == "__main__":
    box_shape = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), gp_Pnt(1, 1, 1)).Shape()
    mesh = ShapeToMeshConvertor.convert_to_pyvista_mesh(box_shape)
    
    mesh.plot()

