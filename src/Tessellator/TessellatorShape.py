import util

# Cumtom
from Mesh.Mesh import Mesh

# OCC
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool_Triangulation
from OCC.Core.BRepTools import breptools_Clean

from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopLoc import TopLoc_Location

from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.Bnd import Bnd_Box

from OCC.Core.Graphic3d import Graphic3d_Vec3d
from OCC.Core.Prs3d import prs3d_GetDeflection
from OCC.Core.Precision import precision_Confusion
from OCC.Core.gp import gp_Pnt
from OCC.Core.Bnd import Bnd_Box

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere

from OCC.Display.SimpleGui import init_display
from OCC.Display.OCCViewer import Viewer3d
from OCC.Core.Quantity import (
    Quantity_Color,
    Quantity_TOC_RGB,
    Quantity_NOC_WHITE
)

class TessellatorShape:
    def __init__(self, shape: TopoDS_Shape) -> None:
        self.VertexToIntHash: dict[tuple, int] = {}
        self.IntToVertexHash: dict[int, tuple] = {}
        self.Vertices = []
        self.Faces = []
        self.bndBox = Bnd_Box()
        self.Shape = shape
        self.Neighbors = []

        self.__InitBndBox(self.bndBox, shape)
        bRepMesh = self.__InitedBRepMesh(self.bndBox, shape)
        self.__InitMesh(shape)

        self.Mesh_: Mesh = Mesh(self.Vertices, self.Faces)
        self.Mesh_.RandomSampling(100)
        pass

    def __InitMesh(self, shape: TopoDS_Shape):
        explorer = TopExp_Explorer()

        explorer.Init(shape, TopAbs_FACE)

        while (explorer.More()):
            face = explorer.Current()
            self.__InitedMesh(face)
            explorer.Next()

    def __InitBndBox(self, bndBox: Bnd_Box, shape: TopoDS_Shape):
        brepbndlib_Add(shape, bndBox)

    def __InitedAngDeflection(self):
        # init basic value
        angDeflection_max, angDeflection_min = 0.8, 0.2
        quality = 5.0

        # calculate angle deflection
        angDeflection_gap = (angDeflection_max - angDeflection_min) / 10
        angDeflection = max(angDeflection_max - (quality *
                                                 angDeflection_gap), angDeflection_min)
        return angDeflection

    def __InitedLinDeflection(self, bndBox: Bnd_Box):
        # calculate linear deflection
        gvec1 = Graphic3d_Vec3d(bndBox.CornerMin().X(),
                                bndBox.CornerMin().Y(), bndBox.CornerMin().Z())
        gvec2 = Graphic3d_Vec3d(bndBox.CornerMax().X(),
                                bndBox.CornerMax().Y(), bndBox.CornerMax().Z())
        deflection = prs3d_GetDeflection(gvec1, gvec2, 0.001)

        linDeflaction = max(deflection, precision_Confusion())

        return linDeflaction

    def __InitedBRepMesh(self, bndBox: Bnd_Box, shape: TopoDS_Shape):
        angDeflection = self.__InitedAngDeflection()
        linDeflaction = self.__InitedLinDeflection(bndBox)
        breptools_Clean(shape)
        bmesh = BRepMesh_IncrementalMesh(
            shape, linDeflaction, False, angDeflection, False)
        return bmesh

    def __InitedMesh(self, face):
        loc = TopLoc_Location()

        poly = BRep_Tool_Triangulation(face, loc)
        if (poly is None):
            return
        
        nodeNumbers = poly.NbNodes()
        nodes = poly.InternalNodes()
        vertices = []

        for i in range(nodeNumbers):
            pnt = nodes.Value(i).Transformed(loc.Transformation())
            vertices.append(pnt.Coord())
            self.__AddVertex(pnt.Coord())

        triangles = poly.InternalTriangles()
        for i in range(triangles.Lower(), triangles.Upper() + 1):
            index = triangles.Value(i).Get()
            self.__IndexToFace(index, vertices)
        

    def __AddVertexToIntHash(self, vertex: tuple[float, float, float]):
        index = len(self.VertexToIntHash)
        self.VertexToIntHash[vertex] = index
        self.IntToVertexHash[index] = vertex
        pass

    def __AddVertex(self, vertex: tuple[int, int, int]):
        if (not vertex in self.Vertices):
            self.Vertices.append(vertex)
            self.__AddVertexToIntHash(vertex)
        pass

    def __IndexToFace(self, index: tuple[int, int, int], vertices: list[tuple[float, float, float]]):
        ret = []
        for i in index:
            vertex = vertices[i - 1]
            ret.append(self.VertexToIntHash[vertex])
        face = (3, ret[0], ret[1], ret[2])
        if (not face in self.Faces):
            self.Faces.append(face)


if (__name__ == "__main__"):
    import pyvista as pv
    box1 = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), gp_Pnt(1, 1, 1)).Shape()

    tlt = TessellatorShape(box1)
    points = tlt.Mesh_.GetMeshPoints()
    # 포인트와 면을 사용하여 메쉬를 생성합니다.
    mesh = pv.PolyData(points)
    

    # 메쉬를 시각화합니다.
    mesh.plot()

