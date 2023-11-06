import util

from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Iterator, TopoDS_Shape
from Reader.STLReader import STPReader

import pyvista as pv
from Tessellator.TessellatorShape import TessellatorShape
import numpy as np
from scipy.spatial.distance import cdist
import open3d as o3d
from open3d.cpu.pybind.geometry import TriangleMesh as o3dMesh


class TessellatorCompound:
    def __init__(self, compound: TopoDS_Compound) -> None:
        self.Compound: TopoDS_Compound = compound
        self.PartList: list[TessellatorShape] = []
        self.__InitCompound()
        self.__InitNeighbors()
        pass

    def __InitCompound(self):
        shapeIter: TopoDS_Iterator = TopoDS_Iterator(self.Compound)
        while (shapeIter.More()):
            shape: TopoDS_Shape = shapeIter.Value()
            self.PartList.append(TessellatorShape(shape))
            shapeIter.Next()

    def PyvistaMeshList(self):
        meshList: list[pv.PolyData] = []
        for mesh in self.PartList:
            meshList.append(pv.PolyData(mesh.Mesh_.Vertices, mesh.Mesh_.Faces))
        return meshList

    def O3dMeshList(self):
        meshModelList: list[o3dMesh] = []
        for mesh in self.PartList:
            triangles = mesh.Mesh_.Faces
            vertices = np.asarray(mesh.Mesh_.Vertices)
            o3dTriangles = np.ndarray(
                shape=(len(triangles), len(triangles[0]) - 1))

            for i in range(len(triangles)):
                o3dTriangles[i][0] = triangles[i][1]
                o3dTriangles[i][1] = triangles[i][2]
                o3dTriangles[i][2] = triangles[i][3]

            o3dTriangles = np.asarray(o3dTriangles)
            o3dMeshModel: o3dMesh = o3d.geometry.TriangleMesh()
            o3dMeshModel.vertices = o3d.utility.Vector3dVector(vertices)
            o3dMeshModel.triangles = o3d.utility.Vector3iVector(o3dTriangles)

            meshModelList.append(o3dMeshModel)

        return meshModelList

    def __InitNeighbors(self):
        for i in range(len(self.PartList)):
            for j in range(len(self.PartList)):
                if (i == j):
                    continue

                if (not self.PartList[i].bndBox.IsOut(self.PartList[j].bndBox)):
                    if (not i in self.PartList[j].Neighbors):
                        self.PartList[j].Neighbors.append(i)
                        self.PartList[i].Neighbors.append(j)


def GetChamferDistance(S1, S2):
    S1 = np.asarray(S1)
    S2 = np.asarray(S2)
    # S1의 각 포인트에서 S2 포인트까지의 거리 행렬 계산
    distances1 = cdist(S1, S2, 'euclidean')
    distances2 = cdist(S2, S1, 'euclidean')

    # 각 S1 포인트에서 가장 짧은 거리 추출
    min_distances1 = np.min(distances1, axis=1)
    min_distances2 = np.min(distances2, axis=1)

    return sum(min_distances1) / len(S1) + sum(min_distances2) / len(S2)


if (__name__ == "__main__"):
    import pyvista as pv
    plotter = pv.Plotter()
    plotter.background_color = "white"
    stpReader = STPReader("AirCompressor_Filtered.stp")
    tlt = TessellatorCompound(stpReader.STPShape)

    ret = 0
    for mesh in tlt.PartList:
        mesh = pv.PolyData(mesh.Vertices, mesh.Faces)
        cleanedmesh = mesh.clean()
        cleanedmesh = cleanedmesh.triangulate()
        deci = cleanedmesh.decimate(0.99)
        plotter.add_mesh(deci)
        ret += GetChamferDistance(deci.points, mesh.points)

    print(ret)
    plotter.show()
