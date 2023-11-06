import util

from Reader.STLReader import STPReader
from Tessellator.TessellatorCompound import TessellatorCompound
import pyvista as pv
import open3d as o3d
from open3d.cpu.pybind.geometry import TriangleMesh as o3dMesh #type: ignore


class CompoundedReader:
    def __init__(self, fileName) -> None:
        self.FileName = fileName
        pass

    def RunForPyvista(self):
        stpReader = STPReader(self.FileName)
        self.Tlt = TessellatorCompound(stpReader.STPShape)

        meshList: list[pv.PolyData] = self.Tlt.PyvistaMeshList()

        for i in range(len(meshList)):
            meshList[i] = meshList[i].clean()
            meshList[i] = meshList[i].triangulate()

        poly = pv.PolyData()

        for i in range(len(meshList)):
            poly += meshList[i]
        return poly

    def RunForO3d(self):
        stpReader = STPReader(self.FileName)
        self.Tlt = TessellatorCompound(stpReader.STPShape)

        meshList: list[o3dMesh] = self.Tlt.O3dMeshList()
        resultMesh: o3dMesh = o3d.geometry.TriangleMesh()

        for i in range(len(meshList)):
            resultMesh += meshList[i]
        return resultMesh


class ModelListReader:
    def __init__(self, fileName) -> None:
        self.FileName = fileName
        pass

    def RunForPyvista(self) -> list[pv.PolyData]:
        stpReader = STPReader(self.FileName)
        self.Tlt = TessellatorCompound(stpReader.STPShape)

        meshList: list[pv.PolyData] = self.Tlt.PyvistaMeshList()

        for i in range(len(meshList)):
            meshList[i] = meshList[i].clean()
            meshList[i] = meshList[i].triangulate()

        return meshList

    def RunForO3d(self) -> list[o3dMesh]:
        stpReader = STPReader(self.FileName)
        self.Tlt = TessellatorCompound(stpReader.STPShape)

        meshList: list[o3dMesh] = self.Tlt.O3dMeshList()
        return meshList
