import util

import os
from Tessellator.TessellatorCompound import TessellatorCompound
from ModelUtil.RGClusterInfo import RGInfo
from Reader.STLReader import STPReader
import pyvista as pv


class RegionGrowing:
    def __init__(self, fileName, k) -> None:
        self.FileName: str = fileName
        rginfo = RGInfo()
        self.MeshList = []
        self.Neighbors = []
        self.Tlt = None
        self.__Initialization()

        self.ClosedSet = []
        self.Buffer = []
        self.PartName: str = os.path.splitext(self.FileName)[0]
        self.W: float = rginfo.HashNametoW[(self.PartName, k)]
        self.K: int = rginfo.HashNametoK[(self.PartName, k)] - 1
        self.N: int = len(self.MeshList)

        self.Colors = ["red", "blue", "yellow", "green", "purple", "pink",
                       "black", "Orange", "gray", "brown", "olive", "beige", "violet", "navy"]
        pass

    def Run(self):
        # 이웃 리스트
        for i in self.Tlt.PartList:
            self.Neighbors.append(i.Neighbors)

        finalCluster = []
        self.ClosedSet = []
        for _ in range(len(self.MeshList)):
            min_vol = 1e9
            min_index = -1
            for i in range(len(self.MeshList)):
                if (self.__GetBoundingBox(self.MeshList[i]) < min_vol) and (not i in self.ClosedSet):
                    min_index = i
                    min_vol = self.__GetBoundingBox(self.MeshList[i])

            if (not min_index in self.ClosedSet and min_index != -1):
                self.K -= 1
                self.__Dfs(min_index, self.W, self.__GetBoundingBox(
                    self.MeshList[min_index]))
                cluster = []
                for i in self.Buffer:
                    cluster.append(i)
                finalCluster.append(cluster)
                self.Buffer.clear()

        clusteredMeshes: list[pv.PolyData] = [pv.PolyData()
                                              for _ in range(len(finalCluster))]

        for i in range(len(finalCluster)):
            for j in range(len(finalCluster[i])):
                if (self.MeshList[finalCluster[i][j]] is not None):
                    if (clusteredMeshes[i] is None):
                        mesh: pv.PolyData = pv.PolyData()
                        mesh.deep_copy(self.MeshList[finalCluster[i][j]])
                        clusteredMeshes[i] = mesh
                    else:
                        mesh: pv.PolyData = pv.PolyData()
                        mesh.deep_copy(
                            self.MeshList[finalCluster[i][j]])
                        clusteredMeshes[i] += mesh

        return clusteredMeshes

    def __Initialization(self):
        stpReader = STPReader(self.FileName)
        self.Tlt = TessellatorCompound(stpReader.STPShape)

        meshList: list[pv.PolyData] = self.Tlt.PyvistaMeshList()

        for i in range(len(meshList)):
            meshList[i] = meshList[i].clean()
            meshList[i] = meshList[i].triangulate()

        self.MeshList = meshList
        return

    def __GetBoundingBox(self, mesh: pv.PolyData):
        return mesh.n_faces

    def __Dfs(self, part, percent, faces):
        if (part in self.ClosedSet):
            return
        self.N -= 1
        if (self.N == self.K):
            self.N += 1
            return

        self.ClosedSet.append(part)
        self.Buffer.append(part)

        for j in range(len(self.Neighbors[part])):
            if (percent * self.__GetBoundingBox(self.MeshList[self.Neighbors[part][j]]) < faces):
                if (self.Neighbors[part][j] in self.ClosedSet):
                    continue
                self.__Dfs(self.Neighbors[part][j], percent, faces)
        pass

    def Display(self):
        plt = pv.Plotter()
        ml = self.Run()
        for i in range(len(ml)):
            plt.add_mesh(ml[i], self.Colors[i])

        plt.show()
        pass


if (__name__ == "__main__"):
    plt = pv.Plotter()

    rg = RegionGrowing("ControlValve.stp", 2)
    rg.Display()
