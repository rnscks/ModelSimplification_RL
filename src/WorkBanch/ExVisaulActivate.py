import WorkBanch.util as util

import pyvista as pv
from Cluster.RegionGrowing import RegionGrowing
from Optimization.ACOR import ACOR
from ModelUtil.ModelReader import CompoundedReader, ModelListReader
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


class RGExVisual:
    def __init__(self, fileName, numbering, iteration, w, k) -> None:
        self.FileName = fileName
        self.DataSet = {'id': [],
                        'w': [],
                        'k': [],
                        'mesh': [],
                        'CD': [],
                        'volume': []}
        self.Numbering = numbering
        self.Iteration = iteration
        self.W = w
        self.K = k
        pass

    def Run(self):
        rg = RegionGrowing(fileName=self.FileName, k=self.K, w=self.W)
        meshList = rg.Run()

        cr = CompoundedReader(self.FileName)
        refModel = cr.Run()

        acor = ACOR(columnOfArchive=50, numberOfCluster=len(meshList), objectivePercent=0.5,
                    q=0.1, zeta=0.85, iteration=self.Iteration, objectiveFace=1000, refModel=refModel, meshList=meshList)

        optimizationResult = acor.Run()
        resultModel: pv.PolyData = pv.PolyData()

        for i in range(len(meshList)):
            decimatedMesh = meshList[i].decimate(optimizationResult[i])
            resultModel += decimatedMesh

        decimationFactor = self.__CalDeicimatePercentByFace(
            resultModel, refModel)
        decimatedModel = refModel.decimate(decimationFactor)
        self.__Display(resultModel, decimatedModel,
                       refModel, optimizationResult)

    def __CalDeicimatePercentByFace(self, mesh: pv.PolyData, refModel: pv.PolyData):
        target_faces = mesh.n_faces

        current_faces = refModel.n_faces
        decimation_factor = (current_faces - target_faces) / current_faces
        return decimation_factor

    def __GetChamferDistance(self, S1: np.array, S2: np.array):
        distances1 = cdist(S1, S2, 'euclidean')
        distances2 = cdist(S2, S1, 'euclidean')

        if (len(distances1) == 0 or len(distances2) == 0):
            return None

        min_distances1 = np.min(distances1, axis=1)
        min_distances2 = np.min(distances2, axis=1)

        return (sum(min_distances1) / len(S1) + sum(min_distances2) / len(S2))

    def __Display(self, resultModel, decimatedModel, refModel, decimatePercentTable):

        plotter = pv.Plotter(shape=(1, 3))
        plotter.background_color = "white"

        plotter.subplot(0, 1)
        refModel.save("o.stl")

        plotter.add_text("Optimization", position=(30.2, 140.9),
                         font_size=24, color='black', font='arial')
        plotter.add_text("Mesh: " + str(resultModel.n_faces),
                         position=(30.2, 80.9), font_size=24, color='black', font='arial')
        plotter.add_text("Decimate Percent: " + str(resultModel.n_faces / refModel.n_faces), position=(10, 950),
                         font_size=24, color='black', font='arial')
        plotter.add_text("Chamfer Distance: " + str(round(self.__GetChamferDistance(resultModel.points,
                         refModel.points, 1), 2)), position=(30.2, 30.9), font_size=24, color='black', font='arial')
        resultModel.save("op.stl")
        plotter.add_mesh(resultModel, color='red')

        plotter.subplot(0, 0)
        plotter.add_text("QEM", position=(30.2, 140.9),
                         font_size=24, color='black', font='arial')
        plotter.add_text("Chamfer Distance: " + str(round(self.__GetChamferDistance(decimatedModel.points, refModel.points, 1), 2)), position=(30.2, 30.9),
                         font_size=24, color='black', font='arial')
        plotter.add_text("Mesh: " + str(decimatedModel.n_faces), position=(30.2, 80.9),
                         font_size=24, color='black', font='arial')
        plotter.add_text("Decimate Percent: " + str(decimatedModel.n_faces / refModel.n_faces), position=(10, 950),
                         font_size=24, color='black', font='arial')
        decimatedModel.save("all.stl")

        plotter.add_mesh(decimatedModel, color='red')
        plotter.subplot(0, 2)

        plotter.add_text("w =" + str(self.W) + ',' + "k ="+str(self.K), position=(30.2, 30.9),
                         font_size=24, color='black', font='arial')
        colors = ["red", "white", "yellow", "green",
                  "blue", "purple", "pink", "black", "brown"]
        meshList = ModelListReader(self.FileName).Run()

        for i in range(len(meshList)):
            if (meshList[i] is not None):
                decimatedModel = meshList[i].decimate(decimatePercentTable[i])
                index = (i) % len(colors)
                plotter.add_mesh(decimatedModel, color=colors[index])

        plotter.show()


if (__name__ == "__main__"):
    fileName = input()
    ex = RGExVisual(fileName, 0, 100, 0.5, 100)

    ex.Run()
