import util
import os
import pyvista as pv
from Cluster.RegionGrowing import RegionGrowing
from Optimization.ACOR import ACOR
from ModelUtil.ModelReader import CompoundedReader
from PlotDraw.BoxPlotDrawer import BoxPlotDrawer
from PlotDraw.MeanLinePoltDrawer import MeanLinePlotDrawer
import pandas as pd
import numpy as np
import time
from Evaluation.ChamferDistance import ChamferDistance
from MakeDir import MakeDir


class RGEx:
    def __init__(self, fileName, numbering, iteration, modelName, isModelStore, isPlotStore, exSetNumber) -> None:
        self.ExSetNumber = exSetNumber
        self.IsPlotStore = isPlotStore
        self.IsModelStore = isModelStore
        self.FileName = fileName
        self.ModelName = modelName
        self.DataSet = {'Model Name': [],
                        'Algorithm': [],
                        'K': [],
                        'mesh': [],
                        'CD': [],
                        'Vertex CD': [],
                        'Triangle CD': [],
                        'Volume': [],
                        'Decimate Percent': [],
                        'time': []}
        self.Numbering = numbering
        self.Iteration = iteration
        self.RootMkd = MakeDir(self.FileName[:4] + str(self.Numbering))
        self.MkdID = 0
        pass

    def SetK(self, k):
        self.K = k
        self.Mkd = MakeDir(
            self.FileName[:4] + str(self.Numbering) + str(self.MkdID))
        self.Mkd.IntoDir(self.RootMkd.DirName)
        self.MkdID += 1

    def Run(self):
        cr = CompoundedReader(self.FileName)
        refModel = cr.RunForPyvista()
        if (self.IsModelStore is True):
            self.Mkd.Run()
            self.__WriteSTLModel('ref', refModel, 0, 0)

        for j in range(self.ExSetNumber):
            rg = RegionGrowing(fileName=self.FileName, k=self.K)
            meshList = rg.Run()

            acor = ACOR(columnOfArchive=50, numberOfCluster=len(meshList), objectivePercent=0.5,
                        q=0.1, zeta=0.85, iteration=self.Iteration, objectiveFace=1000, refModel=refModel, meshList=meshList)

            acorStartTime = time.time()
            optimizationResult = acor.Run()
            acorEndTime = time.time() - acorStartTime

            resultModel: pv.PolyData = pv.PolyData()

            for i in range(len(meshList)):
                decimatedMesh = meshList[i].decimate(optimizationResult[i])
                resultModel += decimatedMesh

            decimationFactor = self.__CalDeicimatePercentByFace(
                resultModel, refModel)
            mergedQemStartTime = time.time()
            mergedDecimatedModel = refModel.decimate(decimationFactor)
            mergedQemEndTime = time.time() - mergedQemStartTime

            speratedDecimatedModel: pv.PolyData = pv.PolyData()

            speratedQemStartTime = time.time()
            for i in range(len(meshList)):
                decimatedMesh = meshList[i].decimate(decimationFactor)
                speratedDecimatedModel += decimatedMesh

            speratedQemEndTime = time.time() - speratedQemStartTime

            if (self.IsModelStore is True):
                self.__WriteSTLModel('Optimization', resultModel, len(meshList), j)
                self.__WriteSTLModel(
                    'Merged QEM', mergedDecimatedModel, len(meshList), j)
                self.__WriteSTLModel(
                    'Separate QEM', speratedDecimatedModel, len(meshList), j)
                
            self.__WriteDataSet(resultModel, refModel,
                                meshList, acorEndTime, 'Optimization')
            self.__WriteDataSet(mergedDecimatedModel,
                                refModel, meshList, mergedQemEndTime, 'Merged QEM')
            self.__WriteDataSet(speratedDecimatedModel,
                                refModel, meshList, speratedQemEndTime, 'Separate QEM')

    def __WriteSTLModel(self, id: str, model: pv.PolyData, k: int, index: int):
        model.save(os.path.join(self.Mkd.FolderPath, str(k) + '_' +
                   id + '_' + str(index) + '_' + self.FileName[:4] + '.stl'))
        pass

    def __WriteDataSet(self, model, refmodel, meshList, time, id):
        cfd = ChamferDistance(model, refmodel)
        self.DataSet['Model Name'].append(self.ModelName)
        self.DataSet['Algorithm'].append(id)
        self.DataSet['K'].append(str(len(meshList)))
        self.DataSet['mesh'].append(model.n_faces)
        self.DataSet['Vertex CD'].append(cfd.VRun())
        self.DataSet['Triangle CD'].append(cfd.TriRun())
        self.DataSet['CD'].append(cfd.Run())
        self.DataSet['time'].append(time)
        self.DataSet['Decimate Percent'].append(
            model.n_faces / refmodel.n_faces)
        self.DataSet['Volume'].append(
            np.abs(model.volume - refmodel.volume))

    def Done(self):  
        df = pd.DataFrame(self.DataSet)
        if (self.IsPlotStore is True):
            mpld = MeanLinePlotDrawer(exprimentDataSet = df, modelName=self.ModelName)
            pld = BoxPlotDrawer(exprimentDataSet = df, modelName=self.ModelName)
            mpld.Run()
            pld.Run()

        return df

    def __CalDeicimatePercentByFace(self, mesh: pv.PolyData, refModel: pv.PolyData):
        target_faces = mesh.n_faces

        current_faces = refModel.n_faces
        decimation_factor = (current_faces - target_faces) / current_faces
        return decimation_factor


if (__name__ == "__main__"):
    fileName = input()
    ex = RGEx(fileName, 0, 1)
    w = []
    k = []
    n = 14

    for i in range(n):
        w.append(float(input()))

    for i in range(n):
        k.append(int(input()))

    for i in range(n):
        ex.SetK(k[i])
        ex.Run()

    ex.Done()
