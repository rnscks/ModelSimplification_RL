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
from ModelUtil.ModelFileList import FileNameList


class RGEx:
    def __init__(self, fileName, numbering, iteration, modelName, isModelStore, isPlotStore, exSetNumber, k) -> None:
        self.ExSetNumber = exSetNumber
        self.IsPlotStore = isPlotStore
        self.IsModelStore = isModelStore
        self.FileName = fileName
        self.ModelName = modelName
        self.DataSet = {'Model Name': [],
                        'Algorithm': [],
                        'K': [],
                        'CD': []}
        self.Numbering = numbering
        self.Iteration = iteration
        self.RootMkd = MakeDir(self.ModelName + str(self.Numbering))
        self.K = k
        self.MkdID = 1
        pass

    def SetK(self, k):
        self.K = k
        self.Mkd = MakeDir(
            self.ModelName + str(self.Numbering) + str(self.MkdID))
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
                self.__WriteSTLModel(
                    'QEM(Optimized)', resultModel, len(meshList), j)
                self.__WriteSTLModel(
                    'QEM(Merged)', mergedDecimatedModel, len(meshList), j)
                self.__WriteSTLModel(
                    'QEM(Separated)', speratedDecimatedModel, len(meshList), j)

            self.__WriteDataSet(resultModel, refModel,
                                meshList, acorEndTime, 'QEM(Optimized)')
            self.__WriteDataSet(mergedDecimatedModel,
                                refModel, meshList, mergedQemEndTime, 'QEM(Merged)')
            self.__WriteDataSet(speratedDecimatedModel,
                                refModel, meshList, speratedQemEndTime, 'QEM(Separated)')

    def __WriteSTLModel(self, id: str, model: pv.PolyData, k: int, index: int):
        model.save(os.path.join(self.Mkd.FolderPath, str(k) + '_' +
                   id + '_' + str(index) + '_' + self.FileName[:4] + '.stl'))
        pass

    def __WriteDataSet(self, model, refmodel, meshList, time, id):
        cfd = ChamferDistance(model, refmodel)
        self.DataSet['Model Name'].append(self.ModelName)
        self.DataSet['Algorithm'].append(id)
        self.DataSet['K'].append(str(len(meshList)))
        self.DataSet['CD'].append(cfd.TriRun())

    def Done(self):
        df = pd.DataFrame(self.DataSet)
        if (self.IsPlotStore is True):
            mpld = MeanLinePlotDrawer(
                exprimentDataSet=df, modelName=self.ModelName, numbering=self.Numbering)
            pld = BoxPlotDrawer(
                exprimentDataSet=df, modelName=self.ModelName, numbering=self.Numbering)
            mpld.Run()
            pld.Run()

        return df

    def __CalDeicimatePercentByFace(self, mesh: pv.PolyData, refModel: pv.PolyData):
        target_faces = mesh.n_faces

        current_faces = refModel.n_faces
        decimation_factor = (current_faces - target_faces) / current_faces
        return decimation_factor


if (__name__ == "__main__"):
    fnl = FileNameList()
    modelName = fnl.CurrentModelName()
    fileName = fnl.CurrentModleFileName()
    partNumber = fnl.HashFileNameToPartNumber[fileName]
    idNumbering = 0

    while (True):
        rgex = RGEx(fileName, idNumbering, 100, modelName, True, True, 10, 1)
        for i in range(1, partNumber + 1):
            rgex.SetK(i)
            resultDataFrame: pd.DataFrame = rgex.Run()

        rgex.Done()
        idNumbering += 1
