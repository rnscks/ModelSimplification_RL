import util
from Evaluation.ChamferDistance import ChamferDistance

import numpy as np
import open3d as o3d

from scipy.spatial.distance import cdist
import pyvista as pv


class ACOR:
    def __init__(self, columnOfArchive, numberOfCluster, objectivePercent, q, zeta, iteration, refModel, objectiveFace, meshList) -> None:
        self.q = q
        self.zeta = zeta
        self.ObjectivePercent = objectivePercent
        self.Iteration = iteration
        self.N = numberOfCluster
        self.K = columnOfArchive
        self.MeshList = meshList
        self.SolutionArchive = np.zeros((self.K, self.N + 2))
        self.OmegaList = np.zeros((self.K, ))
        self.RefModel: pv.PolyData = pv.PolyData()
        self.RefModel.deep_copy(refModel)
        self.ObjectFace = objectiveFace
        self.InitialCD = 0
        # self.__InitializeInitialCD()
        self.__InitializeSolutionArchive()

        return

    def Run(self):
        for i in range(self.Iteration):
            print(i)
            self.__SelectTheGFunctionSamplingNumbers()

        return self.SolutionArchive[0]

    def __DecimateByMatrix(self, percents, meshList):
        deci = None
        for i in range(len(percents)):
            mesh: pv.PolyData = pv.PolyData()
            mesh.deep_copy(meshList[i])

            if (deci == None):
                deci = mesh.decimate(percents[i])
            else:
                deci += mesh.decimate(percents[i])
        chd = ChamferDistance(deci, self.RefModel)

        bi = chd.VRun()
        ci = deci.n_faces / self.RefModel.n_faces

        return bi * ci

    def __CalOmega(self, index):
        return ((self.q * self.K * np.sqrt(2 * np.pi)) ** -1) * np.exp(-(np.square(index - 1)/(2 * np.square(self.q) * np.square(self.K))))

    def __InitializeInitialCD(self) -> None:
        decimatedModel = pv.PolyData()
        decimatedModel.deep_copy(self.RefModel)
        decimatedModel.decimate(90)
        cfd = ChamferDistance(decimatedModel, self.RefMode)
        self.InitialCD = cfd.TriRun()
        return

    def __InitializeSolutionArchive(self) -> None:
        DEVIATION = 1

        for i in range(self.K):
            sampingNumbers = []

            while len(sampingNumbers) < self.N:
                sampleNumber = np.random.normal(
                    self.ObjectivePercent, DEVIATION)
                if (self.ObjectivePercent + 0.3 <= sampleNumber < self.ObjectivePercent + 0.4):
                    sampingNumbers.append(sampleNumber)
            sampingNumbers = np.asarray(sampingNumbers)

            for j in range(len(sampingNumbers)):
                self.SolutionArchive[i, j] = sampingNumbers[j]
            self.SolutionArchive[i, self.N] = self.__DecimateByMatrix(
                sampingNumbers, self.MeshList)

        for i in range(self.K):
            self.SolutionArchive[i, self.N + 1] = self.__CalOmega(i)
            self.OmegaList[i] = self.SolutionArchive[i, self.N + 1]
        sortedIndices = np.argsort(self.SolutionArchive[:, self.N + 1])
        self.SolutionArchive = np.copy(self.SolutionArchive[sortedIndices])
        return

    def __SelectTheGFunctionSamplingNumbers(self) -> None:
        p = self.OmegaList/sum(self.OmegaList)
        ret = np.zeros((self.N))

        for i in range(self.N):
            l = np.random.choice(len(self.OmegaList), p=p)
            deviation = 0

            for j in range(self.K):
                deviation += np.absolute(
                    self.SolutionArchive[j, i] - self.SolutionArchive[l, i])
            deviation *= ((self.zeta) * ((self.K - 1) ** -1))

            while True:
                ret[i] = np.random.normal(
                    self.SolutionArchive[l, i], deviation)
                if (self.ObjectivePercent + 0.3 <= ret[i] <= self.ObjectivePercent + 0.4):
                    break

        self.__AppendSolution(ret)

        return

    def __AppendSolution(self, solution):
        currentObjectiveValue = self.__DecimateByMatrix(
            solution, self.MeshList)
        if (currentObjectiveValue > self.SolutionArchive[self.K - 1,
                                                         self.N]):
            return

        for i in range(self.N):
            self.SolutionArchive[self.K - 1, i] = solution[i]
        self.SolutionArchive[self.K - 1,
                             self.N] = self.__DecimateByMatrix(solution, self.MeshList)
        sortedIndices = np.argsort(self.SolutionArchive[:, self.N])
        self.SolutionArchive = np.copy(self.SolutionArchive[sortedIndices])

        for i in range(self.K):
            self.SolutionArchive[i, self.N + 1] = self.__CalOmega(i)
            self.OmegaList[i] = self.SolutionArchive[i, self.N + 1]
