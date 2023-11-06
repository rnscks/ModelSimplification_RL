import util
import pyvista as pv
import open3d as o3d
import numpy as np
from ModelUtil.ModelReader import CompoundedReader
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree


class PMD:
    def __init__(self, mesh1: pv.PolyData, mesh2: pv.PolyData) -> None:
        self.M1 = mesh1
        self.M2 = mesh2
        pass

    def Run(self):
        fpmd = self.__GetMinTriangle(self.M1, self.M2)
        bpmd = self.__GetMinTriangle(self.M2, self.M1)
        return np.sum(fpmd) + np.sum(bpmd)

    def __GetTriangleCentroids(self, mesh):
        vertices = np.array(mesh.vertices)
        triangles = np.array(mesh.triangles)
        triangleCentroids = []

        for triangle in triangles:
            triangleVertices = vertices[triangle]
            triangleCenter = np.mean(triangleVertices, axis=0)
            triangleCentroids.append(triangleCenter)

        return triangleCentroids

    def __GetMinTriangle(self, m1, m2):
        originalCentroids = self.__GetTriangleCentroids(m1)
        targetCentroids = self.__GetTriangleCentroids(m2)
        originalTree = cKDTree(originalCentroids)
        minDistances = []
        for targetCentroid in targetCentroids:
            _, index = originalTree.query(targetCentroid)
            minDistances.append(np.linalg.norm(
                targetCentroid - originalCentroids[index]))
        return minDistances
