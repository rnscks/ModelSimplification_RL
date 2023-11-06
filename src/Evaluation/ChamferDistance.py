import util
import pyvista as pv
import open3d as o3d
import numpy as np
from ModelUtil.ModelReader import CompoundedReader
from scipy.spatial.distance import cdist


class ChamferDistance:
    def __init__(self, mesh1: pv.PolyData, mesh2: pv.PolyData) -> None:
        self.M1 = mesh1
        self.M2 = mesh2
        pass

    def __MakePointCloud(self, mesh: pv.PolyData):
        vertices = mesh.points
        triangles = mesh.faces.reshape(-1, 4)
        tri = np.zeros((triangles.shape[0], triangles.shape[1] - 1))
        for i in range(len(triangles)):
            tri[i][0] = triangles[i][1]
            tri[i][1] = triangles[i][2]
            tri[i][2] = triangles[i][3]

        o3d_mesh = o3d.geometry.TriangleMesh()

        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(tri)
        sampled_points = o3d_mesh.sample_points_uniformly(
            number_of_points=10000, use_triangle_normal=False)
        arr = np.asarray(sampled_points.points)
        return arr

    def VRun(self):
        s1 = self.M1.points
        s2 = self.M2.points

        distances1 = cdist(s1, s2, 'euclidean')
        distances2 = cdist(s2, s1, 'euclidean')

        if (len(distances1) == 0 or len(distances2) == 0):
            return None

        min_distances1 = np.min(distances1, axis=1)
        min_distances2 = np.min(distances2, axis=1)

        return (sum(min_distances1) / len(s1) + sum(min_distances2) / len(s2))

    def __GetTriangleCenterPoint(self, model: pv.PolyData):
        ret = []
        triangles = model.faces.reshape(-1, 4)

        for tri in triangles:
            p1 = model.points[tri[1]]
            p2 = model.points[tri[2]]
            p3 = model.points[tri[3]]

            centroid = (p1 + p2 + p3) / 3
            ret.append(centroid)

        for p in model.points:
            ret.append(p)

        np.asarray(ret)

        return ret

    def TriRun(self):
        s1 = self.__GetTriangleCenterPoint(self.M1)
        s2 = self.__GetTriangleCenterPoint(self.M2)

        distances1 = cdist(s1, s2, 'euclidean')
        distances2 = cdist(s2, s1, 'euclidean')

        if (len(distances1) == 0 or len(distances2) == 0):
            return None

        min_distances1 = np.min(distances1, axis=1)
        min_distances2 = np.min(distances2, axis=1)

        return (sum(min_distances1) / len(s1) + sum(min_distances2) / len(s2))

    def Run(self):
        s1 = self.__MakePointCloud(self.M1)
        s2 = self.__MakePointCloud(self.M2)

        distances1 = cdist(s1, s2, 'euclidean')
        distances2 = cdist(s2, s1, 'euclidean')

        if (len(distances1) == 0 or len(distances2) == 0):
            return None

        min_distances1 = np.min(distances1, axis=1)
        min_distances2 = np.min(distances2, axis=1)

        return (sum(min_distances1) / len(s1) + sum(min_distances2) / len(s2))


if (__name__ == "__main__"):

    pass
