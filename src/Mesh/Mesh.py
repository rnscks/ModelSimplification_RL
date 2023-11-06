import WorkBanch.util as util

from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Display.OCCViewer import Viewer3d

import math as m
import numpy as np
import random

from OCC.Core.Bnd import Bnd_Box
from Shaper.Shaper import Shaper
from scipy.spatial.distance import cdist


class Mesh:
    def __init__(self, vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int, int]]) -> None:
        self.Vertices: list[tuple[float, float, float]] = vertices
        self.Faces = faces
        self.TriArea: np.ndarray = []
        self.SamplingPoints = []
        pass

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

    def GetMeshPoints(self):
        points = []
        for vertex in self.Vertices:
            if (vertex == None):
                continue
            points.append(vertex)

        for point in self.SamplingPoints:
            points.append(point)

        return points

    def __InitTriangleArea(self):
        for index in self.Faces:    
            v1 = gp_Vec(gp_Pnt(self.Vertices[index[1]][0],self.Vertices[index[1]][1],self.Vertices[index[1]][2]),
                        gp_Pnt(self.Vertices[index[2]][0],self.Vertices[index[2]][1],self.Vertices[index[2]][2]))
            v2 = gp_Vec(gp_Pnt(self.Vertices[index[1]][0],self.Vertices[index[1]][1],self.Vertices[index[1]][2]),
                        gp_Pnt(self.Vertices[index[3]][0],self.Vertices[index[3]][1],self.Vertices[index[3]][2]))
            area = (0.5) * (v1.CrossMagnitude(v2))
            self.TriArea.append(area)

    def RandomSampling(self, num):
        self.__InitTriangleArea()
        self.TriArea = np.asarray(self.TriArea)

        cumSumArea = np.cumsum(self.TriArea)
        totalArea = cumSumArea[-1]
        cnt = [0 for _ in range(len(self.Faces))]

        for _ in range(num):
            randomArea = random.uniform(0, totalArea)

            triIndex = np.searchsorted(cumSumArea, randomArea)

            cnt[triIndex] += 1
            A = gp_Vec(self.Vertices[self.Faces[triIndex][1]][0], self.Vertices[self.Faces[triIndex][1]][1],self.Vertices[self.Faces[triIndex][1]][2])
            B = gp_Vec(self.Vertices[self.Faces[triIndex][2]][0], self.Vertices[self.Faces[triIndex][2]][1],self.Vertices[self.Faces[triIndex][2]][2])
            C = gp_Vec(self.Vertices[self.Faces[triIndex][3]][0], self.Vertices[self.Faces[triIndex][3]][1],self.Vertices[self.Faces[triIndex][3]][2])

            r1 = random.random()
            r2 = random.random()

            A.Multiply(1 - m.sqrt(r1))
            B.Multiply(m.sqrt(r1) * (1 - r2))
            C.Multiply((m.sqrt(r1)) * r2)

            pnt = A + B + C
            pnt = gp_Pnt(pnt.XYZ())
            self.SamplingPoints.append(pnt.Coord())
