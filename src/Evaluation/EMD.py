import util
import pyvista as pv
import numpy as np
from scipy.optimize import linear_sum_assignment


class EMD:
    def __init__(self, mesh1: pv.PolyData, mesh2: pv.PolyData) -> None:
        self.M1 = mesh1
        self.M2 = mesh2
        pass

    def Run(self):
        p1 = self.M1.points
        p2 = self.M2.points
        # 거리 매트릭스 계산 (유클리드 거리 사용)
        distanceMatrix = np.zeros(len(p1), len(p2))
        for i in range(len(p1)):
            for j in range(len(p2)):
                distanceMatrix[i][j] = np.linalg.norm(p1[i] - p2[j])

        # Linear Sum Assignment을 통해 EMD 계산
        row_ind, col_ind = linear_sum_assignment(distanceMatrix)
        emdDistance = distanceMatrix[row_ind, col_ind].sum()
        return emdDistance
