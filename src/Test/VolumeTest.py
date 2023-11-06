import util
import os
import open3d as o3d
import numpy as np

import pyvista as pv
import matplotlib.pyplot as plt
from open3d.cpu.pybind.geometry import TriangleMesh as o3dMesh
from ModelUtil.ModelReader import CompoundedReader
from ModelUtil.ModelFileList import FileNameList

fl = FileNameList()
x = []
pyvistaVolumes = []
pyvistaBoundingVolumes = []
o3dBoundingVolumes = []
for _ in range(fl.MaxSize):
    cr = CompoundedReader(fl.CurrentModleFileName())
    x.append(fl.CurrentModelName())

    pyvistaModel = cr.RunForPyvista()
    o3dModel: o3dMesh = cr.RunForO3d()
    bounding_box = pyvistaModel.bounds

    # 바운딩 박스의 부피 계산
    bounding_box_volume = (bounding_box[1]-bounding_box[0]) * (
        bounding_box[3]-bounding_box[2]) * (bounding_box[5]-bounding_box[4])
    pyvistaVolumes.append(pyvistaModel.volume)
    pyvistaBoundingVolumes.append(bounding_box_volume)
    o3dBoundingVolumes.append(o3dModel.get_oriented_bounding_box().volume())
    fl.Next()

# X 축에 문자열 레이블을 사용하여 선 그래프 그리기
plt.plot(x, pyvistaVolumes, label='pyvista volumes', marker='o')
plt.plot(x, pyvistaBoundingVolumes,
         label='pyvista bounding volumes', marker='s')
plt.plot(x, o3dBoundingVolumes, label='o3d bounding volumes', marker='d')

# 그래프 제목 및 축 레이블 설정
plt.title('Volume Chart', fontsize=40)


plt.xlabel('Model Name', fontsize=30)

# 범례 표시
plt.ylabel('Volume', fontsize=30)
plt.legend(fontsize=20)

# X 축 레이블 폰트 크기 조절
plt.xticks(fontsize=30)

# Y 축 레이블 폰트 크기 조절
plt.yticks(fontsize=20)
plt.show()
