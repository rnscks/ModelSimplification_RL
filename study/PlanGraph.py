import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 평면 방정식의 계수
A, B, C, D = 1, 2, 3, -10

# 평면의 법선 벡터
normal = np.array([A, B, C])

# 평면 위의 점을 생성
point_on_plane = normal * D / np.linalg.norm(normal)

# 그래프 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 평면 위의 그리드 포인트 생성
xx, yy = np.meshgrid(range(-10, 11), range(-10, 11))
zz = (-A * xx - B * yy - D) / C

# 평면 그리기
ax.plot_surface(xx, yy, zz, alpha=0.5)

# 축 레이블 설정
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
