import torch
import torch.nn as nn

# 가상의 3차원 데이터 (예: 배치 크기=32, 높이=10, 너비=10)
data = torch.rand(32, 10, 10)

# 데이터 평탄화 (배치 차원을 유지하면서 나머지 차원을 하나의 차원으로 펼침)
flattened_data = data.view(32, -1)

# 완전 연결 레이어 생성
linear_layer = nn.Linear(10 * 10, 50)  # 100개의 특징을 50개로 변환

# 평탄화된 데이터를 레이어에 입력
output = linear_layer(flattened_data)