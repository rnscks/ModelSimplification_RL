import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, features, adjacency):
        # 인접 행렬을 사용하여 특징을 집계
        aggregated_features = torch.mm(adjacency, features)
        # 선형 변환 적용
        transformed_features = self.linear(aggregated_features)
        # ReLU 활성화 함수 적용
        return F.relu(transformed_features)

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(in_features, hidden_features)
        self.layer2 = GCNLayer(hidden_features, out_features)

    def forward(self, features, adjacency):
        x = self.layer1(features, adjacency)
        x = self.layer2(x, adjacency)
        return x
