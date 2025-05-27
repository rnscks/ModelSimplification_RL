import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# 노드 특성 (3개 노드, 각 노드에 2차원 특성)
x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)

# 엣지 인덱스 (양방향 엣지 2개)
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)

# 엣지 특성 (4개의 엣지에 대해 2차원 특성)
edge_attr = torch.tensor([[0.1],
                          [0.3],
                          [0.5],
                          [0.7]], dtype=torch.float)

# 데이터 객체 생성
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# GCNConv 레이어 정의
conv = GCNConv(in_channels=2, out_channels=2)

# forward 패스 시도
out = conv(data.x, data.edge_index, data.edge_attr)
print(out)
