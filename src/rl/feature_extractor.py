from typing import List, Dict
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.rl.agent.agent import GRAPH
from src.rl.point_net import get_model

class GNN_PointNetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.conv1 = GCNConv(observation_space['node'].shape[1]+64, 128)
        self.conv2 = GCNConv(128, features_dim)
        self.point_net = get_model(out_dim=64, normal_channel=False) 
        
        
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 현재 장치 확인
        device = next(self.parameters()).device
        batch_size = obs['node'].shape[0]

        # 각 배치에 대해 그래프 데이터를 처리
        data_list = []
        for i in range(batch_size):
            pointcloud = torch.as_tensor(obs['pointcloud'][i], dtype=torch.float32, device=device) 
            pointcloud[torch.isnan(pointcloud)] = 0.0  # NaN 값을 0으로 대체    
            point_features = self.point_net(pointcloud) 
            
            node = torch.as_tensor(obs['node'][i], dtype=torch.float32, device=device)
            node = torch.cat([node, point_features], dim=1)  # 노드 특성에 포인트넷 특성 추가   
            node[torch.isnan(node)] = 0.0  # NaN 값을 0으로 대체    
            edge_index = torch.as_tensor(obs['edge_index'][i], dtype=torch.long, device=device)
            edge_attr = torch.as_tensor(obs['edge_attr'][i], dtype=torch.float32, device=device)
            # 유효한 엣지만 선택
            mask = edge_index.sum(dim=1) != 0
            edge_index = edge_index[mask]
            edge_attr = edge_attr[mask]

            data = Data(x=node, edge_index=edge_index.T, edge_attr=edge_attr)
            data_list.append(data)

        # 여러 그래프를 단일 배치로 결합
        batch = Batch.from_data_list(data_list).to(device)
        
        x = F.relu(self.conv1(batch.x, batch.edge_index, batch.edge_attr))
        x = self.conv2(x, batch.edge_index, batch.edge_attr)
        
        # 노드 특성의 평균을 배치 단위로 계산
        batch_size = batch.batch.max().item() + 1
        x = torch.cat([torch.mean(x[batch.batch == i], dim=0, keepdim=True) for i in range(batch_size)], dim=0)
        return x
    
class GNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.conv1 = GCNConv(observation_space['node'].shape[1], 128)
        self.conv2 = GCNConv(128, features_dim)
        
        
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:   
        # 현재 장치 확인
        device = next(self.parameters()).device
        batch_size = obs['node'].shape[0]

        # 각 배치에 대해 그래프 데이터를 처리
        data_list = []
        for i in range(batch_size):
            node = torch.as_tensor(obs['node'][i], dtype=torch.float32, device=device)
            edge_index = torch.as_tensor(obs['edge_index'][i], dtype=torch.long, device=device)
            edge_attr = torch.as_tensor(obs['edge_attr'][i], dtype=torch.float32, device=device)
            # 유효한 엣지만 선택
            mask = edge_index.sum(dim=1) != 0
            edge_index = edge_index[mask]
            edge_attr = edge_attr[mask]

            data = Data(x=node, edge_index=edge_index.T, edge_attr=edge_attr)
            data_list.append(data)

        # 여러 그래프를 단일 배치로 결합
        batch = Batch.from_data_list(data_list).to(device)
        
        x = F.relu(self.conv1(batch.x, batch.edge_index, batch.edge_attr))
        x = self.conv2(x, batch.edge_index, batch.edge_attr)
        
        # 노드 특성의 평균을 배치 단위로 계산
        batch_size = batch.batch.max().item() + 1
        x = torch.cat([torch.mean(x[batch.batch == i], dim=0, keepdim=True) for i in range(batch_size)], dim=0)
        return x