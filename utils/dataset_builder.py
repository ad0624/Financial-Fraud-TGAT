import torch
from torch_geometric.data import Data

def build_synthetic_graph():
    # 节点特征
    x = torch.rand((100, 16))  # 100 个节点，每个 16 维特征

    # 边连接 (随机)
    edge_index = torch.randint(0, 100, (2, 500))  # 500 条边

    # 边上的时间戳（模拟）
    edge_time = torch.rand((500, 1))

    # 节点标签（二分类：欺诈 or 非欺诈）
    y = torch.randint(0, 2, (100,))

    return Data(x=x, edge_index=edge_index, edge_time=edge_time, y=y)
