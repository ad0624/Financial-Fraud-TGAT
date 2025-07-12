import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class TemporalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(1, dim)

    def forward(self, timestamps):
        return torch.sin(self.lin(timestamps))

class TemporalGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__(aggr='add')  # 使用加法聚合
        # 注意力层的输入维度：源节点特征 + 目标节点特征 + 时间编码（均为out_channels）
        attn_input_dim = 2 * out_channels + time_dim
        self.attn = nn.Linear(attn_input_dim, 1)
        self.lin = nn.Linear(in_channels, out_channels)
        self.temporal = TemporalEncoding(time_dim)

    def forward(self, x, edge_index, edge_time):
        t_enc = self.temporal(edge_time)
        return self.propagate(edge_index, x=x, t_enc=t_enc)

    def message(self, x_i, x_j, t_enc, index):
        x_j = self.lin(x_j)
        x_i = self.lin(x_i)
        alpha_input = torch.cat([x_i, x_j, t_enc], dim=1)
        alpha = F.leaky_relu(self.attn(alpha_input))
        alpha = softmax(alpha, index)
        return x_j * alpha

class TGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, time_dim=16):
        super().__init__()
        self.conv1 = TemporalGATConv(in_dim, hidden_dim, time_dim)
        self.conv2 = TemporalGATConv(hidden_dim, hidden_dim, time_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_time):
        x = F.relu(self.conv1(x, edge_index, edge_time))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index, edge_time)
        return self.classifier(x)
