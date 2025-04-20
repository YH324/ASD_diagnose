import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATConv, SAGPooling, GlobalAttention
import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=16):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=16):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class DeepGraphSAGE(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=64):
        super().__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)

        self.pool = SAGPooling(hidden_channels, ratio=0.5)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index)) + x1
        x3 = F.relu(self.conv3(x2, edge_index)) + x2

        x, edge_index, _, batch, _, _ = self.pool(x3, edge_index, None, batch)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

class GATv2Net(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=32, heads=4):
        super(GATv2Net, self).__init__()
        
        self.edge_encoder = torch.nn.Linear(1, hidden_channels)

        self.conv1 = GATv2Conv(num_node_features + hidden_channels, hidden_channels, heads=heads, dropout=0.5)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, dropout=0.5)

        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if edge_attr is not None:
            # 编码边特征
            edge_feat = self.edge_encoder(edge_attr.unsqueeze(-1))  # [E, hidden]
            # 创建空的邻接节点表示
            edge_msg = torch.zeros(x.size(0), edge_feat.size(1), device=x.device)  # [N, hidden]
            # 将边特征聚合到目标节点（edge_index[1] 是目标节点索引）
            edge_msg = edge_msg.index_add(0, edge_index[1], edge_feat)
            # 拼接节点原始特征和边特征聚合结果
            x = torch.cat([x, edge_msg], dim=1)

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.elu(x)

        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

