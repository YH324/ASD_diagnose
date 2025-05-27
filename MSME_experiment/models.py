import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GATConv, SAGPooling, GlobalAttention
from torch_geometric.nn import GATConv, SAGEConv, GATv2Conv
from torch_geometric.nn.pool import TopKPooling
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn.aggr import SortAggregation as SortAggr
from message_kernel import compute_propagation_kernel, propagate_message
from torch_sparse import SparseTensor
from torch_geometric.nn import GraphNorm
import numpy as np
import random
import os
from config import seed

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  
torch.cuda.manual_seed_all(seed)  
os.environ['PYTHONHASHSEED'] = str(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
    
# MSMESAGE (基于GraphSAGE的改造)
class MSMESAGE(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=64, k=70, num_anatomical_classes=6):
        super(MSMESAGE, self).__init__()
        self.k = k
        self.hidden_channels = hidden_channels
        self.num_anatomical_classes = num_anatomical_classes

        # GraphSAGE卷积层
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.norm1 = GraphNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.norm2 = GraphNorm(hidden_channels)

        # 系统核 B_k：每个解剖类别对应一个结构编码核
        self.B_k = torch.nn.Embedding(num_anatomical_classes, hidden_channels * hidden_channels)

        # 位置编码器
        self.position_encoder = torch.nn.Linear(3, hidden_channels)

        # Top-K 池化层，与MSMEGAT保持一致
        self.pool = TopKPooling(hidden_channels, ratio=0.7)

        # 分类器，与MSMEGAT保持一致
        self.lin1 = torch.nn.Linear(hidden_channels * 3, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        device = next(self.parameters()).device

        # 输入数据转移至设备
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)
        batch = data.batch.to(device)
        coords = data.coords.to(device)
        anatomical_class = data.anatomical_class.to(device)

        self.pool = self.pool.to(device)
        
        # 第一层 - 与MSMEGAT相似的结构
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_attr.squeeze()).to(device)

        # 系统核 B_k
        B_k_raw = self.B_k(anatomical_class)
        B_k_matrix = B_k_raw.view(-1, self.hidden_channels, self.hidden_channels)

        # 位置扰动 Δ_j
        e_j = self.position_encoder(coords)
        Ae_j = adj @ e_j  # 与MSMEGAT一致，使用邻接矩阵
        Ae_j = Ae_j.unsqueeze(1)
        Δ_j = torch.bmm(Ae_j, B_k_matrix.transpose(1, 2)).squeeze(1)
        Δ_j_expanded = Δ_j.unsqueeze(2).expand(-1, -1, self.hidden_channels)

        # 计算传播核
        W_j = compute_propagation_kernel(B_k_matrix, Δ_j_expanded)

        
        
        # 第一层SAGEConv卷积
        x = self.conv1(x, edge_index)
        x = self.norm1(x, batch)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # 添加结构扰动
        x = x + Δ_j

        # 消息传递
        x = propagate_message(x, W_j)

        # Top-K 池化
        x, edge_index, _, batch, perm, _ = self.pool(x, edge_index, None, batch)
        
        # 第二层 - 与MSMEGAT保持一致
        edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_attr).to(device)
        anatomical_class = anatomical_class[perm]
        coords = coords[perm]

        # 重新构建 B_k 和 Δ_j
        B_k_raw = self.B_k(anatomical_class)
        B_k_matrix = B_k_raw.view(-1, self.hidden_channels, self.hidden_channels)

        e_j = self.position_encoder(coords)
        Ae_j = adj @ e_j
        Ae_j = Ae_j.unsqueeze(1)
        Δ_j = torch.bmm(Ae_j, B_k_matrix.transpose(1, 2)).squeeze(1)
        Δ_j_expanded = Δ_j.unsqueeze(2).expand(-1, -1, self.hidden_channels)
        W_j = compute_propagation_kernel(B_k_matrix, Δ_j_expanded)

        # 添加结构扰动并传播消息
        x = x + Δ_j
        x = propagate_message(x, W_j)

        # 第二层SAGEConv卷积
        x = self.conv2(x, edge_index)
        x = self.norm2(x, batch)
        x = F.elu(x)

        # 多尺度特征融合 - 与MSMEGAT完全一致
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x = torch.cat([x_mean, x_max, x_sum], dim=1)

        # 分类器
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)

# MSMEGAT (基于GATv2的改造)
class MSMEGAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=64, heads=4, k=70, num_anatomical_classes=6):
        super(MSMEGAT, self).__init__()
        self.k = k
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.num_anatomical_classes = num_anatomical_classes

        # GATv2 卷积层
        self.conv1 = GATv2Conv(num_node_features, hidden_channels // heads, heads=heads, dropout=0.5)
        self.norm1 = GraphNorm(hidden_channels)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=1, dropout=0.5)
        self.norm2 = GraphNorm(hidden_channels)
        #self.conv1 = GATConv(num_node_features, hidden_channels // heads, heads=heads,dropout=0.5)
        #self.conv2 = GATConv(hidden_channels, hidden_channels, heads=1,dropout=0.5)

        # 系统核 B_k：每个解剖类别对应一个结构编码核
        self.B_k = torch.nn.Embedding(num_anatomical_classes, hidden_channels * hidden_channels)

        # 位置编码器 ψ(p_j)
        self.position_encoder = torch.nn.Linear(3, hidden_channels)

        # Top-K 池化层
        self.pool = TopKPooling(hidden_channels, ratio=0.7)

        # 分类器
        self.lin1 = torch.nn.Linear(hidden_channels *3, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        device = next(self.parameters()).device

        # 输入数据转移至设备
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)
        batch = data.batch.to(device)
        coords = data.coords.to(device)
        anatomical_class = data.anatomical_class.to(device)

        self.pool = self.pool.to(device)
        
        # 第一层--------------------
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_attr.squeeze()).to(device)

        # 系统核 B_k：获取每个节点对应的结构核矩阵
        B_k_raw = self.B_k(anatomical_class)  # [N, H*H]
        B_k_matrix = B_k_raw.view(-1, self.hidden_channels, self.hidden_channels)  # [N, H, H]

        # 位置扰动 Δ_j = A ψ(p_j) @ B_k^T
        e_j = self.position_encoder(coords)  # [N, H]
        Ae_j = adj @ e_j # [N, H]
        Ae_j = Ae_j.unsqueeze(1)  # [N, 1, H]
        Δ_j = torch.bmm(Ae_j, B_k_matrix.transpose(1, 2)).squeeze(1)  # [N, H]
        Δ_j_expanded = Δ_j.unsqueeze(2).expand(-1, -1, self.hidden_channels)  # [N, H, H]

        # 使用传播核计算 W_j = B_k + Δ_j
        W_j = compute_propagation_kernel(B_k_matrix, Δ_j_expanded)  # 计算传播核 W_j

        # 第一层 GATv2 卷积
        x = self.conv1(x, edge_index)
        x = self.norm1(x, batch) 
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # 添加结构扰动
        x = x + Δ_j

        # 消息传递使用 W_j
        x = propagate_message(x, W_j)  # 传播核

        # Top-K 池化
        x, edge_index, _, batch, perm, _ = self.pool(x, edge_index, None, batch)
        
        # 第二层---------------------
        edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_attr).to(device)
        anatomical_class = anatomical_class[perm]  # 只保留 TopK 选中的节点
        coords = coords[perm]  # 同上

        # 重新构建 B_k 和 Δ_j
        B_k_raw = self.B_k(anatomical_class)  # [N', H*H]
        B_k_matrix = B_k_raw.view(-1, self.hidden_channels, self.hidden_channels)  # [N', H, H]

        e_j = self.position_encoder(coords)  # [N', H]
        
        Ae_j = adj @ e_j  # [N', H]
        Ae_j = Ae_j.unsqueeze(1)  # [N', 1, H]
        Δ_j = torch.bmm(Ae_j, B_k_matrix.transpose(1, 2)).squeeze(1)  # [N', H]
        Δ_j_expanded = Δ_j.unsqueeze(2).expand(-1, -1, self.hidden_channels)  # [N', H, H]
        W_j = compute_propagation_kernel(B_k_matrix, Δ_j_expanded)

        # 使用新传播核进行消息传递
        x = x + Δ_j  # 添加结构扰动
        x = propagate_message(x, W_j)

        # 第二层 GATv2 卷积
        x = self.conv2(x, edge_index)
        x = self.norm2(x, batch)
        x = F.elu(x)

        # mean-max 池化聚合
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)

        x = torch.cat([x_mean, x_max, x_sum], dim=1)
        # x = torch.cat([x_max, x_sum], dim=1)

        # 分类器
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=16):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # 两个分类器，与MSMEGAT保持一致
        self.lin1 = torch.nn.Linear(hidden_channels * 3, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 第一层卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # 第二层卷积
        x = self.conv2(x, edge_index)
        
        # 多尺度特征融合 - 与MSMEGAT一致
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x = torch.cat([x_mean, x_max, x_sum], dim=1)
        
        # 与MSMEGAT一致的分类器
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=1)

# 修改GraphSAGE，使用与MSMEGAT相同的读出设置
class GraphSAGE(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=16):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
        # 两个分类器，与MSMEGAT保持一致
        self.lin1 = torch.nn.Linear(hidden_channels * 3, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 第一层卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # 第二层卷积
        x = self.conv2(x, edge_index)
        
        # 多尺度特征融合 - 与MSMEGAT一致
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x = torch.cat([x_mean, x_max, x_sum], dim=1)
        
        # 与MSMEGAT一致的分类器
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=1)

# 修改GAT，使用与MSMEGAT相同的读出设置
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=16, heads=4, k=70):
        super(GAT, self).__init__()
        self.hidden_channels = hidden_channels

        self.conv1 = GATConv(num_node_features, hidden_channels // heads, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=1)
        
        # 两个分类器，与MSMEGAT保持一致
        self.lin1 = torch.nn.Linear(hidden_channels * 3, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 第一层卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # 第二层卷积
        x = self.conv2(x, edge_index)
        
        # 多尺度特征融合 - 与MSMEGAT一致
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x = torch.cat([x_mean, x_max, x_sum], dim=1)
        
        # 与MSMEGAT一致的分类器
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)