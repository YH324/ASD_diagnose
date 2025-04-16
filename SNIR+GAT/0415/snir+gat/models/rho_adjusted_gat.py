import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from config import GAT_HIDDEN_DIM, GAT_HEADS, GAT_OUT_DIM, GAT_DROPOUT

class RhoAdjustedGAT(nn.Module):
    def __init__(self, in_dim, rho=None):
        super(RhoAdjustedGAT, self).__init__()
        self.rho = rho  # shape: (num_nodes,) or None

        adjusted_in_dim = in_dim + 1 if rho is not None else in_dim

        self.gat1 = GATConv(adjusted_in_dim, GAT_HIDDEN_DIM, heads=GAT_HEADS, dropout=GAT_DROPOUT)

        self.gat2 = GATConv(GAT_HIDDEN_DIM * GAT_HEADS, GAT_HIDDEN_DIM, heads=1, dropout=GAT_DROPOUT)

        self.dropout = nn.Dropout(GAT_DROPOUT)
        self.fc = nn.Linear(GAT_HIDDEN_DIM, GAT_OUT_DIM)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # x: [N, in_dim]

        # 如果传入了 rho，作为节点特征加偏置
        if self.rho is not None:
            # rho_feat = self.rho[data.node_idx]  # shape: [N,]
            rho_feat = self.rho[data.node_idx.to(self.rho.device)].unsqueeze(1)  # [N,1]
            rho_feat = rho_feat.to(x.device)  # ✅ 保证和 x 同设备
            x = torch.cat([x, rho_feat], dim=1)

        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.gat2(x, edge_index)
        x = F.elu(x)

        x = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]
        out = self.fc(x)
        return out
