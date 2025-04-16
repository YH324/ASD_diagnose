import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from config import GAT_HIDDEN_DIM, GAT_HEADS, GAT_OUT_DIM, GAT_DROPOUT

class VanillaGAT(nn.Module):
    def __init__(self, in_dim):
        super(VanillaGAT, self).__init__()
        self.gat1 = GATConv(in_dim, GAT_HIDDEN_DIM, heads=GAT_HEADS, dropout=GAT_DROPOUT)
        self.gat2 = GATConv(GAT_HIDDEN_DIM * GAT_HEADS, GAT_HIDDEN_DIM, heads=1, dropout=GAT_DROPOUT)

        self.dropout = nn.Dropout(GAT_DROPOUT)
        self.fc = nn.Linear(GAT_HIDDEN_DIM, GAT_OUT_DIM)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.gat2(x, edge_index)
        x = F.elu(x)

        # 图级 pooling
        x = global_mean_pool(x, batch)  # shape: [num_graphs, hidden_dim]

        out = self.fc(x)  # shape: [num_graphs, target_dim]
        return out
