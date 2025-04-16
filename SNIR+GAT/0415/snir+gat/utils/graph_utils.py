import torch
import numpy as np

def build_edge_index(adj_matrix, selected_idx=None, threshold=0.0):
    """
    将邻接矩阵转换为 edge_index（PyG 格式）
    - adj_matrix: np.array of shape (160, 160)
    - selected_idx: 可选，保留节点的索引（e.g., rho 裁剪后）
    - threshold: 连边强度的下限（小于该值不连边）

    返回：
    - edge_index: LongTensor of shape [2, num_edges]
    """
    if selected_idx is not None:
        # 仅保留指定 ROI（rho 裁剪后）
        adj_matrix = adj_matrix[np.ix_(selected_idx, selected_idx)]

    edge_list = []
    N = adj_matrix.shape[0]

    for i in range(N):
        for j in range(N):
            if i != j and abs(adj_matrix[i, j]) > threshold:
                edge_list.append([i, j])  # 有向边：i → j

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index  # shape: [2, num_edges]
