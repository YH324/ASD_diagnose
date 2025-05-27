# message_kernel.py
import torch

# 计算传播核 W_j = B_k + Δ_j
def compute_propagation_kernel(B_k, delta_j):
    assert B_k.size(1) == delta_j.size(1), f"Dimension mismatch: {B_k.size(1)} vs {delta_j.size(1)}"
    return B_k + delta_j
    #return B_k
    #return delta_j

# 使用传播核进行消息传递 m_ij = W_j * h_j
def propagate_message(h_j, W_j):
    return torch.bmm(W_j, h_j.unsqueeze(2)).squeeze(2)
