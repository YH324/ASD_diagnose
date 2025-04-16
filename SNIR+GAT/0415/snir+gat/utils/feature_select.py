import numpy as np
from config import RHO_ESTIMATE_PATH, RHO_THRESHOLD

def load_rho_mask():
    """加载 rho 并返回布尔掩码：哪些节点被保留"""
    rho = np.load(RHO_ESTIMATE_PATH)
    if rho.ndim == 1 and rho.shape[0] == 160 * 9:
        print("[INFO] rho shape is flat (1440), reshaping to (160, 9)")
        rho = rho.reshape(160, 9)

    assert rho.shape[0] == 160, f"rho shape invalid: {rho.shape}, expected (160, 9) or similar"
    
    important_mask = np.abs(rho) > RHO_THRESHOLD
    selected_idx = np.where(important_mask)[0]  # list of indices

    print(f"[特征选择] 保留 ROI 数量: {len(selected_idx)} / 160")
    return important_mask, selected_idx  # bool array, int array

def prune_node_features(Z, selected_idx):
    """
    对单个样本的节点特征 Z ∈ (160, 9)，保留选中的 ROI 节点
    返回：Z_pruned ∈ (m, 9)
    """
    return Z[selected_idx, :]

def apply_pruning_to_dataset(dataset, selected_idx):
    """
    输入：原始 dataset（每个样本含 Z），裁剪特征
    返回：Z_pruned 添加进样本
    """
    for sample in dataset:
        Z = sample['Z']  # shape: (160, 9)
        Z_pruned = prune_node_features(Z, selected_idx)  # shape: (m, 9)
        sample['Z_pruned'] = Z_pruned  # 添加新的键
    
    return dataset