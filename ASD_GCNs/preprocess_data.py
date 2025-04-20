import os
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

threshold = 0.05  

def load_data(feature_path, adj_dir, info_csv):
    features = np.load(feature_path, allow_pickle=True)
    features = np.array(features)

    info_df = pd.read_csv(info_csv)

    graph_data_list = []
    valid_subject_ids = []

    for i, row in info_df.iterrows():
        subject_id = row['ID']
        label = 1 if 'ASD' in subject_id else 0

        adj_path = os.path.join(adj_dir, f"z{subject_id}.txt")
        if not os.path.exists(adj_path):
            print(f"[警告] 邻接矩阵文件不存在: {adj_path}，跳过该样本。")
            continue

        try:
            adj_matrix = np.loadtxt(adj_path)
        except Exception as e:
            print(f"[错误] 加载邻接矩阵失败: {adj_path}，错误信息: {e}，跳过该样本。")
            continue

        try:
            x = torch.tensor(features[i], dtype=torch.float)  # shape: [num_nodes, num_features]
        except Exception as e:
            print(f"[错误] 加载特征失败: index={i}, 错误信息: {e}，跳过该样本。")
            continue

        # 只保留达到阈值的边并提取边权
        adj_matrix = np.triu(adj_matrix, k=0)  # 去掉重复边（保留自环）
        mask = np.abs(adj_matrix) > threshold
        row, col = np.where(mask)

        if len(row) == 0:
            print(f"[警告] 所有边都被过滤掉了: {adj_path}，跳过该样本。")
            continue

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(adj_matrix[row, col], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label], dtype=torch.long))
        graph_data_list.append(data)
        valid_subject_ids.append(subject_id)

    return graph_data_list, valid_subject_ids

def preprocess_data(feature_path, adj_dir, info_csv, train_ratio=0.6, val_ratio=0.2):
    graph_data_list, subject_ids = load_data(feature_path, adj_dir, info_csv)

    labels = [data.y.item() for data in graph_data_list]
    print(f"加载成功样本数: {len(graph_data_list)}，类别分布: {np.bincount(labels)}")

    test_ratio = 1 - train_ratio - val_ratio
    gss = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
    train_val_idx, test_idx = next(gss.split(graph_data_list, groups=subject_ids))

    train_val_data = [graph_data_list[i] for i in train_val_idx]
    train_val_subjects = [subject_ids[i] for i in train_val_idx]

    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio_adjusted, random_state=42)
    train_idx, val_idx = next(gss_val.split(train_val_data, groups=train_val_subjects))

    train_data = [train_val_data[i] for i in train_idx]
    val_data = [train_val_data[i] for i in val_idx]
    test_data = [graph_data_list[i] for i in test_idx]

    return train_data, val_data, test_data
