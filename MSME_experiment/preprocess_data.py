import os
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import pandas as pd

threshold = 0.1  # 边阈值

def load_data(feature_path, threshold=threshold, drop_edges=False, drop_prob=0.1,
              add_feat_noise=False, perturb_edge_weight=False, augment_times=1):
    data = np.load(feature_path, allow_pickle=True).item()
    X = data['X']  # 特征矩阵 [n, 90, 90]
    A = data['A']  # 邻接矩阵 [n, 90, 90]
    y = data['y']  # 标签数组 [n]
    template = data['template']  # aal90模板信息

    if 'Anatomical_classification' in template.columns:
        # 在转换为编码前，保存类别名称和编码的映射关系
        anatomical_categories = pd.Categorical(template['Anatomical_classification'])
        anatomical_mapping = {i: name for i, name in enumerate(anatomical_categories.categories)}
        template['Anatomical_classification'] = anatomical_categories.codes
        
        # 输出解剖类别名称和对应的编码
        print("解剖类别映射关系:")
        for code, name in anatomical_mapping.items():
            print(f"编码 {code}: {name}")
    else:
        raise ValueError("模板数据中缺少 'Anatomical_classification' 列")

    graph_data_list = []

    for i in range(len(y)):
        label = y[i]
        feature_matrix = X[i]
        adj_matrix = A[i]
        coords = template[['x', 'y', 'z']].values
        anatomical_class = template['Anatomical_classification'].values

        for _ in range(augment_times):  # 每个图复制 augment_times 次
            # 创建边的阈值
            adj_matrix_aug = np.triu(adj_matrix.copy(), k=0)
            mask = np.abs(adj_matrix_aug) > threshold
            row, col = np.where(mask)

            if len(row) == 0:
                print(f"[警告] 样本 {i} 的增强图中所有边都被过滤，跳过。")
                continue

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_attr = torch.tensor(adj_matrix_aug[row, col], dtype=torch.float)

            if perturb_edge_weight:
                edge_attr = edge_attr + (torch.rand_like(edge_attr) - 0.5) * 0.01

            if drop_edges:
                num_edges = edge_index.size(1)
                keep_mask = torch.rand(num_edges) > drop_prob
                edge_index = edge_index[:, keep_mask]
                edge_attr = edge_attr[keep_mask]

            feat_aug = torch.tensor(feature_matrix.copy(), dtype=torch.float)
            if add_feat_noise:
                noise = torch.randn_like(feat_aug) * 0.01
                feat_aug = feat_aug + noise

            data_obj = Data(
                x=feat_aug,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([label], dtype=torch.long),
                coords=torch.tensor(coords, dtype=torch.float),
                anatomical_class=torch.tensor(anatomical_class, dtype=torch.long)
            )

            graph_data_list.append(data_obj)

    return graph_data_list

def preprocess_data(feature_path, train_ratio=0.7, val_ratio=0.15, augment_times=1):
    graph_data_list = load_data(
        feature_path,
        drop_edges=False,
        drop_prob=0.1,
        add_feat_noise=False,
        perturb_edge_weight=False,
        augment_times=augment_times  # 控制每个图增强几次
    )

    labels = [data.y.item()-1 for data in graph_data_list]
    print(f"加载并增强后的图样本数: {len(graph_data_list)}，类别分布: {np.bincount(labels)}")

    test_ratio = 1 - train_ratio - val_ratio

    # train + val vs test
    train_val_data, test_data = train_test_split(
        graph_data_list, 
        test_size=test_ratio, 
        stratify=labels, 
        random_state=100
    )

    # train vs val
    train_labels = [data.y.item() for data in train_val_data]
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_ratio_adjusted,
        stratify=train_labels,
        random_state=100
    )

    return train_data, val_data, test_data
