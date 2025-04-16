import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from config import (
    INFO_CSV, NODE_FEATURE_CSV, FC_TEMPLATE,
    TARGET_LIST, NUM_ROI, ROI_FEATURE_DIM
)

def load_node_static_features():
    """加载静态节点特征（xyz 坐标）"""
    node_df = pd.read_csv(NODE_FEATURE_CSV)
    Y_xyz = node_df[['x(mm)', 'y(mm)', 'z(mm)']].values.astype(float)  # shape: (160, 3)

    scaler = StandardScaler()
    Y_std = scaler.fit_transform(Y_xyz)
    
    # 加入subnet encoding
    subnet_onehot = pd.get_dummies(node_df['Subnetwork'])  # shape: (160, 6)
    Y_enriched = np.concatenate([Y_std, subnet_onehot.values], axis=1)  # shape: (160, 9)

    return Y_enriched  # shape: (160, 3)

def load_subjects():
    """读取行为数据、筛选有目标变量的被试"""
    df = pd.read_csv(INFO_CSV)
    
    # 只保留目标列和 ID
    cols = ['ID'] + TARGET_LIST
    df = df[cols].dropna().reset_index(drop=True)
    
    # 添加 FC 路径
    df['FC_path'] = df['ID'].apply(lambda x: FC_TEMPLATE(x))
    
    return df

def load_all_samples(Y_enriched):
    """
    构造所有被试的样本数据：
    - 加载每个被试的 FC 矩阵
    - 构造 Z = A @ Y_enriched
    - 输出结构化样本列表
    """
    info_df = load_subjects()
    sample_list = []

    for _, row in info_df.iterrows():
        sub_id = row['ID']
        fc_path = row['FC_path']

        try:
            A = np.loadtxt(fc_path)
            if A.shape != (NUM_ROI, NUM_ROI):
                print(f"[跳过] ID={sub_id}，邻接矩阵尺寸错误：{A.shape}")
                continue
            
            # 构造节点传播特征 Z = A @ Y
            Z = A @ Y_enriched  # shape: (160, d)

            # 构造样本 dict
            sample = {
                'ID': sub_id,
                'adj': A,
                'Z': Z,  # 传播后的节点特征
                'target': row[TARGET_LIST].values.astype(np.float32)  # shape: (T,)
            }
            sample_list.append(sample)

        except Exception as e:
            print(f"[跳过] ID={sub_id}，读取失败：{e}")

    return sample_list

def load_processed_dataset(path):
    """
    读取预处理数据集，格式为 list[dict]，每个 dict 包含：
    - 'Z': 节点特征 (160, D)
    - 'adj': 邻接矩阵 (160, 160)
    - 'target': 回归目标 (float/int)
    """
    return torch.load(path)
