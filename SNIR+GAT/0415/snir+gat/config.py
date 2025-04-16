# config.py

import numpy as np
import os

# ==== 🗂️ 数据路径配置 ====
DATA_ROOT = '../hx/'
INFO_CSV = os.path.join(DATA_ROOT, '../hxinfo.csv')
NODE_FEATURE_CSV = '../x_dosenbach.csv'  # ROI 坐标信息
FC_TEMPLATE = lambda ID: os.path.join(DATA_ROOT, f'z{ID}.txt')

RHO_ESTIMATE_PATH = '../rho_estimate.npy'  # Lasso 估计的 rho 存储路径
DATASET_PATH = 'data/processed_dataset.pt'
# ==== 🎯 任务目标 ====
TARGET_LIST = ['SRS_mannerisms']  # 多任务时支持多个

# ==== 🧠 脑图参数 ====
NUM_ROI = 160          # ROI 节点总数
ROI_FEATURE_DIM = 9    # 每个 ROI 的特征维度（如 xyz + others）

# ==== 📉 rho 剪枝参数 ====
RHO_THRESHOLD = 1e-4   # rho 的裁剪阈值（节点裁剪）

# ==== 🔧 GAT 模型结构参数 ====
GAT_HIDDEN_DIM = 64
GAT_HEADS = 4
GAT_DROPOUT = 0.6
GAT_OUT_DIM = len(TARGET_LIST)  # 多任务输出数
# 模型使用调整
USE_RHO_ADJUSTED = True  # 若为 True 则使用 RhoAdjustedGAT


# ==== 🧪 训练参数 ====
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
RANDOM_SEED = 42
MAX_EPOCHS = 300
PATIENCE = 20
LEARNING_RATE = 1e-3
BATCH_SIZE = 1  # 每个图是一个样本，所以 mini-batch=1 是常见写法

# ==== 📊 评估 ====
PRIMARY_METRIC = 'r2'  # 可选: 'r2', 'pearson', 'mae'
