# config.py

# 数据路径
feature_path = 'processed_brain_data_dti.npy'
info_csv = 'aal90.csv'

# 模型配置
model_name = 'MSMEGAT'
num_classes = 2
hidden_dim = 64

dropout = 0.5

# 训练配置
epochs = 200
lr = 0.0001
weight_decay = 1e-3
train_ratio = 0.7
val_ratio = 0.15
batch_size = 24
seed= 999

l1 = 0.5 #0.5
l2 = 1 #1
# 学习率调度
scheduler_factor = 0.8
scheduler_patience = 10

# 综合评分权重
alpha = 0.2

# 日志输出目录前缀（实际run_id自动生成）
result_dir_prefix = "results/classification"
