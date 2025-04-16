import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
from datetime import datetime
from config import *
from utils.feature_select import load_rho_mask, apply_pruning_to_dataset
from utils.graph_utils import build_edge_index
from models.vanilla_gat import VanillaGAT
from models.rho_adjusted_gat import RhoAdjustedGAT  
from train.evaluate import evaluate, plot_predictions,log_metrics,plot_training_curves
import numpy as np
import os 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = LEARNING_RATE
WEIGHT_DECAY = 0
EPOCHS = MAX_EPOCHS

def build_dataset(raw_dataset, selected_idx):
    """
    输入：原始 dataset 列表，输出 PyG Data 列表
    """
    pyg_data_list = []

    for sample in raw_dataset:
        Z_pruned = sample['Z_pruned']        # shape: (m, 9)
        adj = sample['adj']
        target = sample['target']            # scalar 或向量

        x = torch.tensor(Z_pruned, dtype=torch.float)
        edge_index = build_edge_index(adj, selected_idx=selected_idx)
        y = torch.tensor(target, dtype=torch.float).unsqueeze(0)  # shape: [1]

        data = Data(x=x, edge_index=edge_index, y=y)
        data.node_idx = torch.tensor(selected_idx, dtype=torch.long)

        pyg_data_list.append(data)

    return pyg_data_list


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(DEVICE)
        data.node_idx = data.node_idx.to(DEVICE)  # 保证在 GPU 上

        optimizer.zero_grad()

        out = model(data)       # shape: [batch_size, 1]
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


def main():
    # 1. 加载原始数据（你需要提供）
    raw_dataset = torch.load(DATASET_PATH)  # 每个 sample 包含 'Z', 'adj', 'target'

    # 2. rho 筛选节点 + 构建 Z_pruned
    important_mask, selected_idx = load_rho_mask()
    raw_dataset = apply_pruning_to_dataset(raw_dataset, selected_idx)

    # 3. 构造 PyG 格式数据
    data_list = build_dataset(raw_dataset, selected_idx)
    loader = DataLoader(data_list, batch_size=BATCH_SIZE, shuffle=True)

    # 4. 初始化模型
    in_dim = data_list[0].x.shape[1]
    # model = VanillaGAT(in_dim).to(DEVICE)
    # model = RhoAdjustedGAT(in_dim, rho=load_rho_tensor()).to(DEVICE)  # 可替换
    if USE_RHO_ADJUSTED:
        rho = np.load(RHO_ESTIMATE_PATH)
        model = RhoAdjustedGAT(in_dim, rho=torch.tensor(rho, dtype=torch.float32)).to(DEVICE)
    else:
        model = VanillaGAT(in_dim).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    # 5. 训练
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"results/run_{run_id}"
    os.makedirs(result_dir, exist_ok=True)

    log_path=f"{result_dir}/metrics_log.csv"
    save_path1=f"{result_dir}/figures/pred_vs_target.png"
    save_path2=f"{result_dir}/figures/training_curves.png"
    
    # 初始化最优指标
    best_mse = float('inf')
    best_checkpoint = None
    best_preds, best_targets = None, None

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, loader, optimizer, criterion)
        mse, r2, preds, targets = evaluate(model, loader, DEVICE)

        print(f"[Epoch {epoch+1}] Train Loss: {loss:.4f} | Val MSE: {mse:.4f} | R2: {r2:.4f}")
        log_metrics(epoch + 1, loss, mse, r2, log_path)

        if mse < best_mse:
            best_mse = mse
            best_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'mse': mse,
                'r2': r2,
                'config': {
                    'learning_rate': LR,
                    'batch_size': BATCH_SIZE,
                    'max_epochs': EPOCHS,
                    'use_rho_adjusted': USE_RHO_ADJUSTED,
                    'selected_idx': selected_idx,
                    'input_dim': in_dim
                }
            }
            best_preds, best_targets = preds, targets

    # 保存最优模型
    checkpoint_path = f'{result_dir}/checkpoints'
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save(best_checkpoint, f'{checkpoint_path}/best_checkpoint.pt')
    print(f"Best checkpoint saved with MSE: {best_mse:.4f}")

    # 保存最佳 epoch 信息
    with open(os.path.join(result_dir, "best_epoch.txt"), "w") as f:
        f.write(f"Best Epoch: {best_checkpoint['epoch']}\n")
        f.write(f"Best MSE: {best_checkpoint['mse']:.6f}\n")
        f.write(f"Best R²: {best_checkpoint['r2']:.6f}\n")
    
    # 可视化（使用最优预测）
    plot_predictions(best_preds, best_targets, save_path1)
    plot_training_curves(log_path, save_path2)

if __name__ == "__main__":
    main()
