import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_graph_dataset,plot_all_metrics,accuracy,auc_score
from preprocess_data import preprocess_data
from sklearn.metrics import roc_auc_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import csv
import os
from models import GCN, GraphSAGE, MSMESAGE, GAT, MSMEGAT
from config import *

def disturbance_regularization(model):
    total_reg = 0
    # 你可以根据需要，选择正则化哪个层的扰动项，例如 B_k 或者 position_encoder
    for name, param in model.named_parameters():
        if 'B_k' in name:  # 假设 B_k 是需要正则化的扰动部分
            total_reg += torch.norm(param, p=2)  # L2范数
    return total_reg

def disturbance_regularization(model):
    # 定义每个扰动项的正则化强度（超参数 lambda）
    reg_config = {
        'B_k': l1,
        'position_encoder': 0.05,
        'Δ_j': l2
    }

    total_reg_loss = 0
    for name, param in model.named_parameters():
        for key, lambda_reg in reg_config.items():
            if key in name:
                reg_loss = torch.norm(param, p=2)
                total_reg_loss += lambda_reg * reg_loss
    return total_reg_loss

def train(model, train_loader, optimizer, device, model_name):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)

        # 计算交叉熵损失
        loss = F.nll_loss(output, data.y - 1)  # 修正为从0开始

        # 只在 MSMEGAT 或 MSMESAGE 模型中启用扰动正则化
        if model_name in ['MSMEGAT', 'MSMESAGE']:
            loss += disturbance_regularization(model)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    predictions, labels = [], []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            probs = output.exp()  # 获取每个类的概率
            pred = probs.max(1)[1]  # 获取预测的类别（最大概率的索引）

            predictions.extend(probs[:, 1].cpu().numpy())  # 使用类别 1 的概率作为 AUC 的输入
            labels.extend((data.y - 1).cpu().numpy())  # 调整标签使其从 0 开始

        correct += pred.eq(data.y - 1).sum().item()  # 将标签从 1 开始调整为 0 开始

    acc = correct / len(loader.dataset)  # 计算准确率
    auc = roc_auc_score(labels, predictions)  # 计算 AUC

    return acc, auc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    result_dir = f"{result_dir_prefix}:run_{run_id}/{model_name}"
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, "metrics_log.csv")

    train_data, val_data, test_data = preprocess_data(feature_path,train_ratio=train_ratio, val_ratio=val_ratio)
    train_loader, val_loader, test_loader = load_graph_dataset(train_data, val_data, test_data, batch_size=batch_size)

    model_map = {
        'MSMEGAT': MSMEGAT,
        'MSMESAGE': MSMESAGE,
        'GCN': GCN,
        'GraphSAGE': GraphSAGE,
        'GAT': GAT
    }

    model_class = model_map[model_name]
    model = model_class(num_node_features=train_loader.dataset[0].num_node_features, num_classes=num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=scheduler_factor, patience=scheduler_patience, verbose=True)

    best_val_score, best_val_acc, best_val_auc = 0, 0, 0
    metrics_history = {'epoch': [], 'loss': [], 'val_acc': [], 'val_auc': [], 'val_f1': [], 'score': []}

    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, device, model_name)
        val_acc, val_auc = evaluate(model, val_loader, device)

        val_preds, val_targets = [], []
        for data in val_loader:
            data = data.to(device)
            with torch.no_grad():
                pred = model(data).max(1)[1]
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend((data.y - 1).cpu().numpy())
        val_f1 = f1_score(val_targets, val_preds)
        val_score = alpha * val_acc + (1 - alpha) * val_auc

        scheduler.step(val_score)

        if val_score > best_val_score:
            best_val_score = val_score
            best_val_acc = val_acc
            best_val_auc = val_auc
            torch.save(model.state_dict(), f'{result_dir}/best_model.pth')

        metrics_history['epoch'].append(epoch)
        metrics_history['loss'].append(train_loss)
        metrics_history['val_acc'].append(val_acc)
        metrics_history['val_auc'].append(val_auc)
        metrics_history['val_f1'].append(val_f1)
        metrics_history['score'].append(val_score)

        print(f"[model {model_name}{seed} Epoch {epoch:03d}] Loss: {train_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}, F1: {val_f1:.4f}, Score: {val_score:.4f}")

    # 测试阶段
    model.load_state_dict(torch.load(f'{result_dir}/best_model.pth'))
    test_acc, test_auc = evaluate(model, test_loader, device)
    print(f'\nBest Val - Acc: {best_val_acc:.4f}, AUC: {best_val_auc:.4f}, Score: {best_val_score:.4f}')
    print(f'Test - Acc: {test_acc:.4f}, AUC: {test_auc:.4f}')
    #with torch.no_grad(): 
        # 确保数据格式正确
     #   if isinstance(val_data, tuple) and len(val_data) > 1:
      #      analysis_data = val_data[1]
       # else:
        #    analysis_data = next(iter(val_loader))  # 使用验证加载器的第一个批次
        #analysis_hook(model, analysis_data, device, step="best")

    # 保存日志
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'val_acc', 'val_auc', 'val_f1', 'score'])
        for i in range(len(metrics_history['epoch'])):
            writer.writerow([
                metrics_history['epoch'][i],
                metrics_history['loss'][i],
                metrics_history['val_acc'][i],
                metrics_history['val_auc'][i],
                metrics_history['val_f1'][i],
                metrics_history['score'][i]
            ])

    plot_all_metrics(metrics_history, result_dir)

if __name__ == '__main__':
    main()
