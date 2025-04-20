import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_data, accuracy, auc_score, plot_all_metrics
from preprocess_data import preprocess_data
import time
import argparse
from sklearn.metrics import roc_auc_score
from models import GCN, GraphSAGE, DeepGraphSAGE, GATv2Net
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import csv
import os

do_ablation = True
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    predictions = []
    labels = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            pred = output.max(1)[1]
            predictions.extend(output[:, 1].cpu().numpy())
            labels.extend(data.y.cpu().numpy())
        correct += pred.eq(data.y).sum().item()
    acc = correct / len(loader.dataset)
    auc = roc_auc_score(labels, predictions)
    return acc, auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.3, help='Ratio of validation data')
    parser.add_argument('--model', type=str, default='GATv2Net',
                    choices=['GCN', 'GraphSAGE', 'DeepGraphSAGE', 'GATv2Net'],
                    help='Model to use')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据路径
    feature_path = '../Y.npy'
    adj_dir = '../hx/'
    info_csv = '../hxinfo.csv'
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    result_dir = f"results/classification:run_{run_id}/{args.model}"
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, "metrics_log.csv")

    # === 添加日志记录数据结构 ===
    metrics_history = {
        'epoch': [],
        'loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_f1': [],
        'score': [],
    }

    train_data, val_data, test_data = preprocess_data(feature_path, adj_dir, info_csv, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    # 加载数据
    train_loader, val_loader, test_loader = load_data(train_data, val_data, test_data)

    # 选择模型
    if args.model == 'GCN':
        model_class = GCN
    elif args.model == 'GraphSAGE':
        model_class = GraphSAGE
    elif args.model == 'DeepGraphSAGE':
        model_class = DeepGraphSAGE
    elif args.model == 'GATv2Net':
        model_class = GATv2Net
    else:
        raise ValueError(f"Unknown model type: {args.model}")


    # 初始化模型
    model = model_class(num_node_features=train_loader.dataset[0].num_node_features, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)


    # 训练模型
    best_val_score = 0
    best_val_acc = 0
    best_val_auc = 0
    alpha = 0.5  # acc 与 auc 权重比例，可调

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train(model, train_loader, optimizer, device)
        val_acc, val_auc = evaluate(model, val_loader, device)
        val_pred_labels = []
        val_true_labels = []
        for data in val_loader:
            data = data.to(device)
            with torch.no_grad():
                output = model(data)
                pred = output.max(1)[1]
                val_pred_labels.extend(pred.cpu().numpy())
                val_true_labels.extend(data.y.cpu().numpy())
        val_f1 = f1_score(val_true_labels, val_pred_labels)
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

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch:03d}, LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, '
              f'Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, Score: {val_score:.4f}, Time: {time.time() - start:.4f}s')

    # 测试模型
    model.load_state_dict(torch.load(f'{result_dir}/best_model.pth'))
    test_acc, test_auc = evaluate(model, test_loader, device)
    print(f'BEST val ACC:{best_val_acc}, BEST val AUC:{best_val_auc}, COMBINED SCORE: {best_val_score:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}')

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
                metrics_history['score'][i],
            ])
    plot_all_metrics(metrics_history, result_dir)
        
    # === Ablation Analysis ===
    def ablation_analysis(model, data_sample, device):
        model.eval()
        data_sample = data_sample.to(device)
        baseline_output = model(data_sample)
        baseline_prob = baseline_output[0, 1].item()

        importances = []
        for i in range(data_sample.x.shape[0]):
            x_copy = data_sample.x.clone()
            x_copy[i] = 0  # 抹除该节点特征
            data_copy = data_sample.clone()
            data_copy.x = x_copy
            with torch.no_grad():
                output = model(data_copy)
                prob = output[0, 1].item()
            delta = baseline_prob - prob
            importances.append(delta)

        return importances

    if do_ablation:
        ablation_data = test_loader.dataset[0]  # 可以换 index
        importance_scores = ablation_analysis(model, ablation_data, device)

        top_k = 20
        top_nodes = sorted(enumerate(importance_scores), key=lambda x: -abs(x[1]))[:top_k]
        print(f"\nTop {top_k} Important Nodes (based on Ablation):")
        for idx, score in top_nodes:
            print(f"Node {idx}: Δ Prob = {score:.4f}")

if __name__ == '__main__':
    main()
