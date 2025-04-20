import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(train_data, val_data, test_data, batch_size=32):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def auc_score(output, labels):
    probs = torch.exp(output)  # 还原成真实概率
    scores = probs[:, 1].cpu().detach().numpy()
    return roc_auc_score(labels.cpu().numpy(), scores)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def plot_all_metrics(metrics_history, result_dir):
    metrics = ['loss', 'val_acc', 'val_auc', 'val_f1', 'score']
    num_metrics = len(metrics)

    fig, axs = plt.subplots(num_metrics, 1, figsize=(8, 4 * num_metrics))  # 纵向排布

    for i, metric in enumerate(metrics):
        axs[i].plot(metrics_history['epoch'], metrics_history[metric], label=metric)
        axs[i].set_title(metric)
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(metric)
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'all_metrics.png'))
    plt.close()
