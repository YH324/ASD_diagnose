import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from datetime import datetime

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            all_preds.append(out.cpu().numpy())
            all_targets.append(data.y.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    mse = mean_squared_error(targets, preds)
    r2 = r2_score(targets, preds)
    return mse, r2, preds, targets

def log_metrics(epoch, loss, mse, r2, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = not os.path.exists(path)
    row = pd.DataFrame([{
        'epoch': epoch,
        'train_loss': loss,
        'val_mse': mse,
        'val_r2': r2
    }])
    row.to_csv(path, mode='a', header=header, index=False)

def plot_predictions(preds, targets, save_path1):
    os.makedirs(os.path.dirname(save_path1), exist_ok=True)
    plt.figure()
    plt.scatter(targets, preds, alpha=0.6)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('GAT Predictions vs Targets')
    plt.savefig(save_path1)
    plt.close()

def plot_training_curves(log_path, save_path2):
    os.makedirs(os.path.dirname(save_path2), exist_ok=True)

    df = pd.read_csv(log_path)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_mse'], label='Val MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss & MSE')

    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['val_r2'], label='Val R2')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.title('Validation R²')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path2)
    plt.close()
