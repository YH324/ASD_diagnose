import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import traceback
from torch_sparse import SparseTensor
from sklearn.preprocessing import minmax_scale
from config import *
from models import MSMEGAT
from message_kernel import compute_propagation_kernel, propagate_message
from torch_geometric.data import DataLoader, Data
from utils import load_graph_dataset
from preprocess_data import preprocess_data
from tqdm import tqdm
import pandas as pd  


# -------------- 配置与辅助函数 -------------- #

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_figure(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

ANATOMICAL_LABELS = {
    0: "Frontal Lobe",
    1: "Occipital Lobe",
    2: "Parietal Lobe",
    3: "Prefontal Lobe",
    4: "Subcortical Lobe",
    5: "Temporal"
}

# -------------- 分析函数 -------------- #
def analyze_feature_perturbations(model, data_loader, output_dir, anatomical_labels=ANATOMICAL_LABELS):
    """
    研究特征扰动 (Δh_j)
    - 计算Δh_j = Δ_j * x (原特征与扰动的交互)
    - 分析不同脑区特征变化的模式
    - 比较ASD和HC组在特征扰动上的差异
    """
    print("开始分析特征扰动 (Δh_j)...")
    ensure_dir(output_dir)
    device = get_device()
    model.to(device)
    model.eval()

    # 创建存储结构，按类别分开存储
    delta_h_data = {0: [], 1: []}  # 0=HC, 1=ASD
    feature_contributions = {0: {}, 1: {}}  # 记录每个特征的贡献
    
    # 从每个类别随机选择样本
    all_data = list(data_loader)
    class_grouped_data = {0: [], 1: []}
    for batch in all_data:
        for i in range(len(batch.y)):
            single_data = batch[i]
            label = single_data.y.item()-1
            class_grouped_data[label].append(single_data)
    
    # 从每个类别随机选择一个样本
    selected_samples = {0: None, 1: None}
    for label in [0, 1]:
        if class_grouped_data[label]:
            selected_samples[label] = random.choice(class_grouped_data[label])
            print(f"已为类别 {label} 随机选择样本")
    
    with torch.no_grad():
        # 分析随机选择的样本
        for label, sample in selected_samples.items():
            if sample is None:
                continue
                
            print(f"分析类别 {label} 的样本特征扰动")
            
            sample = sample.to(device)
            
            try:
                # 获取原始特征和其他数据
                x_orig, edge_index, edge_attr, batch_idx, coords, anatomical_class_batch = \
                    sample.x, sample.edge_index, sample.edge_attr, sample.batch, sample.coords, sample.anatomical_class
                
                # 如果batch_idx为None，创建一个全零的批次索引
                if batch_idx is None:
                    batch_idx = torch.zeros(sample.num_nodes, dtype=torch.long, device=device)
                
                # 计算扰动Δ_j
                current_edge_attr = edge_attr
                if current_edge_attr is not None and current_edge_attr.ndim > 1:
                    current_edge_attr = current_edge_attr.squeeze()
                if current_edge_attr is None:
                    current_edge_attr = torch.ones(edge_index.size(1), device=device)

                adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                                  value=current_edge_attr,
                                  sparse_sizes=(sample.num_nodes, sample.num_nodes)).to(device)

                # 获取B_k和计算Δ_j
                B_k_raw = model.B_k(anatomical_class_batch)
                B_k_matrix = B_k_raw.view(-1, model.hidden_channels, model.hidden_channels)

                e_j_pos = model.position_encoder(coords)
                Ae_j = adj @ e_j_pos
                Ae_j = Ae_j.unsqueeze(1)
                delta_j = torch.bmm(Ae_j, B_k_matrix.transpose(1, 2)).squeeze(1)
                
                # 获取第一层卷积后的节点表示
                # 对原始特征应用GATv2卷积
                x_conv = model.conv1(x_orig, edge_index)
                x_conv = model.norm1(x_conv, batch_idx)
                x_conv = F.elu(x_conv)
                
                # 计算特征扰动：Δh_j = Δ_j * x_conv
                # 1. 计算扰动权重
                delta_weights = delta_j  # [N, hidden_channels]
                
                # 2. 计算特征扰动
                delta_h_j = delta_weights * x_conv  # 逐元素相乘: [N, hidden_channels]
                
                # 3. 计算扰动对原特征的相对影响
                relative_impact = torch.abs(delta_h_j) / (torch.abs(x_conv) + 1e-8)  # 防止除零
                
                # 将结果转换为numpy数组
                delta_h_j_np = delta_h_j.cpu().numpy()
                x_conv_np = x_conv.cpu().numpy()
                relative_impact_np = relative_impact.cpu().numpy()
                anatomical_classes_np = anatomical_class_batch.cpu().numpy()
                
                # 保存结果
                delta_h_data[label].append({
                    'delta_h_j': delta_h_j_np,
                    'original_features': x_conv_np,
                    'relative_impact': relative_impact_np,
                    'anatomical_classes': anatomical_classes_np
                })
                
                # 分析每个特征维度的贡献
                feature_dims = delta_h_j.shape[1]
                for dim in range(feature_dims):
                    # 按解剖区域分组分析特征维度
                    for anat_class in np.unique(anatomical_classes_np):
                        class_mask = anatomical_classes_np == anat_class
                        
                        # 计算该特征维度在该解剖区域的平均扰动
                        mean_impact = np.mean(np.abs(delta_h_j_np[class_mask, dim]))
                        
                        # 记录特征贡献
                        if anat_class not in feature_contributions[label]:
                            feature_contributions[label][anat_class] = np.zeros(feature_dims)
                        
                        feature_contributions[label][anat_class][dim] = mean_impact
                
                # 可视化每个解剖区域的特征扰动模式
                unique_classes = np.unique(anatomical_classes_np)
                
                # 1. 解剖区域的平均特征扰动热力图
                fig, axes = plt.subplots(len(unique_classes), 1, figsize=(12, 4*len(unique_classes)), dpi=120)
                if len(unique_classes) == 1:
                    axes = [axes]
                
                for i, anat_class in enumerate(unique_classes):
                    class_mask = anatomical_classes_np == anat_class
                    if np.sum(class_mask) == 0:
                        continue
                        
                    # 计算该解剖区域的平均特征扰动
                    mean_delta_h = np.mean(np.abs(delta_h_j_np[class_mask]), axis=0)
                    
                    # 绘制热力图
                    im = axes[i].imshow(mean_delta_h.reshape(1, -1), cmap='viridis', aspect='auto')
                    axes[i].set_title(f"{anatomical_labels.get(anat_class, f'Class {anat_class}')} - Mean Feature Perturbation")
                    axes[i].set_xlabel("Feature Dimension")
                    axes[i].set_yticks([])
                    
                    # 添加颜色条
                    plt.colorbar(im, ax=axes[i])
                
                plt.tight_layout()
                save_figure(fig, os.path.join(output_dir, f"feature_perturbation_heatmap_class_{label}.png"))
                
                # 2. 特征扰动的主成分分析 (PCA)
                if delta_h_j_np.shape[0] > 2:  # 需要至少3个样本
                    from sklearn.decomposition import PCA
                    
                    # 对特征扰动应用PCA
                    pca = PCA(n_components=2)
                    delta_h_pca = pca.fit_transform(delta_h_j_np)
                    
                    # 绘制PCA散点图，按解剖区域着色
                    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
                    
                    for anat_class in unique_classes:
                        class_mask = anatomical_classes_np == anat_class
                        if np.sum(class_mask) < 2:  # 需要至少2个点才能绘制
                            continue
                            
                        ax.scatter(
                            delta_h_pca[class_mask, 0], 
                            delta_h_pca[class_mask, 1],
                            label=anatomical_labels.get(anat_class, f'Class {anat_class}'),
                            alpha=0.7,
                            s=50
                        )
                    
                    ax.set_title(f"PCA of Feature Perturbations ({'ASD' if label == 1 else 'HC'})")
                    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
                    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
                    ax.legend()
                    ax.grid(alpha=0.3)
                    
                    plt.tight_layout()
                    save_figure(fig, os.path.join(output_dir, f"feature_perturbation_pca_class_{label}.png"))
                
                # 3. 相对影响分析 - 哪些特征受扰动影响最大
                mean_relative_impact = np.mean(relative_impact_np, axis=0)
                
                # 排序并找出最受影响的特征
                top_features_idx = np.argsort(mean_relative_impact)[::-1][:10]  # 前10个最受影响的特征
                
                fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
                ax.bar(
                    np.arange(len(top_features_idx)),
                    mean_relative_impact[top_features_idx],
                    color='skyblue',
                    edgecolor='navy'
                )
                ax.set_title(f"Top 10 Most Affected Features ({'ASD' if label == 1 else 'HC'})")
                ax.set_xlabel("Feature Index")
                ax.set_ylabel("Mean Relative Impact")
                ax.set_xticks(np.arange(len(top_features_idx)))
                ax.set_xticklabels([str(idx) for idx in top_features_idx])
                ax.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                save_figure(fig, os.path.join(output_dir, f"top_affected_features_class_{label}.png"))
                
            except Exception as e:
                print(f"警告: 分析类别 {label} 的样本时出错: {e}")
                traceback.print_exc()
        
        # 对比ASD和HC的特征扰动模式
        if delta_h_data[0] and delta_h_data[1]:
            try:
                # 分析共有的解剖区域
                common_anat_classes = set(feature_contributions[0].keys()) & set(feature_contributions[1].keys())
                
                for anat_class in common_anat_classes:
                    hc_contribution = feature_contributions[0][anat_class]
                    asd_contribution = feature_contributions[1][anat_class]
                    
                    # 计算差异
                    diff_contribution = asd_contribution - hc_contribution
                    
                    # 找出差异最大的特征
                    top_diff_idx = np.argsort(np.abs(diff_contribution))[::-1][:20]  # 前20个差异最大的特征
                    
                    # 绘制差异条形图
                    fig, ax = plt.subplots(figsize=(14, 8), dpi=120)
                    
                    x = np.arange(len(top_diff_idx))
                    width = 0.35
                    
                    ax.bar(x - width/2, hc_contribution[top_diff_idx], width, label='HC', color='#3498db', alpha=0.7, edgecolor='black')
                    ax.bar(x + width/2, asd_contribution[top_diff_idx], width, label='ASD', color='#e74c3c', alpha=0.7, edgecolor='black')
                    
                    # 绘制差异线
                    for i, idx in enumerate(top_diff_idx):
                        ax.plot([i-width/2, i+width/2], [hc_contribution[idx], asd_contribution[idx]], 'k-', alpha=0.5)
                    
                    ax.set_title(f"ASD vs HC Feature Perturbation - {anatomical_labels.get(anat_class, f'Class {anat_class}')}")
                    ax.set_xlabel("Feature Index")
                    ax.set_ylabel("Mean Feature Perturbation")
                    ax.set_xticks(x)
                    ax.set_xticklabels([str(idx) for idx in top_diff_idx])
                    ax.legend()
                    ax.grid(axis='y', alpha=0.3)
                    
                    plt.tight_layout()
                    save_figure(fig, os.path.join(output_dir, f"asd_vs_hc_feature_diff_region_{anat_class}.png"))
                    
                # 创建所有解剖区域的热力图比较
                all_regions = sorted(list(common_anat_classes))
                feature_dims = next(iter(feature_contributions[0].values())).shape[0]
                
                # 准备热力图数据
                hc_heatmap = np.zeros((len(all_regions), feature_dims))
                asd_heatmap = np.zeros((len(all_regions), feature_dims))
                diff_heatmap = np.zeros((len(all_regions), feature_dims))
                
                for i, anat_class in enumerate(all_regions):
                    hc_heatmap[i] = feature_contributions[0][anat_class]
                    asd_heatmap[i] = feature_contributions[1][anat_class]
                    diff_heatmap[i] = asd_heatmap[i] - hc_heatmap[i]
                
                # 绘制热力图
                fig, axes = plt.subplots(3, 1, figsize=(14, 18), dpi=120)
                
                # HC热力图
                im0 = axes[0].imshow(hc_heatmap, cmap='viridis', aspect='auto')
                axes[0].set_title("HC - Feature Perturbation by Brain Region")
                axes[0].set_ylabel("Brain Region")
                axes[0].set_yticks(np.arange(len(all_regions)))
                axes[0].set_yticklabels([anatomical_labels.get(r, f'Class {r}') for r in all_regions])
                plt.colorbar(im0, ax=axes[0])
                
                # ASD热力图
                im1 = axes[1].imshow(asd_heatmap, cmap='viridis', aspect='auto')
                axes[1].set_title("ASD - Feature Perturbation by Brain Region")
                axes[1].set_ylabel("Brain Region")
                axes[1].set_yticks(np.arange(len(all_regions)))
                axes[1].set_yticklabels([anatomical_labels.get(r, f'Class {r}') for r in all_regions])
                plt.colorbar(im1, ax=axes[1])
                
                # 差异热力图
                im2 = axes[2].imshow(diff_heatmap, cmap='RdBu_r', aspect='auto')
                axes[2].set_title("ASD - HC Difference in Feature Perturbation")
                axes[2].set_xlabel("Feature Dimension")
                axes[2].set_ylabel("Brain Region")
                axes[2].set_yticks(np.arange(len(all_regions)))
                axes[2].set_yticklabels([anatomical_labels.get(r, f'Class {r}') for r in all_regions])
                plt.colorbar(im2, ax=axes[2])
                
                plt.tight_layout()
                save_figure(fig, os.path.join(output_dir, "feature_perturbation_comparison_heatmap.png"))
                
            except Exception as e:
                print(f"比较ASD和HC特征扰动时出错: {e}")
                traceback.print_exc()
    
    print("特征扰动分析完成。")


# 更新主函数以包含新的特征扰动分析
def run_msmegat_interpretability_analysis(
    model_path,
    data_loader,
    output_base_dir,
    num_node_features,
    num_classes,
    hidden_channels,
    heads,
    num_anatomical_classes,
    anatomical_labels_map=ANATOMICAL_LABELS
):
    """
    主函数，用于运行MSMEGAT模型的可解释性分析。
    包括空间扰动分析和特征扰动分析。
    """
    print(f"开始MSMEGAT模型可解释性分析，结果将保存在: {output_base_dir}")
    device = get_device()

    # 1. 构建模型实例
    model = MSMEGAT(
        num_node_features=num_node_features,
        num_classes=num_classes,
        hidden_channels=hidden_channels,
        heads=heads,
        num_anatomical_classes=num_anatomical_classes
    )
    print(f"加载模型权重从: {model_path}")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        print(f"加载模型权重时出错: {e}")
        print("这可能是因为模型定义参数与保存的权重不匹配，或者 'MSMEGAT' 类未正确导入。")
        print("请确保传入的参数与训练时一致，并且 MSMEGAT 类已正确导入。")
        return

    model.to(device)
    model.eval()

    # 执行空间扰动分析
    # analyze_spatial_perturbations(model, data_loader, os.path.join(output_base_dir, "spatial_perturbations_analysis"),
    #                             anatomical_labels=anatomical_labels_map)
    
    # 执行特征扰动分析
    analyze_feature_perturbations(model, data_loader, os.path.join(output_base_dir, "feature_perturbations_analysis"),
                                 anatomical_labels=anatomical_labels_map)

    print("MSMEGAT模型可解释性分析完成。")


if __name__ == '__main__':
    print("运行 analysis.py 示例 (使用真实模型和数据加载逻辑)...")

    # --- 1. 定义您的模型路径和数据加载 ---
    # !! 您需要修改这些路径和参数以匹配您的设置 !!
    PATH_TO_TRAINED_MODEL = "results/classification:run_20250527_1842/MSMEGAT/best_model.pth" 
    OUTPUT_DIR_BASE = "vis"

    # --- 2. 定义模型构建参数 (必须与训练时一致) ---
    # !! 替换为您的真实参数 !!
    _NUM_NODE_FEATURES = 90  # 示例: AAL116脑区特征数
    _NUM_CLASSES = 2          # 示例: ASD vs HC 二分类
    _HIDDEN_CHANNELS = 64
    _HEADS = 4
    _NUM_ANATOMICAL_CLASSES = 6 # 示例: 6个主要的解剖分区

    # --- 3. 准备 DataLoader ---
    print(f"使用自定义函数从 '{feature_path}' 加载和预处理数据...")
    try:
        # 调用您的 preprocess_data 函数
        # 确保传递给 preprocess_data 的参数是它所期望的
        train_data, val_data, test_data = preprocess_data(feature_path,train_ratio=train_ratio, val_ratio=val_ratio)
        print("数据预处理完成。")

        _, val_loader, test_loader = load_graph_dataset(train_data, val_data, test_data, batch_size=batch_size)
        print(f"测试数据 DataLoader (batch_size={batch_size}) 已创建，用于分析。")

    except Exception as e:
        print(f"在数据加载/预处理阶段发生错误: {e}")
        print("请检查 feature_path 是否正确，以及 preprocess_data/load_graph_dataset 函数的参数和实现。")
        exit()


    # --- 4. 检查模型文件是否存在 (可选但推荐) ---
    if not os.path.exists(PATH_TO_TRAINED_MODEL):
        print(f"错误: 找不到模型文件 '{PATH_TO_TRAINED_MODEL}'")
        print("请创建一个模拟模型文件用于测试，或提供正确的路径。")

        # 确保 MSMEGAT 类被导入，即使只是mock
        if 'MSMEGAT' not in globals() or MSMEGAT.__name__ == 'MSMEGAT' and not MSMEGAT.__module__.startswith('models'):
            print("MSMEGAT 类未正确导入。无法创建模拟模型。请修复导入。")
       

    # --- 5. 调用主分析函数 ---
    # 确保 MSMEGAT, compute_propagation_kernel, propagate_message 已正确导入
    # 如果在脚本顶部的 try-except 中导入失败，这里可能会使用 mock 或引发错误
    if MSMEGAT.__name__ == 'MSMEGAT' and not MSMEGAT.__module__.startswith('models'): # 检查是否是真实的MSMEGAT
        print("警告: 似乎正在使用模拟的 MSMEGAT 类。分析可能不准确或失败。")
        print("请确保 `from models.msmegat_model import MSMEGAT` 成功执行。")

    run_msmegat_interpretability_analysis(
        model_path=PATH_TO_TRAINED_MODEL,
        data_loader=val_loader,
        output_base_dir=OUTPUT_DIR_BASE,
        num_node_features=_NUM_NODE_FEATURES,
        num_classes=_NUM_CLASSES,
        hidden_channels=_HIDDEN_CHANNELS,
        heads=_HEADS,
        num_anatomical_classes=_NUM_ANATOMICAL_CLASSES,
        anatomical_labels_map=ANATOMICAL_LABELS)
    print(f"示例分析脚本运行结束. 结果应保存在 '{OUTPUT_DIR_BASE}' 目录。")
