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
    0: "Frontal Lobe", # 例如: 额叶
    1: "Occipital Lobe",# 例如: 顶叶
    2: "Parietal Lobe",# 例如: 颞叶
    3: "Prefontal Lobe",# 例如: 枕叶
    4: "Subcortical Lobe",  # 例如: 边缘叶
    5: "Temporal" # 例如: 小脑和脑干
}

# -------------- 分析函数 -------------- #
def analyze_spatial_perturbations(model, data_loader, output_dir, anatomical_labels=ANATOMICAL_LABELS):
    """
    研究空间扰动 (Δ_j)
    - 计算并可视化 ||Δ_j|| 在各脑区的分布
    - 分析ASD和HC组间脑区扰动差异
    - 生成脑图可视化不同脑区的扰动范数
    """
    print("开始分析空间扰动 (Δ_j)...")
    ensure_dir(output_dir)
    device = get_device()
    model.to(device)
    model.eval()

    # 创建存储结构，按类别分开存储
    delta_j_data = {0: [], 1: []}  # 假设二分类问题：0=HC, 1=ASD
    all_delta_j_magnitudes = []
    all_anatomical_classes = []
    all_labels = []
    
    # 记录每个类别的前10个扰动最大的节点
    top_nodes = {0: [], 1: []}

    # 将数据加载器数据转为列表以便随机抽样
    all_data = list(data_loader)
    
    # 从每个类别随机选择一个样本
    selected_samples = {0: None, 1: None}
    
    # 首先将数据按类别分组
    class_grouped_data = {0: [], 1: []}
    for batch in all_data:
        for i in range(len(batch.y)):
            # 抽取单个样本
            single_data = batch[i]
            label = single_data.y.item()-1
            class_grouped_data[label].append(single_data)
    
    # 从每个类别随机选择一个样本
    for label in [0, 1]:
        if class_grouped_data[label]:
            selected_samples[label] = random.choice(class_grouped_data[label])
            print(f"已为类别 {label} 随机选择样本")
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    with torch.no_grad():
        # 分析随机选择的样本
        for label, sample in selected_samples.items():
            if sample is None:
                print(f"警告: 未找到类别 {label} 的样本")
                continue
                
            print(f"分析类别 {label} 的样本")
            
            sample = sample.to(device)
            
            try:
                # 从模型前向传播中提取/重新计算 Δ_j
                x_orig, edge_index, edge_attr, batch_idx, coords, anatomical_class_batch = \
                    sample.x, sample.edge_index, sample.edge_attr, sample.batch, sample.coords, sample.anatomical_class
                
                # 如果batch_idx为None，创建一个全零的批次索引
                if batch_idx is None:
                    batch_idx = torch.zeros(sample.num_nodes, dtype=torch.long, device=device)
                
                current_edge_attr = edge_attr
                if current_edge_attr is not None and current_edge_attr.ndim > 1:
                    current_edge_attr = current_edge_attr.squeeze()
                if current_edge_attr is None:
                    current_edge_attr = torch.ones(edge_index.size(1), device=device)

                adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                                  value=current_edge_attr,
                                  sparse_sizes=(sample.num_nodes, sample.num_nodes)).to(device)

                B_k_raw = model.B_k(anatomical_class_batch)
                B_k_matrix = B_k_raw.view(-1, model.hidden_channels, model.hidden_channels)

                e_j_pos = model.position_encoder(coords)
                Ae_j = adj @ e_j_pos
                Ae_j = Ae_j.unsqueeze(1)
                delta_j = torch.bmm(Ae_j, B_k_matrix.transpose(1, 2)).squeeze(1)
                
                # 计算范数
                delta_j_magnitudes = torch.norm(delta_j, p=2, dim=1).cpu().numpy()
                
                # 保存数据用于后续分析
                delta_j_data[label].append({
                    'magnitudes': delta_j_magnitudes,
                    'anatomical_classes': anatomical_class_batch.cpu().numpy()
                })
                
                # 找出扰动范数最大的前10个节点
                top_indices = np.argsort(delta_j_magnitudes)[-10:][::-1]
                top_nodes[label] = [{'index': int(idx), 
                                     'magnitude': float(delta_j_magnitudes[idx]), 
                                     'anatomical_class': int(anatomical_class_batch[idx].cpu().numpy())} 
                                    for idx in top_indices]
                
                # 按解剖学类别分组可视化
                unique_classes = np.unique(anatomical_class_batch.cpu().numpy())
                magnitudes_by_class = {}
                for cls in unique_classes:
                    indices = np.where(anatomical_class_batch.cpu().numpy() == cls)[0]
                    magnitudes_by_class[cls] = delta_j_magnitudes[indices]
                
                # 解剖学类别的扰动分布 - 更美观的箱线图
                fig, ax = plt.subplots(figsize=(14, 8))
                
                box_data = [magnitudes_by_class[cls] for cls in unique_classes]
                box_labels = [anatomical_labels.get(cls, f"Class {cls}") for cls in unique_classes]
                
                # 使用更好的配色方案
                colors = plt.cm.viridis(np.linspace(0, 0.8, len(unique_classes)))
                bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, 
                                boxprops=dict(alpha=0.8), 
                                medianprops=dict(color='red'))
                
                # 美化箱线图
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1.5)
                
                ax.set_title(f"Spatial Perturbation by Anatomical Region ({'ASD' if label == 1 else 'HC'})")
                ax.set_ylabel("||Δ_j||")
                ax.set_xlabel("Anatomical Region")
                plt.xticks(rotation=30, ha="right")
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                save_figure(fig, os.path.join(output_dir, f"spatial_perturbation_by_anatomy_class_{label}.png"))
                
                # 记录所有数据用于组间比较
                all_delta_j_magnitudes.extend(delta_j_magnitudes)
                all_anatomical_classes.extend(anatomical_class_batch.cpu().numpy())
                all_labels.extend([label] * len(delta_j_magnitudes))
                
                print(f"完成类别 {label} 的样本分析")
                
            except Exception as e:
                print(f"警告: 分析类别 {label} 的样本时出错: {e}")
                traceback.print_exc()  # 打印完整的异常堆栈
        
        # 保存每个类别前10个扰动最大的节点
        for label in [0, 1]:
            df = pd.DataFrame(top_nodes[label])
            df['region_name'] = df['anatomical_class'].apply(lambda x: anatomical_labels.get(x, f"Class {x}"))
            df.to_csv(os.path.join(output_dir, f"top10_perturbed_nodes_class_{label}.csv"), index=False)
            
        # 找出两个类别前10名节点的交集
        if top_nodes[0] and top_nodes[1]:
            nodes_hc = set([node['index'] for node in top_nodes[0]])
            nodes_asd = set([node['index'] for node in top_nodes[1]])
            common_nodes = nodes_hc.intersection(nodes_asd)
            
            if common_nodes:
                # 保存交集节点
                common_data = []
                for node_idx in common_nodes:
                    hc_mag = next(node['magnitude'] for node in top_nodes[0] if node['index'] == node_idx)
                    asd_mag = next(node['magnitude'] for node in top_nodes[1] if node['index'] == node_idx)
                    anat_class = next(node['anatomical_class'] for node in top_nodes[0] if node['index'] == node_idx)
                    common_data.append({
                        'node_index': node_idx,
                        'hc_magnitude': hc_mag,
                        'asd_magnitude': asd_mag,
                        'difference': asd_mag - hc_mag,
                        'percent_diff': ((asd_mag - hc_mag) / hc_mag * 100) if hc_mag != 0 else float('inf'),
                        'anatomical_class': anat_class,
                        'region_name': anatomical_labels.get(anat_class, f"Class {anat_class}")
                    })
                
                # 保存交集数据
                df_common = pd.DataFrame(common_data)
                df_common.to_csv(os.path.join(output_dir, "common_top_perturbed_nodes.csv"), index=False)
        
        # 如果有足够的样本数据进行组间比较
        if delta_j_data[0] and delta_j_data[1]:
            try:
                # 将数据转换为数组以便进行组间比较
                all_delta_j_magnitudes = np.array(all_delta_j_magnitudes)
                all_anatomical_classes = np.array(all_anatomical_classes)
                all_labels = np.array(all_labels)
                
                # 组间比较：对每个解剖类别，比较ASD和HC的扰动分布
                unique_anatomical_classes = np.unique(all_anatomical_classes)
                
                # 创建组间比较表格
                comparison_data = []
                for cls in unique_anatomical_classes:
                    cls_indices = np.where(all_anatomical_classes == cls)[0]
                    hc_indices = np.where(np.logical_and(all_anatomical_classes == cls, np.array(all_labels) == 0))[0]
                    asd_indices = np.where(np.logical_and(all_anatomical_classes == cls, np.array(all_labels) == 1))[0]
                    
                    if len(hc_indices) > 0 and len(asd_indices) > 0:
                        hc_mean = np.mean(all_delta_j_magnitudes[hc_indices])
                        asd_mean = np.mean(all_delta_j_magnitudes[asd_indices])
                        mean_diff = asd_mean - hc_mean
                        
                        # 计算差异百分比
                        if hc_mean != 0:
                            percent_diff = (mean_diff / hc_mean) * 100
                        else:
                            percent_diff = float('inf') if mean_diff > 0 else float('-inf')
                        
                        # 收集用于箱线图的数据
                        hc_values = all_delta_j_magnitudes[hc_indices]
                        asd_values = all_delta_j_magnitudes[asd_indices]
                        
                        comparison_data.append({
                            'anatomical_class': cls,
                            'region_name': anatomical_labels.get(cls, f"Class {cls}"),
                            'hc_mean': hc_mean,
                            'asd_mean': asd_mean,
                            'hc_values': hc_values,
                            'asd_values': asd_values,
                            'mean_diff': mean_diff,
                            'percent_diff': percent_diff
                        })
                
                # 按差异大小排序
                comparison_data.sort(key=lambda x: abs(x['mean_diff']), reverse=True)
                
                # 可视化前N个差异最大的区域
                top_n = min(10, len(comparison_data))
                top_diffs = comparison_data[:top_n]
                
                # 创建HC vs ASD扰动差异的高质量柱状图
                fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
                x = np.arange(top_n)
                width = 0.35
                
                # 绘制ASD和HC的柱状图，使用更好的颜色
                rects1 = ax.bar(x - width/2, [d['hc_mean'] for d in top_diffs], width, 
                               label='HC', color='#3498db', alpha=0.85, edgecolor='black', linewidth=1)
                rects2 = ax.bar(x + width/2, [d['asd_mean'] for d in top_diffs], width, 
                               label='ASD', color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1)
                
                ax.set_ylabel('Mean ||Δ_j||', fontweight='bold')
                ax.set_title('Brain Regions with Largest Perturbation Difference', fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels([d['region_name'] for d in top_diffs], rotation=30, ha='right')
                ax.legend(frameon=True, facecolor='white', edgecolor='gray')
                
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                plt.tight_layout()
                save_figure(fig, os.path.join(output_dir, "hc_vs_asd_perturbation_differences.png"))
                
                # 创建差异百分比的条形图 - 更美观的版本
                fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
                
                # 使用渐变色而不是简单的红蓝
                norm = plt.Normalize(min([d['percent_diff'] for d in top_diffs]), 
                                     max([d['percent_diff'] for d in top_diffs]))
                colors = plt.cm.RdBu_r(norm([d['percent_diff'] for d in top_diffs]))
                
                bars = ax.bar(x, [d['percent_diff'] for d in top_diffs], color=colors, 
                             edgecolor='black', linewidth=1)
                
                ax.set_ylabel('Percent Difference (%)', fontweight='bold')
                ax.set_title('Perturbation Difference Between ASD and HC', fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels([d['region_name'] for d in top_diffs], rotation=30, ha='right')
                
                # 添加水平线表示零差异
                ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                
                plt.tight_layout()
                save_figure(fig, os.path.join(output_dir, "perturbation_percent_differences.png"))
                
                # 添加ASD-HC扰动范数差异分布的箱线图
                fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
                
                # 计算每个解剖区域的差异
                diff_data = []
                diff_labels = []
                
                for d in comparison_data:
                    if len(d['hc_values']) > 0 and len(d['asd_values']) > 0:
                        # 计算差异向量 - 需要处理长度不同的情况
                        hc_vals = d['hc_values']
                        asd_vals = d['asd_values']
                        
                        # 使用较短的长度
                        min_len = min(len(hc_vals), len(asd_vals))
                        diffs = asd_vals[:min_len] - hc_vals[:min_len]
                        
                        diff_data.append(diffs)
                        diff_labels.append(d['region_name'])
                
                if diff_data:
                    # 按中位数差异排序
                    median_diffs = [np.median(d) for d in diff_data]
                    sorted_indices = np.argsort(median_diffs)[::-1]
                    
                    sorted_diff_data = [diff_data[i] for i in sorted_indices]
                    sorted_diff_labels = [diff_labels[i] for i in sorted_indices]
                    
                    # 选择前10个最大差异的区域
                    plot_n = min(10, len(sorted_diff_data))
                    plot_data = sorted_diff_data[:plot_n]
                    plot_labels = sorted_diff_labels[:plot_n]
                    
                    # 使用更好的配色方案
                    bp_colors = plt.cm.viridis(np.linspace(0, 0.8, plot_n))
                    bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True,
                                    boxprops=dict(alpha=0.8),
                                    medianprops=dict(color='red', linewidth=1.5))
                    
                    # 美化箱线图
                    for patch, color in zip(bp['boxes'], bp_colors):
                        patch.set_facecolor(color)
                        patch.set_edgecolor('black')
                        patch.set_linewidth(1.5)
                    
                    ax.set_title('ASD-HC Perturbation Magnitude Differences by Region', fontweight='bold')
                    ax.set_ylabel('Δ||Δ_j|| (ASD - HC)', fontweight='bold')
                    ax.set_xlabel('Anatomical Region', fontweight='bold')
                    plt.xticks(rotation=30, ha='right')
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    # 添加零线
                    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
                    
                    plt.tight_layout()
                    save_figure(fig, os.path.join(output_dir, "asd_hc_perturbation_difference_boxplot.png"))
                
                # 保存详细的比较数据到CSV
                df = pd.DataFrame([{k: v for k, v in d.items() if k not in ['hc_values', 'asd_values']} 
                                  for d in comparison_data])
                df.to_csv(os.path.join(output_dir, "perturbation_comparison_by_region.csv"), index=False)
                    
            except Exception as e:
                print(f"进行组间比较分析时出错: {e}")
                traceback.print_exc()
    
    print("空间扰动分析完成。")


# 在主函数中只调用空间扰动分析
def run_msmegat_interpretability_analysis(
    model_path,             # 路径: 已训练模型的 .pth 文件
    data_loader,            # PyG DataLoader: 用于分析的数据 (通常是测试集)
    output_base_dir,        # 路径: 保存分析结果的目录
    # --- 模型构建所需的核心参数 ---
    num_node_features,      # int: 节点特征维度
    num_classes,            # int: 输出类别数
    hidden_channels,        # int: 隐藏层维度
    heads,                  # int: GATv2注意力头数
    num_anatomical_classes, # int: 解剖类别数量
    # --- 可选参数 ---
    anatomical_labels_map=ANATOMICAL_LABELS
):
    """
    主函数，用于运行MSMEGAT模型的可解释性分析。
    专注于空间扰动分析。
    """
    print(f"开始MSMEGAT空间扰动分析，结果将保存在: {output_base_dir}")
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

    # 只执行空间扰动分析
    analyze_spatial_perturbations(model, data_loader, os.path.join(output_base_dir, "spatial_perturbations_analysis"),
                                 anatomical_labels=anatomical_labels_map)

    print("MSMEGAT空间扰动分析完成。")


if __name__ == '__main__':
    print("运行 analysis.py 示例 (使用真实模型和数据加载逻辑)...")

    # --- 1. 定义您的模型路径和数据加载 ---
    # !! 您需要修改这些路径和参数以匹配您的设置 !!
    PATH_TO_TRAINED_MODEL = "results/classification:run_20250520_0546/MSMEGAT/best_model.pth" 
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
