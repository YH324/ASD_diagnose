# preprocess.py
import torch
from utils.data_loader import load_node_static_features, load_all_samples
from config import DATASET_PATH

def main():
    Y_enriched = load_node_static_features()
    samples = load_all_samples(Y_enriched)
    torch.save(samples, DATASET_PATH)
    print(f"[保存成功] 数据集已保存至 {DATASET_PATH}，共 {len(samples)} 个样本")

if __name__ == '__main__':
    main()
