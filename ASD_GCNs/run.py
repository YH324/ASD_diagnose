# run.py - 多组实验
import subprocess

configs = [
    {'model': 'GCN', 'lr': '0.001'},
    {'model': 'GraphSAGE', 'lr': '0.001'},
    {'model': 'DeepGraphSAGE', 'lr': '0.001'},
    # {'model': 'GATv2Net', 'lr': '0.001'},
]

for config in configs:
    command = [
        'python', 'train.py',
        '--epochs', '150',
        '--lr', config['lr'],
        '--weight_decay', '0.0005',
        '--train_ratio', '0.6',
        '--val_ratio', '0.2',
        '--model', config['model']
    ]
    print(f"Running with config: {config}")
    subprocess.run(command)
