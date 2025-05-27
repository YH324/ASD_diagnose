import subprocess
import re
model_list = ['MSMEGAT','GCN', 'GraphSAGE', 'GAT']
l1_list = [0.01, 0.05, 0.1, 0.2, 0.5, 1]

for model in model_list:
#for l2 in l1_list:
    #for l1 in l1_list:
    # 替换 config.py 中的 model_name 行
    with open('config.py', 'r') as f:
        config = f.read()

    config = re.sub(r"model_name\s*=.*", f"model_name = '{model}'", config)
    #config = re.sub(r"l2\s*=.*", f"l2 = {l2}", config)
    #config = re.sub(r"l1\s*=.*", f"l1 = {l1}", config)

    with open('config.py', 'w') as f:
        f.write(config)

    print(f"\nRunning model: {model}")
    #print(f"\nRunning L2:{l2} L1:{l1}")
    subprocess.run(['python', 'train.py'])
