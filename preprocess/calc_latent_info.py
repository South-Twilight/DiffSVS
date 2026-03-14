import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def compute_and_print_stats(tsv_path):
    print(f"📂 正在读取清单: {tsv_path}")
    df = pd.read_csv(tsv_path, sep='\t')
    
    path_col = 'latent_path' if 'latent_path' in df.columns else 'mel_path'
    latent_paths = df[path_col].tolist()
    
    total_sum = None
    total_sq_sum = None
    total_frames = 0
    
    for path in tqdm(latent_paths, desc="计算中"):
        if not os.path.exists(path):
            continue
            
        latent = np.load(path) # [128, T]
        
        if total_sum is None:
            num_channels = latent.shape[0]
            total_sum = np.zeros(num_channels, dtype=np.float64)
            total_sq_sum = np.zeros(num_channels, dtype=np.float64)
            
        total_sum += np.sum(latent, axis=1)
        total_sq_sum += np.sum(latent ** 2, axis=1)
        total_frames += latent.shape[1]
        
    if total_frames == 0:
        print("❌ 错误：未找到任何有效数据！")
        return

    # 计算 128 维的均值和标准差
    channel_mean = total_sum / total_frames
    channel_var = (total_sq_sum / total_frames) - (channel_mean ** 2)
    channel_std = np.sqrt(np.maximum(channel_var, 1e-8))
    
    # 格式化输出为可以直接复制的 Python 代码
    print("\n" + "="*60)
    print("🎉 计算完成！请将以下代码直接复制到你的 DiffSVSDataset 类中：")
    print("="*60 + "\n")
    
    # 打印 LATENT_MEAN
    mean_str = ", ".join([f"{x:.6f}" for x in channel_mean])
    print(f"    LATENT_MEAN = [\n        {mean_str}\n    ]\n")
    
    # 打印 LATENT_STD
    std_str = ", ".join([f"{x:.6f}" for x in channel_std])
    print(f"    LATENT_STD = [\n        {std_str}\n    ]\n")
    print("="*60)

if __name__ == "__main__":
    # 替换为你实际的 data.tsv 路径
    YOUR_TSV_PATH = "data/postprocess/data.tsv"
    compute_and_print_stats(YOUR_TSV_PATH)