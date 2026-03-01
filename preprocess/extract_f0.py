import os
import sys
import glob
import torch
import librosa
import logging
import numpy as np
import pandas as pd
import pyworld as pw
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def process_single_file(task, target_dir):
    """
    独立的工作函数，由每个 CPU 核心单独执行
    注意：这里的 task 已经变成了原生的 Python dict，而不是 pandas.Series
    """
    audio_path = task.get('audio_path')
    try:
        # 1. 解析路径
        latent_path = task.get('latent_path', task.get('mel_path'))
        # 生成存放到统一目标文件夹的新路径
        f0_save_path = os.path.join(target_dir, os.path.basename(latent_path).replace('.npy', '_f0.npy'))
        
        # 断点续传：如果已经存在，直接跳过
        if os.path.exists(f0_save_path):
            return True, None

        # ==========================================
        # 2. 从 WAV 中使用 Harvest 算法提取高精度 F0
        # ==========================================
        wav, sr = librosa.load(audio_path, sr=44100)
        
        # 保证 F0 步长 512 * 4 = VAE 步长 2048
        exact_frame_period = (512 / 44100) * 1000 
        f0_high_res, _ = pw.harvest(wav.astype(np.float64), sr, frame_period=exact_frame_period)
        f0_high_res = f0_high_res.astype(np.float32)

        # 获取 VAE 目标长度
        latent_shape = np.load(latent_path, mmap_mode='r').shape
        target_T = latent_shape[1] 

        # ==========================================
        # 3. 安全填充：保护清音区
        # ==========================================
        f0_series = pd.Series(f0_high_res)
        f0_filled = f0_series.replace(0, np.nan).ffill().bfill().fillna(0).values
        
        f0_tensor_filled = torch.from_numpy(f0_filled).unsqueeze(0).unsqueeze(0)
        uv_mask = (torch.from_numpy(f0_high_res) > 0).float().unsqueeze(0).unsqueeze(0)

        # ==========================================
        # 4. 精确降采样：4倍池化
        # ==========================================
        f0_latent = F.avg_pool1d(f0_tensor_filled, kernel_size=4, stride=4)
        uv_latent = F.max_pool1d(uv_mask, kernel_size=4, stride=4)

        # ==========================================
        # 5. 处理微小边缘对齐误差
        # ==========================================
        current_T = f0_latent.shape[2]
        
        if current_T > target_T:
            f0_latent = f0_latent[:, :, :target_T]
            uv_latent = uv_latent[:, :, :target_T]
        elif current_T < target_T:
            pad_len = target_T - current_T
            f0_latent = F.pad(f0_latent, (0, pad_len), mode='replicate')
            uv_latent = F.pad(uv_latent, (0, pad_len), mode='replicate')

        # 最终盖上发声掩码并保存
        f0_final = f0_latent * uv_latent
        np.save(f0_save_path, f0_final.squeeze().numpy())
        
        return True, None

    except Exception as e:
        # 返回失败标志和具体的报错信息
        return False, f"[{audio_path}] 报错: {str(e)}"


def extract_and_align_f0_fast(data_dir, target_dir, max_workers=None):
    tsv_files = glob.glob(os.path.join(data_dir, '*.tsv'))
    if not tsv_files:
        print(f"❌ 未找到任何 .tsv 文件于 {data_dir}")
        return

    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)
        
    print(f"🚀 启动多进程极速模式！正在使用 {max_workers} 个 CPU 核心并行处理...")

    for tsv_path in tsv_files:
        print(f"\n📂 正在处理: {os.path.basename(tsv_path)}")
        df = pd.read_csv(tsv_path, sep='\t', low_memory=False)
        
        success_count = 0
        error_count = 0
        
        # 💡 优化点 1：将 DataFrame 转为原生 dict 列表，大幅降低进程间通信开销
        tasks = df.to_dict('records')
        
        # 💡 优化点 2：剥离 with 语法，手动控制进程池，以便实现强制终止
        executor = ProcessPoolExecutor(max_workers=max_workers)
        futures = []
        
        try:
            # 提交任务
            for task in tasks:
                futures.append(executor.submit(process_single_file, task, target_dir))
            
            # 使用 tqdm 监控
            with tqdm(total=len(tasks), desc="提取 F0") as pbar:
                for future in as_completed(futures):
                    success, error_msg = future.result()
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        logging.error(error_msg)
                    pbar.update(1)
                    
            print(f"✅ {os.path.basename(tsv_path)} 处理完成！成功: {success_count}, 失败: {error_count}")
            
            # 正常完成，优雅关闭
            executor.shutdown(wait=True)

        except KeyboardInterrupt:
            # 💡 优化点 3：捕获 Ctrl+C，暴力叫停
            print("\n🛑 接收到中止信号 (Ctrl+C)！正在紧急终止所有任务...")
            
            # 取消所有尚未分配给子进程的任务 (Python 3.9+)
            for future in futures:
                future.cancel()
            
            # 不等待运行中的任务完成，立即关闭进程池
            executor.shutdown(wait=False, cancel_futures=True)
            
            print("💀 进程池已强制关闭。退出程序。")
            # 使用 os._exit(1) 直接在操作系统层面杀掉主进程，不给 C 扩展死锁的机会
            os._exit(1) 


if __name__ == "__main__":
    YOUR_DATA_DIR = "./data/final" 
    TARGET_DIR = "./data/feat_extract/f0"
    
    os.makedirs(TARGET_DIR, exist_ok=True)
    extract_and_align_f0_fast(YOUR_DATA_DIR, TARGET_DIR, max_workers=16)
