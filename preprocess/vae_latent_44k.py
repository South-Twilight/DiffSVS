import os
import sys

# 解决跨目录调用 modules 报错
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import math
import logging
import argparse
import pandas as pd
from glob import glob
import numpy as np
from tqdm import tqdm

import torch
import torchaudio
import pyloudnorm as pyln
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.distributed import init_process_group
import torch.multiprocessing as mp

from modules.vae import DiffRhythmVAE

# 强制 Torchaudio 使用更安全的多进程后端 (如果你的环境装了 soundfile)
try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# 2. 数据集加载：TSV_VAE_Dataset
# ==========================================
class TSV_VAE_Dataset(Dataset):
    def __init__(self, tsv_path, target_sr=44100, target_loudness=-23.0) -> None:
        super().__init__()
        if os.path.isdir(tsv_path):
            files = glob(os.path.join(tsv_path, '*.tsv'))
            self.df = pd.concat([pd.read_csv(file, sep='\t') for file in files])
        else:
            self.df = pd.read_csv(tsv_path, sep='\t')
            
        self.audio_paths = self.df['audio_path'].tolist()
        self.target_paths = self.df['mel_path'].tolist() 
        self.target_sr = target_sr
        self.target_loudness = target_loudness  
        
        # 【极其关键的修改】：不要在这里初始化 pyln.Meter()
        self.meter = None  

    def __len__(self):
        return len(self.audio_paths)

    def normalize_loudness(self, wav):
        # 懒加载：确保 meter 只在真正工作的子进程内部被创建，避免多进程 Pickling 导致的 C 指针越界
        if self.meter is None:
            self.meter = pyln.Meter(self.target_sr)

        wav_np = wav.squeeze(0).cpu().numpy()
        if np.isnan(wav_np).any() or np.isinf(wav_np).any():
            return wav, None

        try:
            loudness = self.meter.integrated_loudness(wav_np)
        except ValueError:
            return wav, None

        gain = self.target_loudness - loudness
        max_gain, min_gain = 20.0, -20.0 
        if gain > max_gain or gain < min_gain:
            return wav, None

        normalized_wav_np = pyln.normalize.loudness(wav_np, loudness, self.target_loudness)
        if np.isnan(normalized_wav_np).any() or np.isinf(normalized_wav_np).any():
            return wav, None

        peak = np.abs(normalized_wav_np).max()
        if peak > 1.0:
            normalized_wav_np = normalized_wav_np / peak 

        return torch.tensor(normalized_wav_np).unsqueeze(0), True 

    def __getitem__(self, index):
        skip = 0
        audio_path = self.audio_paths[index]
        target_path = self.target_paths[index] 
        
        try:
            wav, orisr = torchaudio.load(audio_path)
        except Exception as e:
            return audio_path, target_path, torch.zeros((1, 1)), 1, 44100

        if wav.shape[0] != 1:  
            wav = wav.mean(0, keepdim=True)
            
        if orisr != self.target_sr:
            wav = torchaudio.functional.resample(wav, orig_freq=orisr, new_freq=self.target_sr)

        audio_duration = wav.shape[-1] / self.target_sr  
        if audio_duration < 1.0:
            return audio_path, target_path, wav, 1, self.target_sr

        wav, skip_f = self.normalize_loudness(wav)
        if skip_f is None:
            logger.info(f'skip audio path: {audio_path}')
            return audio_path, target_path, wav, 1, self.target_sr

        return audio_path, target_path, wav, skip, self.target_sr

# ==========================================
# 3. 核心特征提取及落盘逻辑
# ==========================================
# ==========================================
# 3. 核心特征提取及落盘逻辑
# ==========================================
def process_audio_by_tsv(rank, args):
    # 【新增】：每个进程自己维护一个跳过清单
    local_skipped_records = [] 

    if args.num_gpus > 1:
        init_process_group(
            backend=args.dist_config['dist_backend'], 
            init_method=args.dist_config['dist_url'],
            world_size=args.dist_config['world_size'] * args.num_gpus, 
            rank=rank
        )

    device = torch.device('cuda:{:d}'.format(rank))
    vae = DiffRhythmVAE(device=device, repo_id=args.vae_repo_id)
    
    dataset = TSV_VAE_Dataset(args.tsv_path, target_sr=vae.sampling_rate)
    sampler = DistributedSampler(dataset, shuffle=False) if args.num_gpus > 1 else None
    
    # 保持 num_workers=0 防崩溃
    loader = DataLoader(dataset, sampler=sampler, batch_size=1, num_workers=0, drop_last=False)
    
    loader = tqdm(loader, desc=f"GPU-{rank}") if rank == 0 else loader

    for batch in loader:
        audio_paths, target_paths, wavs, skip_flags, srs = batch
        
        audio_path = audio_paths[0]
        latent_path = target_paths[0] 
        
        # 1. 如果在 Dataset 阶段就被标记跳过了 (时长太短、读取失败或响度异常爆音)
        if skip_flags.item() == 1:
            local_skipped_records.append(f"{audio_path}\t[跳过原因: 音频读取失败/过短/响度归一化异常]")
            continue
            
        # 如果已经存在则直接跳过 (正常跳过，不计入错误日志)
        if os.path.exists(latent_path):
            continue

        try:
            wav = wavs[0]
            sr = srs.item()
            processed_audio = vae.preprocess_audio(wav, sr)
            latents_raw = vae.encode(processed_audio, chunked=True)
            latent_dist_np = latents_raw.cpu().numpy().squeeze(0)  
            
            os.makedirs(os.path.dirname(latent_path), exist_ok=True)
            np.save(latent_path, latent_dist_np)
            
        except Exception as e:
            logger.error(f"处理音频 {audio_path} 失败: {e}")
            # 2. 如果在 VAE 模型推理阶段崩溃
            local_skipped_records.append(f"{audio_path}\t[跳过原因: VAE推理报错 -> {str(e)}]")

    # 【新增】：运行结束时，把跳过清单落盘
    total_skipped = len(local_skipped_records)
    if total_skipped > 0:
        # 获取 tsv_path 所在的目录，将日志存放在那里
        log_dir = os.path.dirname("./data/feat_extract")
        log_file = os.path.join(log_dir, "log", f"skipped_files_gpu{rank}.txt")
        
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(local_skipped_records))
            
        logger.info(f"⚠️ 进程 {rank} 结束！共跳过 {total_skipped} 个文件。详情已保存至: {log_file}")
    else:
        logger.info(f"🎉 进程 {rank} 完美结束！没有跳过任何文件。")

# ==========================================
# 4. 启动入口
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_path", type=str, required=True, help="输入数据的 TSV 文件路径")
    parser.add_argument("--num_gpus", type=int, default=1, help="使用的 GPU 数量")
    parser.add_argument("--vae_repo_id", type=str, default="ASLP-lab/DiffRhythm-vae", help="HuggingFace 上的 VAE 模型仓库 ID")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    args.dist_config = {
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54189",
        "world_size": 1
    }
    
    logger.info(f"准备提取 VAE 特征，采用极致稳定多卡模式...")
    
    # 【极其关键的修改】：仅仅传入 args，不传任何引发多进程崩溃的对象
    if args.num_gpus > 1:
        mp.spawn(process_audio_by_tsv, nprocs=args.num_gpus, args=(args,))
    else:
        process_audio_by_tsv(0, args=args)

"""
CUDA_VISIBLE_DEVICES=1,2,3,4 python -m preprocess.vae_latent_44k \
    --tsv_path /data7/tyx/DiffSVS/data/preprocess/data.tsv \
    --num_gpus 4
"""