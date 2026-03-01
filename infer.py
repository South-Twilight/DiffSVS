import os
import sys
import csv
import argparse
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import random
import ast

import torch
import torch.nn.functional as F 
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from omegaconf import OmegaConf
from tqdm import tqdm
import soundfile as sf

from ldm.models.diffusion.cfm1_audio_sampler import CFMSampler
from ldm.util import instantiate_from_config

# ==========================================
# 1. 常量与词表
# ==========================================
ph_set = [
    "<SP>", "<AP>", "l_zh", "i_zh", "b_zh", "ie_zh", "m_zh", "ei_zh", "sh_zh",
    "uo_zh", "z_zh", "ai_zh", "j_zh", "ian_zh", "n_zh", "f_zh", "ou_zh", "x_zh",
    "in_zh", "s_zh", "uan_zh", "zh_zh", "en_zh", "iao_zh", "u_zh", "g_zh",
    "an_zh", "d_zh", "e_zh", "van_zh", "v_zh", "ia_zh", "ong_zh", "t_zh",
    "h_zh", "uei_zh", "ao_zh", "ch_zh", "eng_zh", "c_zh", "ang_zh", "ve_zh",
    "iou_zh", "uang_zh", "a_zh", "q_zh", "r_zh", "ing_zh", "iang_zh", "vn_zh",
    "o_zh", "p_zh", "uen_zh", "ua_zh", "k_zh", "iong_zh", "uai_zh", "er_zh"
]
PH_PAD_ID = 59 
PITCH_PAD_ID = 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/diffsvs.yaml')
    parser.add_argument("--ckpt", type=str, required=True, help="模型权重路径")
    parser.add_argument("--manifest_path", type=str, default='data/final/test.tsv')
    parser.add_argument("--singer_txt", type=str, default='data/final/singer.txt')
    parser.add_argument("--ddim_steps", type=int, default=50, help="ODE 积分步数")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--scale", type=float, default=4.0, help="CFG 引导系数")
    parser.add_argument("--scales", type=str, default='')
    parser.add_argument("--save_dir", type=str, default='test_outputs')
    parser.add_argument("--num_gpus", type=int, default=1)
    return parser.parse_args()

def safe_path(path):
    os.makedirs(Path(path).parent, exist_ok=True)
    return path

def normalize_loudness(wav, target_loudness):
    rms = np.sqrt(np.mean(wav ** 2) + 1e-8)
    loudness = 20 * np.log10(rms)
    gain = target_loudness - loudness
    normalized_wav = wav * (10 ** (gain / 20))
    return normalized_wav

# ==========================================
# 2. 推理/评测 专属 Dataset
# ==========================================
class DiffSVSEvalDataset(Dataset):
    def __init__(self, manifest_path, singer_txt_path):
        super().__init__()
        df = pd.read_csv(manifest_path, sep='\t')
        
        # 随机抽取 50 条测试，限制长度防 OOM
        if len(df) > 50:
            df = df.sample(n=50, random_state=42)
        if 'duration' in df.columns:
            df = df[df['duration'] < 20.0]
        
        self.dataset = df.reset_index(drop=True)
        
        # 加载统一的歌手字典
        if os.path.exists(singer_txt_path):
            with open(singer_txt_path, 'r', encoding='utf-8') as f:
                unique_singers = [line.strip() for line in f if line.strip()]
            self.singer_to_id = {singer: idx for idx, singer in enumerate(unique_singers)}
        else:
            self.singer_to_id = {'unknown': 0}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset.iloc[idx]
        
        ph_str_list = ast.literal_eval(data['ph'])
        ep_pitches = ast.literal_eval(data['ep_pitches']) 
        notedurs = ast.literal_eval(data['ep_notedurs'])  
        notetypes = ast.literal_eval(data['ep_types'])    
        
        ph2id = {p: i for i, p in enumerate(ph_set)}
        ph_ids = [ph2id.get(p, PH_PAD_ID) for p in ph_str_list] 
        
        singer_name = data.get('singer', 'unknown')
        if pd.isna(singer_name): singer_name = 'unknown'
        spk_id = self.singer_to_id.get(str(singer_name).strip(), 0)

        item = {
            'audio_path': data['audio_path'],
            'name': data.get('item_name', f'test_{idx}'),
            'ph': torch.tensor(ph_ids, dtype=torch.long),      
            'pitches': torch.tensor(ep_pitches, dtype=torch.long), 
            'notedurs': torch.tensor(notedurs, dtype=torch.float32), 
            'notetypes': torch.tensor(notetypes, dtype=torch.long),  
            'spk_id': spk_id, 
        }
        return item

def eval_collate_fn(batch):
    ph_list = [item['ph'] for item in batch]
    pitches_list = [item['pitches'] for item in batch]
    notedurs_list = [item['notedurs'] for item in batch]
    notetypes_list = [item['notetypes'] for item in batch]
    
    ph_padded = torch.nn.utils.rnn.pad_sequence(ph_list, batch_first=True, padding_value=PH_PAD_ID)
    pitches_padded = torch.nn.utils.rnn.pad_sequence(pitches_list, batch_first=True, padding_value=PITCH_PAD_ID)
    notedurs_padded = torch.nn.utils.rnn.pad_sequence(notedurs_list, batch_first=True, padding_value=0.0)
    notetypes_padded = torch.nn.utils.rnn.pad_sequence(notetypes_list, batch_first=True, padding_value=4)
    
    spk_id = torch.tensor([item['spk_id'] for item in batch], dtype=torch.long)
    names = [item['name'] for item in batch]
    audio_paths = [item['audio_path'] for item in batch]
    
    cond = {
        'ph': ph_padded,
        'pitches': pitches_padded,
        'notedurs': notedurs_padded,
        'notetypes': notetypes_padded,
        'spk_id': spk_id,
        # 🌟 欺骗模型使用传入的 dur_gt，而不是内部自动预测，这是为了 CFG 长度绝对一致
        'infer': False 
    }
    
    return cond, names, audio_paths

# ==========================================
# 3. 主生成逻辑
# ==========================================
def initialize_model(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu')["state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    
    sampler = CFMSampler(model, num_timesteps=1000)
    return sampler

@torch.no_grad()
def gen_song(rank, args):
    if args.num_gpus > 1:
        init_process_group(backend='nccl', init_method="tcp://localhost:54189",
                           world_size=args.num_gpus, rank=rank)

    device = torch.device(f"cuda:{int(rank)}")
    sampler = initialize_model(args.config, args.ckpt, device)
    
    dataset = DiffSVSEvalDataset(args.manifest_path, args.singer_txt)
    ds_sampler = DistributedSampler(dataset, shuffle=False) if args.num_gpus > 1 else None
    loader = DataLoader(dataset, sampler=ds_sampler, collate_fn=eval_collate_fn, batch_size=1)

    scales = [float(s) for s in args.scales.split('-')] if args.scales else [args.scale]
    save_dir = args.save_dir
    loader = tqdm(loader) if rank == 0 else loader
    
    csv_data = {'audio_path': [], 'name': [], 'scale': []}
    
    for batch_idx, (cond, names, audio_paths) in enumerate(loader):
        item_name = names[0]
        gt_path = audio_paths[0]
        
        # 移至设备
        for k, v in cond.items():
            if isinstance(v, torch.Tensor):
                cond[k] = v.to(device)
                
        # ==========================================
        # 🌟 核心：独立预测时长并冻结，保障 CFG
        # ==========================================
        padding_mask = (cond['ph'] == PH_PAD_ID)
        _, pred_dur_log = sampler.model.frontend(
            cond['ph'], cond['notedurs'], cond['pitches'], cond['notetypes'], padding_mask
        )
        dur_pred = torch.clamp(torch.round(torch.exp(pred_dur_log) - 1), min=1).long()
        
        # 强制设为 Ground Truth 时长，骗过 apply_model
        cond['dur_gt'] = dur_pred 
        
        # 计算 Latent 总长度
        latent_length = int(dur_pred.sum().item())
        
        # 构建无条件引导 (Unconditional) - 通常把 spk_id 置零或置信为空特征
        uc_cond = cond.copy()
        uc_cond['spk_id'] = torch.zeros_like(cond['spk_id'])
        
        # 纯噪声起点 (假设 Latent 通道为 128，请根据实际情况修改)
        latent_channels = 128
        shape = [latent_channels, latent_length]
        start_code = torch.randn(args.n_samples, latent_channels, latent_length, device=device)

        # 读取真实音频 (用于对比)
        try:
            gt_wav, _ = sf.read(gt_path)
        except:
            gt_wav = np.zeros((1000,))

        # 开始不同尺度采样
        for scale in scales:
            # 采样 CFG
            samples_ddim, _, f0_pred = sampler.sample_cfg(
                S=args.ddim_steps,
                cond=cond,
                batch_size=args.n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc_cond,
                x_T=start_code
            )
            
            # 🌟 VAE 解码 (取代了以前的 Vocoder)
            # 这里调用第一阶段模型(VAE)将 Latent 解码回音频波形
            wav_preds = sampler.model.decode_first_stage(samples_ddim)
            
            cond_gtcodec_dir = os.path.join(save_dir, f'scale_{scale}')
            
            for wav_idx, wav_tensor in enumerate(wav_preds):
                wav = wav_tensor.squeeze().cpu().numpy()
                
                # 响度归一化并保存预测
                pred_save_path = os.path.join(cond_gtcodec_dir, f"{rank}-{batch_idx:04d}[{wav_idx}]_{item_name}_pred.wav")
                wav = normalize_loudness(wav, -23)
                sf.write(safe_path(pred_save_path), wav, 44100, subtype='PCM_16')

                # 保存 GT 音频用于背靠背盲听
                gt_save_path = os.path.join(cond_gtcodec_dir, f"{rank}-{batch_idx:04d}[{wav_idx}]_{item_name}_gt.wav")
                gt_wav_norm = normalize_loudness(gt_wav, -23)
                sf.write(safe_path(gt_save_path), gt_wav_norm, 44100, subtype='PCM_16')

                csv_data['audio_path'].append(pred_save_path)
                csv_data['name'].append(item_name)
                csv_data['scale'].append(scale)
                
    if rank == 0:
        csv_save_path = os.path.join(save_dir, 'inference_results.csv')
        pd.DataFrame(csv_data).to_csv(csv_save_path, index=False)
        print(f"🎉 推理完成！结果清单已保存至: {csv_save_path}")

if __name__ == '__main__':
    args = parse_args()
    if args.num_gpus > 1:
        mp.spawn(gen_song, nprocs=args.num_gpus, args=(args,))
    else:
        gen_song(0, args=args)
