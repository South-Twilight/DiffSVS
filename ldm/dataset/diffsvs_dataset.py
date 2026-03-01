import os
import torch
import numpy as np
import pandas as pd
import glob
import ast
import random
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.distributed as dist
from typing import Optional, Iterator, List

from ldm.dataset.joinaudiodataset_anylen import *

ph_set = [
    "<SP>", "<AP>", "l_zh", "i_zh", "b_zh", "ie_zh", "m_zh", "ei_zh", "sh_zh",
    "uo_zh", "z_zh", "ai_zh", "j_zh", "ian_zh", "n_zh", "f_zh", "ou_zh", "x_zh",
    "in_zh", "s_zh", "uan_zh", "zh_zh", "en_zh", "iao_zh", "u_zh", "g_zh",
    "an_zh", "d_zh", "e_zh", "van_zh", "v_zh", "ia_zh", "ong_zh", "t_zh",
    "h_zh", "uei_zh", "ao_zh", "ch_zh", "eng_zh", "c_zh", "ang_zh", "ve_zh",
    "iou_zh", "uang_zh", "a_zh", "q_zh", "r_zh", "ing_zh", "iang_zh", "vn_zh",
    "o_zh", "p_zh", "uen_zh", "ua_zh", "k_zh", "iong_zh", "uai_zh", "er_zh"
]
# Padding ID
PH_PAD_ID = 59 
PITCH_PAD_ID = 0

class DiffSVSDataset(Dataset):
    def __init__(
        self, split, data_dir, sample_rate=44100, hop_size=2048, 
        latent_crop_len=500, drop_rate=0.1, **kwargs
    ):
        """
        :param data_dir: 存放 tsv 文件和 singer.txt 的目录路径 (如 ./data/final)
        """
        super().__init__()
        self.split = split
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.latent_crop_len = latent_crop_len
        self.drop_rate = drop_rate
        
        # 1. 直接指向对应的 tsv 文件
        manifest_path = os.path.join(data_dir, f'{split}.tsv')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"找不到数据集清单文件: {manifest_path}")
            
        df = pd.read_csv(manifest_path, sep='\t')

        # 对测试集同名文件添加编号后缀防覆盖
        if split == 'test':
            df = self.add_name_num(df)
            
        self.dataset = df
        self.dataset.reset_index(drop=True, inplace=True)
        
        # ==========================================
        # 🌟 核心修改：从固定文件读取歌手列表，保证全集 ID 绝对一致
        # ==========================================
        # 在 Dataset 的 __init__ 里
        singer_file = os.path.join(data_dir, 'singer.txt')
        with open(singer_file, 'r', encoding='utf-8') as f:
            unique_singers = [line.strip() for line in f if line.strip()]
            
        # 因为 'unknown' 在第一行，它自然就是 0
        self.singer_to_id = {singer: idx for idx, singer in enumerate(unique_singers)}
        
        print(f'[{split}] Latent 数据集加载成功: {manifest_path}, 样本数: {len(self.dataset)}, 歌手数: {len(unique_singers)}')

    def add_name_num(self, df):
        name_count_dict = {}
        change = []
        for t in df.itertuples():
            name = getattr(t, 'item_name')
            if name in name_count_dict:
                name_count_dict[name] += 1
            else:
                name_count_dict[name] = 0
            change.append((t.Index, name_count_dict[name]))
            
        df_copy = df.copy()
        for idx, count in change:
            df_copy.loc[idx, 'item_name'] = f"{df_copy.loc[idx, 'item_name']}_{count}"
        return df_copy

    def __getitem__(self, idx):
        # 取出当前行数据
        data = self.dataset.iloc[idx]
        
        # ==========================================
        # 1. 读取并解析新的文本和音符特征 (Token级别)
        # ==========================================
        ph_str_list = ast.literal_eval(data['ph'])
        ph_durs = ast.literal_eval(data['ph_durs'])       # 物理时间 (秒)
        ep_pitches = ast.literal_eval(data['ep_pitches']) # 音高 (0-128)
        notedurs = ast.literal_eval(data['ep_notedurs'])  # 音符时值
        notetypes = ast.literal_eval(data['ep_types'])    # 音符类型
        
        # 映射 Phoneme ID (如果找不到就给 Padding ID 59)
        ph2id = {p: i for i, p in enumerate(ph_set)}
        ph_ids = [ph2id.get(p, PH_PAD_ID) for p in ph_str_list] 
        
        # 🌟 核心：计算 Latent 空间的 dur_gt (利用物理时间精准映射到 46ms 的 Latent 帧)
        cum_durs = np.cumsum(ph_durs) * (self.sample_rate / self.hop_size)
        cum_frames = np.round(cum_durs).astype(int)
        dur_gt = np.diff(np.insert(cum_frames, 0, 0)) # 长度为 [L] 的每个 token 占用的帧数
        
        # ==========================================
        # 2. 读取音频 Latent 和高分辨率 F0
        # ==========================================
        # 兼容你的表头：如果有 latent_path 就读，没有就默认读 mel_path
        latent_file_path = data.get('latent_path', data.get('mel_path'))
        latent_z = np.load(latent_file_path).astype(np.float32) # [C, T_lat]
        
        # 读取f0目录下的高频 f0
        f0_path = latent_file_path.replace('latent', 'f0').replace('.npy', '_f0.npy')
        f0_high_res = np.load(f0_path).astype(np.float32) 
        
        # 将 F0 下采样对齐到 Latent 分辨率
        target_T = latent_z.shape[1]
        f0_tensor = torch.from_numpy(f0_high_res).unsqueeze(0).unsqueeze(0)
        f0_latent = F.interpolate(f0_tensor, size=target_T, mode='linear', align_corners=False).squeeze()
        f0_latent = f0_latent.numpy() # 此时 F0 长度也变成了 T_lat
        
        # ==========================================
        # 3. 安全提取歌手 ID (把未知和空值统统打入 0 号冷宫)
        # ==========================================
        singer_name = data['singer'] if 'singer' in data else 'unknown'
        # 防御 Pandas 读空值变成 NaN (float)
        if pd.isna(singer_name):
            singer_name = 'unknown'
            
        # 查表获取 ID，查不到（比如新歌手）就强制赋 0
        spk_id = self.singer_to_id.get(str(singer_name).strip(), 0)
        
        # ==========================================
        # 4. 同步裁剪逻辑 (Token与Frame必须严格绑定)
        # ==========================================
        if target_T > self.latent_crop_len:
            # 随机找一个 Token 的起始位置作为裁剪起点
            max_start_token = max(0, len(dur_gt) - 20)
            start_token_idx = random.randint(0, max_start_token)
            
            # 计算对应的 Frame 起点
            start_frame = cum_frames[start_token_idx] - dur_gt[start_token_idx]
            end_frame = start_frame + self.latent_crop_len
            
            # 防御性逻辑：如果 end_frame 算出来比原音频还要长，就把窗口往左移
            if end_frame > target_T:
                end_frame = target_T
                start_frame = max(0, end_frame - self.latent_crop_len)
                # 重新寻找对应的 start_token_idx
                start_token_idx = np.searchsorted(cum_frames, start_frame, side='right')
                if start_token_idx >= len(dur_gt):
                    start_token_idx = len(dur_gt) - 1
            
            # 裁剪 Frame 级别数据
            latent_z = latent_z[:, start_frame:end_frame]
            f0_latent = f0_latent[start_frame:end_frame]
            
            # 找到 end_frame 落在哪一个 Token 上
            end_token_idx = np.searchsorted(cum_frames, end_frame)
            
            # 裁剪 Token 级别数据
            ph_ids = ph_ids[start_token_idx : end_token_idx]
            ep_pitches = ep_pitches[start_token_idx : end_token_idx]
            notedurs = notedurs[start_token_idx : end_token_idx]
            notetypes = notetypes[start_token_idx : end_token_idx]
            dur_gt = dur_gt[start_token_idx : end_token_idx]
            
            # 修正最后一个 Token 的时长，防止拉伸复原时超出 latent_crop_len 报错
            if len(dur_gt) > 0:
                actual_frames_covered = np.sum(dur_gt[:-1])
                dur_gt[-1] = self.latent_crop_len - actual_frames_covered
                # 万一出现极端裁切导致负数，强制最小为 1 帧
                dur_gt[-1] = max(1, dur_gt[-1])

        # ==========================================
        # 5. 打包返回标准的 Tensor 字典
        # ==========================================
        item = {
            'latent': torch.from_numpy(latent_z),              
            'f0_gt': torch.from_numpy(f0_latent).unsqueeze(0), 
            'ph': torch.tensor(ph_ids, dtype=torch.long),      
            'pitches': torch.tensor(ep_pitches, dtype=torch.long), 
            'notedurs': torch.tensor(notedurs, dtype=torch.float32), 
            'notetypes': torch.tensor(notetypes, dtype=torch.long),  
            'dur_gt': torch.tensor(dur_gt, dtype=torch.long),  
            'spk_id': spk_id, 
        }
        
        if self.split == 'test':
            item['f_name'] = data['item_name']
            
        return item

    def __len__(self):
        return len(self.dataset)

# YAML 对应的 Wrapper
class DiffSVSTrainDataset(DiffSVSDataset):
    def __init__(self, dataset_cfg):
        super().__init__('train', **dataset_cfg)

class DiffSVSValidationDataset(DiffSVSDataset):
    def __init__(self, dataset_cfg):
        super().__init__('valid', **dataset_cfg)

class DiffSVSTestDataset(DiffSVSDataset):
    def __init__(self, dataset_cfg):
        super().__init__('test', **dataset_cfg)

# ==========================================
# 专属的 Collate Function
# ==========================================
def diffsvs_collate_fn(batch):
    """负责将批次内长短不一的样本动态 Pad 对齐"""
    ph_list = [item['ph'] for item in batch]
    pitches_list = [item['pitches'] for item in batch]
    notedurs_list = [item['notedurs'] for item in batch]
    notetypes_list = [item['notetypes'] for item in batch]
    dur_gt_list = [item['dur_gt'] for item in batch]
    
    latent_list = [item['latent'].transpose(0, 1) for item in batch] # [T_lat, C]
    f0_list = [item['f0_gt'].transpose(0, 1) for item in batch]      # [T_lat, 1]
    
    # 动态 Pad Token (L)
    ph_padded = torch.nn.utils.rnn.pad_sequence(ph_list, batch_first=True, padding_value=PH_PAD_ID)
    pitches_padded = torch.nn.utils.rnn.pad_sequence(pitches_list, batch_first=True, padding_value=PITCH_PAD_ID)
    notedurs_padded = torch.nn.utils.rnn.pad_sequence(notedurs_list, batch_first=True, padding_value=0.0)
    notetypes_padded = torch.nn.utils.rnn.pad_sequence(notetypes_list, batch_first=True, padding_value=4)
    dur_gt_padded = torch.nn.utils.rnn.pad_sequence(dur_gt_list, batch_first=True, padding_value=0)
    
    # 动态 Pad Frame (T)
    latent_padded = torch.nn.utils.rnn.pad_sequence(latent_list, batch_first=True, padding_value=0.0) 
    latent_padded = latent_padded.transpose(1, 2) # [B, C, T_lat]
    
    f0_padded = torch.nn.utils.rnn.pad_sequence(f0_list, batch_first=True, padding_value=0.0) 
    f0_padded = f0_padded.transpose(1, 2) # [B, 1, T_lat]
    
    spk_id = torch.tensor([item['spk_id'] for item in batch], dtype=torch.long)
    
    cond = {
        'ph': ph_padded,
        'pitches': pitches_padded,
        'notedurs': notedurs_padded,
        'notetypes': notetypes_padded,
        'dur_gt': dur_gt_padded,
        'f0_gt': f0_padded,
        'spk_id': spk_id,
        'infer': False
    }
    
    return {
        'audio': latent_padded, 
        'cond': cond           
    }


class DDPIndexBatchSampler(Sampler):    # 让长度相似的音频的indices合到一个batch中以避免过长的pad
    def __init__(self, indices, batch_size, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, max_tokens=80000) -> None:
        
        if num_replicas is None:
            if not dist.is_initialized():
                # raise RuntimeError("Requires distributed package to be available")
                print("Not in distributed mode")
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_initialized():
                # raise RuntimeError("Requires distributed package to be available")
                rank = 0
            else:
                rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.indices = indices
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.batches = self.build_batches()
        print(f"rank: {self.rank}, batches_num {len(self.batches)}")
        # If the dataset length is evenly divisible by replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        # print(num_replicas, len(self.batches))
        
        if self.drop_last and len(self.batches) % self.num_replicas != 0:
            self.batches = self.batches[:len(self.batches)//self.num_replicas*self.num_replicas]
        if len(self.batches) > self.num_replicas:
            self.batches = self.batches[self.rank::self.num_replicas]
        else: # may happen in sanity checking
            self.batches = [self.batches[0]]
        print(f"after split batches_num {len(self.batches)}")
        self.shuffle = shuffle
        if self.shuffle:
            self.batches = np.random.permutation(self.batches)
        self.seed = seed

    def set_epoch(self,epoch):
        self.epoch = epoch
        if self.shuffle:
            np.random.seed(self.seed+self.epoch)
            self.batches = np.random.permutation(self.batches)

    def build_batches(self):
        batches, batch = [], []
        for index in self.indices:
            batch.append(index)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []
        if not self.drop_last and len(batch) > 0:
            batches.append(batch)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        return len(self.batches)
