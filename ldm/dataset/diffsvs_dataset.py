import os
import logging
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

logger = logging.getLogger(__name__)

phn_set = [
    "<PAD>", "<SP>", "<AP>", "l_zh", "i_zh", "b_zh", "ie_zh", "m_zh", "ei_zh", "sh_zh",
    "uo_zh", "z_zh", "ai_zh", "j_zh", "ian_zh", "n_zh", "f_zh", "ou_zh", "x_zh",
    "in_zh", "s_zh", "uan_zh", "zh_zh", "en_zh", "iao_zh", "u_zh", "g_zh",
    "an_zh", "d_zh", "e_zh", "van_zh", "v_zh", "ia_zh", "ong_zh", "t_zh",
    "h_zh", "uei_zh", "ao_zh", "ch_zh", "eng_zh", "c_zh", "ang_zh", "ve_zh",
    "iou_zh", "uang_zh", "a_zh", "q_zh", "r_zh", "ing_zh", "iang_zh", "vn_zh",
    "o_zh", "p_zh", "uen_zh", "ua_zh", "k_zh", "iong_zh", "uai_zh", "er_zh"
]
# Padding ID
PHN_PAD_ID = 0 
PITCH_PAD_ID = 0

LATENT_MEAN = [
    0.012353, -0.344221, 0.054542, -0.023620, 0.098828, -0.436021, 0.135209, 0.172249,
    -0.595445, -0.023968, 0.142585, -0.120974, -0.422959, 0.557496, 0.261246, -0.053289,
    -1.143898, -0.110539, -0.512597, 0.225681, -0.417278, -0.171827, 0.010333, -0.212423,
    0.106318, -0.353358, -0.370434, -0.148029, -0.127178, 0.084678, -0.491690, 0.103924,
    -0.116196, -0.121015, -0.076987, -0.195620, 0.085124, -0.220690, 0.458005, 0.245501,
    0.120026, 0.001371, -0.546904, -0.170008, -0.042606, 0.207365, -0.090883, 0.356523,
    0.910243, -0.202829, -0.014405, 0.088774, 0.105067, -0.629098, 0.025747, 0.418244,
    -0.008324, -0.036068, -0.121886, 0.279945, -0.145816, 0.035409, -0.271826, -0.019726,
    -2.456662, -2.423953, -2.456620, -2.553571, -2.692767, -2.479573, -2.413196, -2.948589,
    -2.718184, -2.508078, -2.517597, -3.013349, -2.655125, -3.638837, -2.517074, -2.326965, 
    -3.818206, -2.518479, -2.662763, -2.440180, -2.499040, -2.607668, -2.423820, -2.474048, 
    -2.408190, -2.479893, -2.885017, -2.462518, -2.534864, -2.353198, -2.666671, -2.578288, 
    -2.707603, -3.312835, -2.502359, -2.437055, -2.461510, -2.560955, -2.495958, -2.657370, 
    -2.385587, -2.420916, -4.672287, -3.081646, -2.467169, -2.989388, -2.403907, -2.645048,
    -3.390251, -2.691507, -2.501164, -2.488224, -2.392056, -4.631340, -2.515375, -2.587672, 
    -2.342592, -2.791153, -2.554764, -2.431936, -2.507150, -2.448552, -2.504414, -2.506529
]

LATENT_STD = [
    0.948583, 0.984568, 1.004998, 0.923014, 0.905033, 1.067618, 0.974933, 0.879994, 0.892793, 
    1.024834, 0.937909, 0.796539, 0.995078, 0.930362, 1.060689, 1.056119, 0.875448, 1.026626, 
    1.012141, 0.951827, 0.972708, 1.081410, 0.928428, 1.093917, 1.001015, 0.973961, 1.005739, 
    1.021581, 1.013210, 1.143471, 0.961724, 1.040090, 0.935791, 0.819157, 0.994036, 0.929813, 
    1.012691, 0.990097, 0.937079, 0.923634, 0.915998, 1.008420, 0.971177, 0.881371, 1.049812, 
    0.783271, 1.011007, 1.104984, 0.586922, 0.992643, 0.955493, 0.971456, 0.981908, 1.260328, 
    0.967143, 1.007262, 1.006943, 0.810648, 0.984147, 0.997785, 0.964443, 0.979036, 1.013533, 
    0.908762, 0.536043, 0.515533, 0.524190, 0.522232, 0.473628, 0.534620, 0.539847, 0.474005, 
    0.501741, 0.493244, 0.485631, 0.455617, 0.506798, 0.458490, 0.551780, 0.518085, 0.361584, 
    0.511495, 0.483305, 0.526432, 0.519139, 0.484076, 0.550910, 0.533316, 0.513402, 0.499242, 
    0.444659, 0.524800, 0.496588, 0.529034, 0.504057, 0.508125, 0.525918, 0.379645, 0.552183, 
    0.524201, 0.542009, 0.464389, 0.525243, 0.497123, 0.529890, 0.551786, 0.267760, 0.417225, 
    0.510800, 0.406938, 0.514479, 0.490125, 0.450825, 0.549976, 0.503247, 0.521817, 0.525538, 
    0.502243, 0.533675, 0.537486, 0.540801, 0.457928, 0.521556, 0.510297, 0.517130, 0.518248, 
    0.504208, 0.519711
]

# 转为 numpy 便于广播归一化：latent 形状 [C, T]，保持 mean/std 为 [C, 1]
_LATENT_MEAN = np.array(LATENT_MEAN, dtype=np.float32).reshape(-1, 1)
_LATENT_STD = np.array(LATENT_STD, dtype=np.float32).reshape(-1, 1)

# 最小 std，防止除零
_LATENT_STD_EPS = np.maximum(_LATENT_STD, 1e-6)


# def normalize_latent(latent, mean=None, std=None):
#     """
#     对 latent 做通道维归一化：(x - mean) / std。
#     :param latent: np.ndarray 或 torch.Tensor，形状 [C, T] 或 [B, C, T]
#     :param mean: 可选，形状 [C] 或 [C, 1]，默认用 LATENT_MEAN
#     :param std:  可选，形状 [C] 或 [C, 1]，默认用 LATENT_STD（内部会加 eps 防除零）
#     :return: 与输入同类型、同形状的归一化结果
#     """
#     is_torch = isinstance(latent, torch.Tensor)
#     if mean is None:
#         mean = _LATENT_MEAN
#     if std is None:
#         std = _LATENT_STD_EPS
#     if is_torch:
#         if not isinstance(mean, torch.Tensor):
#             mean = torch.from_numpy(np.asarray(mean, dtype=np.float32)).to(latent.device)
#         if not isinstance(std, torch.Tensor):
#             std = torch.from_numpy(np.asarray(std, dtype=np.float32)).to(latent.device)
#         # 保证通道维可广播，例如 [C, 1] 或 [1, C, 1]
#         if mean.dim() == 1:
#             mean = mean.view(-1, 1)
#         if std.dim() == 1:
#             std = std.view(-1, 1)
#         return (latent - mean) / std
#     else:
#         mean = np.asarray(mean, dtype=np.float32)
#         std = np.asarray(std, dtype=np.float32)
#         if mean.ndim == 1:
#             mean = mean.reshape(-1, 1)
#         if std.ndim == 1:
#             std = np.maximum(std.reshape(-1, 1), 1e-6)
#         else:
#             std = np.maximum(std, 1e-6)
#         return (latent - mean) / std


# def denormalize_latent(latent, mean=None, std=None):
#     """
#     对归一化后的 latent 做反归一化：x * std + mean。
#     :param latent: np.ndarray 或 torch.Tensor，形状 [C, T] 或 [B, C, T]
#     :param mean: 可选，默认 LATENT_MEAN
#     :param std:  可选，默认 LATENT_STD（与 normalize 一致，不含 eps）
#     :return: 与输入同类型、同形状的反归一化结果
#     """
#     is_torch = isinstance(latent, torch.Tensor)
#     if mean is None:
#         mean = _LATENT_MEAN
#     if std is None:
#         std = _LATENT_STD
#     if is_torch:
#         if not isinstance(mean, torch.Tensor):
#             mean = torch.from_numpy(np.asarray(mean, dtype=np.float32)).to(latent.device)
#         if not isinstance(std, torch.Tensor):
#             std = torch.from_numpy(np.asarray(std, dtype=np.float32)).to(latent.device)
#         if mean.dim() == 1:
#             mean = mean.view(-1, 1)
#         if std.dim() == 1:
#             std = std.view(-1, 1)
#         return latent * std + mean
#     else:
#         mean = np.asarray(mean, dtype=np.float32)
#         std = np.asarray(std, dtype=np.float32)
#         if mean.ndim == 1:
#             mean = mean.reshape(-1, 1)
#         if std.ndim == 1:
#             std = std.reshape(-1, 1)
#         return latent * std + mean


class DiffSVSDataset(Dataset):
    """
    提供 MMDiT 所需的两部分输入：
    - audio：audio 模态的 VAE latent [C, T]
    - cond： text 模态的 music score（phn, note dur, note pitch）及训练用字段（dur_gt, f0_gt, spk_id 等）
    collater 后 batch 为 {'audio': [B,C,T], 'cond': {...}}。
    """
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

        # 预先根据 manifest 的 duration（秒）估算 latent 帧长，用于 max_tokens 动态组 batch
        # 注意：本数据集在 __getitem__ 中会裁剪到 latent_crop_len，因此这里也做同样的截断，
        # 使 sampler 的 tokens 估计与实际 padding/显存更一致。
        if "duration" in self.dataset.columns:
            dur_sec = self.dataset["duration"].astype(np.float32).to_numpy()
            frames = np.round(dur_sec * (self.sample_rate / self.hop_size)).astype(np.int64)
            if self.latent_crop_len is not None and self.latent_crop_len > 0:
                frames = np.minimum(frames, int(self.latent_crop_len))
            frames = np.maximum(frames, 1)
            self.lengths = frames.tolist()
        else:
            # 兜底：没有 duration 列时退化为固定长度
            self.lengths = [int(self.latent_crop_len) if self.latent_crop_len else 1] * len(self.dataset)
        
        # ==========================================
        # 🌟 核心修改：从固定文件读取歌手列表，保证全集 ID 绝对一致
        # ==========================================
        # 在 Dataset 的 __init__ 里
        singer_file = os.path.join(data_dir, 'singer.txt')
        with open(singer_file, 'r', encoding='utf-8') as f:
            unique_singers = [line.strip() for line in f if line.strip()]
            
        # 因为 'unknown' 在第一行，它自然就是 0
        self.singer_to_id = {singer: idx for idx, singer in enumerate(unique_singers)}
        
        logger.info('[%s] Latent 数据集加载成功: %s, 样本数: %s, 歌手数: %s', split, manifest_path, len(self.dataset), len(unique_singers))

    def ordered_indices(self):
        index2dur = self.dataset[['duration']].sort_values(by='duration')
        return list(index2dur.index)

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
        phn_str_list = ast.literal_eval(data['ph'])
        phn_durs = ast.literal_eval(data['ph_durs'])       # 物理时间 (秒)
        ep_pitches = ast.literal_eval(data['ep_pitches']) # 音高 (0-128)
        notedurs = ast.literal_eval(data['ep_notedurs'])  # 音符时值
        notetypes = ast.literal_eval(data['ep_types'])    # 音符类型
        
        # 映射 Phoneme ID (如果找不到则用 PAD ID，与 PHN_PAD_ID 一致)
        ph2id = {p: i for i, p in enumerate(phn_set)}
        phn_ids = [ph2id.get(p, PHN_PAD_ID) for p in phn_str_list] 
        
        # 用 sample_rate / hop_size 将物理时长（秒）转为 latent 帧数
        cum_durs = np.cumsum(phn_durs) * (self.sample_rate / self.hop_size)
        cum_frames = np.round(cum_durs).astype(int)
        dur_gt = np.diff(np.insert(cum_frames, 0, 0))
        # round 可能导致某音素帧数为 0（如 cum 从 0.2→0.4 都 round 成 0），clamp 到至少 1 避免 NaN/空序列
        dur_gt = np.maximum(dur_gt, 1)
        
        # ==========================================
        # 2. 读取目标音频 Latent 和高分辨率 F0
        # ==========================================
        # 兼容你的表头：如果有 latent_path 就读，没有就默认读 mel_path
        latent_file_path = data.get('latent_path', data.get('mel_path'))
        latent_z = np.load(latent_file_path).astype(np.float32)  # [C, T_lat]
        # latent_z = normalize_latent(latent_z)
        
        # 读取f0目录下的高频 f0
        f0_path = latent_file_path.replace('latent', 'f0').replace('.npy', '_f0.npy')
        f0_high_res = np.load(f0_path).astype(np.float32) 
        
        # 将 F0 下采样对齐到 Latent 分辨率
        target_T = latent_z.shape[1]
        f0_tensor = torch.from_numpy(f0_high_res).unsqueeze(0).unsqueeze(0)
        f0_latent = F.interpolate(f0_tensor, size=target_T, mode='linear', align_corners=False).squeeze()
        f0_latent = f0_latent.numpy() # 此时 F0 长度也变成了 T_lat

        # ==========================================
        # 2.1 读取 prompt latent（可选）
        # ==========================================
        prompt_latent_z = None
        prompt_latent_path = None
        # 优先使用更显式的 prompt_latent_paths，其次退回 prompt_mel_paths
        if 'prompt_latent_paths' in data and isinstance(data['prompt_latent_paths'], str):
            try:
                cand_list = ast.literal_eval(data['prompt_latent_paths'])
            except (ValueError, SyntaxError):
                cand_list = []
        elif 'prompt_mel_paths' in data and isinstance(data['prompt_mel_paths'], str):
            try:
                cand_list = ast.literal_eval(data['prompt_mel_paths'])
            except (ValueError, SyntaxError):
                cand_list = []
        else:
            cand_list = []

        if isinstance(cand_list, list) and len(cand_list) > 0:
            # 在候选 prompt 里随机挑一条做 data augmentation
            prompt_latent_path = random.choice(cand_list)
            if isinstance(prompt_latent_path, str) and os.path.exists(prompt_latent_path):
                prompt_latent_z = np.load(prompt_latent_path).astype(np.float32)  # [C, T_p]
                # 为了后续模型使用方便，这里将 prompt 在帧维上对齐到目标 latent 的长度：
                # - 如果更长：随机裁一段与 target_T 等长
                # - 如果更短：右侧 zero-pad 到 target_T
                T_p = prompt_latent_z.shape[1]
                if T_p > target_T:
                    max_start = T_p - target_T
                    start = random.randint(0, max_start)
                    prompt_latent_z = prompt_latent_z[:, start:start+target_T]
                elif T_p < target_T:
                    pad_width = target_T - T_p
                    prompt_latent_z = np.pad(
                        prompt_latent_z,
                        ((0, 0), (0, pad_width)),
                        mode='constant',
                        constant_values=0.0,
                    )
        
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
            # prompt 与 target 同步裁剪，保证长度一致且 backbone 内 T_total 不超过 RoPE max_len
            if prompt_latent_z is not None and prompt_latent_z.shape[1] >= end_frame:
                prompt_latent_z = prompt_latent_z[:, start_frame:end_frame]

            # 找到 end_frame 落在哪一个 Token 上
            end_token_idx = np.searchsorted(cum_frames, end_frame)
            
            # 裁剪 Token 级别数据
            phn_ids = phn_ids[start_token_idx : end_token_idx]
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
        # 5. 打包返回：audio（audio 模态 latent）+ prompt（可选）+ cond（text 模态乐谱 + 训练用字段）
        #    供 MMDiT 使用：audio 为 audio 模态输入，cond 为 text 模态（music score）
        # ==========================================
        item = {
            # audio 模态：VAE latent [C, T]，扩散目标 / 条件输入
            'latent': torch.from_numpy(latent_z),
            # 训练用
            'f0_gt': torch.from_numpy(f0_latent).unsqueeze(0),
            # text 模态（music score）：phn、note dur、note pitch
            'phn': torch.tensor(phn_ids, dtype=torch.long),
            'pitches': torch.tensor(ep_pitches, dtype=torch.long),
            'notedurs': torch.tensor(notedurs, dtype=torch.float32),
            'notetypes': torch.tensor(notetypes, dtype=torch.long),
            'dur_gt': torch.tensor(dur_gt, dtype=torch.long),
            'spk_id': spk_id,
            # 方便推理脚本在 teacher forcing 模式下直接读取 GT latent
            'latent_path': latent_file_path,
        }
        # prompt latent（如果存在）一并返回，后续 collater 会自动打包
        if prompt_latent_z is not None:
            item['prompt_latent'] = torch.from_numpy(prompt_latent_z)
            item['prompt_latent_path'] = prompt_latent_path
        
        if self.split == 'test':
            item['f_name'] = data['item_name']
            
        return item

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def collater(batch):
        """
        将 batch 整理为 MMDiT 输入格式：
        - audio: [B, C, T] 音频 latent（audio 模态），C=latent_channels
        - cond:  dict，text 模态的 music score（phn, note dur, note pitch）+ 训练用字段
        """
        phn_list = [item['phn'] for item in batch]
        pitches_list = [item['pitches'] for item in batch]
        notedurs_list = [item['notedurs'] for item in batch]
        notetypes_list = [item['notetypes'] for item in batch]
        dur_gt_list = [item['dur_gt'] for item in batch]

        latent_list = [item['latent'].transpose(0, 1) for item in batch]  # [T_lat, C]
        f0_list = [item['f0_gt'].transpose(0, 1) for item in batch]       # [T_lat, 1]

        # 可选：prompt latent，形状与 latent 一致 [C, T]，这里同样转成 [T, C]
        has_prompt = 'prompt_latent' in batch[0]
        prompt_list = None
        if has_prompt:
            prompt_list = []
            for item in batch:
                if 'prompt_latent' in item:
                    prompt_list.append(item['prompt_latent'].transpose(0, 1))  # [T_p, C]
                else:
                    # 兜底：若个别样本缺 prompt，则用全零占位，长度对齐该样本 latent
                    T = item['latent'].shape[1]
                    C = item['latent'].shape[0]
                    prompt_list.append(torch.zeros(T, C, dtype=item['latent'].dtype))

        # Pad Token 维 (L)：music score
        phn_padded = torch.nn.utils.rnn.pad_sequence(phn_list, batch_first=True, padding_value=PHN_PAD_ID)
        pitches_padded = torch.nn.utils.rnn.pad_sequence(pitches_list, batch_first=True, padding_value=PITCH_PAD_ID)
        notedurs_padded = torch.nn.utils.rnn.pad_sequence(notedurs_list, batch_first=True, padding_value=0.0)
        notetypes_padded = torch.nn.utils.rnn.pad_sequence(notetypes_list, batch_first=True, padding_value=4)
        dur_gt_padded = torch.nn.utils.rnn.pad_sequence(dur_gt_list, batch_first=True, padding_value=0)

        # Pad Frame 维 (T)：audio latent
        latent_padded = torch.nn.utils.rnn.pad_sequence(latent_list, batch_first=True, padding_value=0.0)
        latent_padded = latent_padded.transpose(1, 2)  # [B, C, T_lat]

        f0_padded = torch.nn.utils.rnn.pad_sequence(f0_list, batch_first=True, padding_value=0.0)
        f0_padded = f0_padded.transpose(1, 2)  # [B, 1, T_lat]

        prompt_padded = None
        if has_prompt and prompt_list is not None:
            prompt_padded = torch.nn.utils.rnn.pad_sequence(prompt_list, batch_first=True, padding_value=0.0)
            prompt_padded = prompt_padded.transpose(1, 2)  # [B, C, T_p]

        spk_id = torch.tensor([item['spk_id'] for item in batch], dtype=torch.long)

        # cond = text 模态（music score: phn, note dur, note pitch）+ 训练辅助
        cond = {
            'phn': phn_padded,
            'pitches': pitches_padded,
            'notedurs': notedurs_padded,
            'notetypes': notetypes_padded,
            'dur_gt': dur_gt_padded,
            'f0_gt': f0_padded,
            'spk_id': spk_id,
            'infer': False
        }

        batch_out = {
            'audio': latent_padded,
            'cond': cond
        }
        # 保持向后兼容：仅当存在 prompt 时才添加该键
        if prompt_padded is not None:
            batch_out['prompt'] = prompt_padded

        return batch_out


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


class DDPIndexBatchSampler(Sampler):    # 让长度相似的音频的indices合到一个batch中以避免过长的pad
    def __init__(self, indices, batch_size, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False,
                 max_tokens: Optional[int] = None, max_sentences: Optional[int] = None,
                 lengths: Optional[List[int]] = None) -> None:
        
        if num_replicas is None:
            if not dist.is_initialized():
                # raise RuntimeError("Requires distributed package to be available")
                logger.info("Not in distributed mode")
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
        self.shuffle = shuffle
        self.seed = seed
        self.max_tokens = int(max_tokens) if max_tokens is not None else None
        self.max_sentences = int(max_sentences) if max_sentences is not None else None
        self.lengths = lengths

        # 保留完整 batch 列表，用于 set_epoch 时重新 shuffle + 分区
        self._full_batches = self.build_batches()
        logger.info("rank: %s, total batches_num %s", self.rank, len(self._full_batches))
        if self.drop_last and len(self._full_batches) % self.num_replicas != 0:
            self._full_batches = self._full_batches[:len(self._full_batches)//self.num_replicas*self.num_replicas]
        self._update_batches_for_epoch()
        logger.info("after split batches_num %s", len(self.batches))

    def _update_batches_for_epoch(self):
        """对完整 batch 列表 shuffle 后按 rank 分区，保证每 epoch 各卡拿到不同数据"""
        full = list(self._full_batches)
        if self.shuffle:
            np.random.seed(self.seed + self.epoch)
            idx = np.random.permutation(len(full))
            full = [full[i] for i in idx]
        if len(full) > self.num_replicas:
            self.batches = full[self.rank::self.num_replicas]
        else:
            self.batches = [full[0]] if full else []

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._update_batches_for_epoch()

    def build_batches(self):
        # 1) max_tokens 模式：按 latent 帧长累计到阈值切 batch
        if self.max_tokens is not None and self.lengths is not None:
            batches: List[List[int]] = []
            batch: List[int] = []
            tok = 0
            max_sent = self.max_sentences if self.max_sentences is not None else self.batch_size

            for index in self.indices:
                t = int(self.lengths[index]) if index < len(self.lengths) else 1
                t = max(t, 1)
                # 如果加入当前样本会超 token 或超句子数，则先封包当前 batch
                if len(batch) > 0 and ((tok + t) > self.max_tokens or (len(batch) >= max_sent)):
                    batches.append(batch)
                    batch = []
                    tok = 0
                # 单样本就超过 max_tokens：也要单独成 batch，避免死循环
                batch.append(index)
                tok += t

            if len(batch) > 0 and not self.drop_last:
                batches.append(batch)
            return batches

        # 2) 退化：按固定 batch_size 切 batch
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
