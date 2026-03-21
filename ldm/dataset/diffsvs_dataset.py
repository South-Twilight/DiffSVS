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
PITCH_PAD_ID = 129

class DiffSVSDataset(Dataset):
    """
    提供 MMDiT 所需的两部分输入：
    - audio：audio 模态的 VAE latent [C, T]
    - cond： text 模态的 music score（phn, note dur, note pitch）及训练用字段（dur_gt, f0_gt, spk_id 等）
    collater 后 batch 为 {'audio': [B,C,T], 'cond': {...}}。
    """
    def __init__(
        self, split, data_dir, sample_rate=44100, hop_size=2048,
        latent_crop_len=500, drop_rate=0.1, max_prompt_len=100, **kwargs
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
        self.max_prompt_len = int(max_prompt_len) if max_prompt_len is not None else 100
        
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
            # 在候选 prompt 里随机挑一条做 data augmentation，当前情况下cond_list只有一条
            prompt_latent_path = random.choice(cand_list)
            if isinstance(prompt_latent_path, str) and os.path.exists(prompt_latent_path):
                prompt_latent_z = np.load(prompt_latent_path).astype(np.float32)  # [C, T_p]
                # 限制 prompt latent 的有效长度，避免在 time 维度上过长：
                # - 至多保留 max_prompt_len 帧（默认 100）
                # - 如果更长：随机裁一段长度为 max_prompt_len
                # - 如果更短：保持原长度，不再强行对齐到 target_T
                T_p = prompt_latent_z.shape[1]
                if T_p > self.max_prompt_len:
                    max_start = T_p - self.max_prompt_len
                    start = random.randint(0, max_start)
                    prompt_latent_z = prompt_latent_z[:, start:start + self.max_prompt_len]
        
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


class DDPIndexBatchSampler(Sampler):
    """
    让长度相近的样本组成 batch，并在 DDP 下按 rank 切分。

    与传统 `sum(lengths) <= max_tokens` 不同，
    这里使用更接近真实显存占用的估计：

        estimated_cost = max_len_in_batch * batch_size

    因为 collate 后通常会 pad 到 batch 内最长长度，
    所以真实显存更接近 `B * T_max`，而不是 `sum(T_i)`。

    参数:
        indices: 已经按长度排序好的样本下标列表
        batch_size: fallback 用；如果 max_tokens=None，则按固定 batch_size 组 batch
        num_replicas: DDP world size
        rank: 当前 rank
        shuffle: 是否每个 epoch 打乱 batch 顺序
        seed: shuffle seed
        drop_last: 是否丢弃最后不完整 batch
        max_tokens: 这里表示“估计后的 batch 体积上限”，即 B * T_max 上限
        max_sentences: 单 batch 最大样本数上限
        lengths: 每个样本的长度估计列表
    """

    def __init__(
        self,
        indices,
        batch_size,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        max_tokens: Optional[int] = None,
        max_sentences: Optional[int] = None,
        lengths: Optional[List[int]] = None,
    ) -> None:
        super().__init__(None)

        if num_replicas is None:
            if not dist.is_initialized():
                logger.info("Not in distributed mode")
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_initialized():
                rank = 0
            else:
                rank = dist.get_rank()

        self.indices = list(indices)
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.drop_last = drop_last

        self.max_tokens = max_tokens if max_tokens is not None else None
        self.max_sentences = max_sentences if max_sentences is not None else batch_size
        self.lengths = lengths

        self.epoch = 0
        self._full_batches = self.build_batches()

        logger.info(
            "[DDPIndexBatchSampler] rank=%s num_replicas=%s total_full_batches=%s "
            "batch_size=%s max_tokens=%s max_sentences=%s",
            self.rank,
            self.num_replicas,
            len(self._full_batches),
            self.batch_size,
            self.max_tokens,
            self.max_sentences,
        )

        if self.drop_last and len(self._full_batches) % self.num_replicas != 0:
            keep = (len(self._full_batches) // self.num_replicas) * self.num_replicas
            self._full_batches = self._full_batches[:keep]

        self._update_batches_for_epoch()

        logger.info(
            "[DDPIndexBatchSampler] rank=%s local_batches=%s",
            self.rank,
            len(self.batches),
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._update_batches_for_epoch()

    def __iter__(self) -> Iterator[List[int]]:
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)

    def _update_batches_for_epoch(self):
        """
        对完整 batch 列表 shuffle 后，再按 rank 切分。
        """
        full = list(self._full_batches)

        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch)
            perm = rng.permutation(len(full))
            full = [full[i] for i in perm]

        if len(full) == 0:
            self.batches = []
            return

        # 按 batch 维度进行 DDP 切分：rank r 取 full[r::num_replicas]
        if len(full) > self.num_replicas:
            self.batches = full[self.rank::self.num_replicas]
        else:
            # batch 数少于卡数时，尽量不报错
            self.batches = full[self.rank:self.rank + 1] if self.rank < len(full) else []

    def build_batches(self):
        """
        构造“完整 batch 列表”（未按 rank 切分）。

        关键改动：
        不再按 sum(lengths) 限制，
        改为按：
            max_len_in_batch * batch_size <= max_tokens
        来限制 batch 体积。
        """
        # 1) 动态 batch 模式：使用 max_tokens + lengths
        if self.max_tokens is not None and self.lengths is not None:
            batches = []
            batch = []
            cur_max_len = 0

            max_sent = self.max_sentences if self.max_sentences is not None else self.batch_size
            max_sent = max(1, int(max_sent))

            for index in self.indices:
                t = self._safe_length(index)

                new_bs = len(batch) + 1
                new_max_len = max(cur_max_len, t)
                estimated_cost = new_bs * new_max_len  # 核心：近似 pad 后体积

                # 如果当前 batch 非空，且加入新样本后超限，则先封包
                if len(batch) > 0 and (
                    estimated_cost > self.max_tokens or len(batch) >= max_sent
                ):
                    batches.append(batch)
                    batch = [index]
                    cur_max_len = t
                else:
                    batch.append(index)
                    cur_max_len = new_max_len

            if len(batch) > 0 and not self.drop_last:
                batches.append(batch)
            elif len(batch) > 0 and self.drop_last:
                # drop_last=True 时，只有达到条件的完整 batch 才保留
                # 这里最后残余 batch 丢弃
                pass

            return batches

        # 2) fallback：固定 batch_size
        batches = []
        batch = []
        for index in self.indices:
            batch.append(index)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []

        if len(batch) > 0 and not self.drop_last:
            batches.append(batch)

        return batches

    def _safe_length(self, index: int) -> int:
        """
        安全读取长度，保证 >= 1
        """
        if self.lengths is None:
            return 1

        if index < 0 or index >= len(self.lengths):
            return 1

        t = self.lengths[index]
        if t is None:
            return 1

        try:
            t = int(t)
        except Exception:
            return 1

        return max(t, 1)
