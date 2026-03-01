import os
import sys
import numpy as np
import torch
import logging
import pandas as pd
import glob
import ast

logger = logging.getLogger(f'main.{__name__}')

sys.path.insert(0, '.')  # nopep8

ph_set = [
  "<SP>",
  "<AP>",
  "l_zh",
  "i_zh",
  "b_zh",
  "ie_zh",
  "m_zh",
  "ei_zh",
  "sh_zh",
  "uo_zh",
  "z_zh",
  "ai_zh",
  "j_zh",
  "ian_zh",
  "n_zh",
  "f_zh",
  "ou_zh",
  "x_zh",
  "in_zh",
  "s_zh",
  "uan_zh",
  "zh_zh",
  "en_zh",
  "iao_zh",
  "u_zh",
  "g_zh",
  "an_zh",
  "d_zh",
  "e_zh",
  "van_zh",
  "v_zh",
  "ia_zh",
  "ong_zh",
  "t_zh",
  "h_zh",
  "uei_zh",
  "ao_zh",
  "ch_zh",
  "eng_zh",
  "c_zh",
  "ang_zh",
  "ve_zh",
  "iou_zh",
  "uang_zh",
  "a_zh",
  "q_zh",
  "r_zh",
  "ing_zh",
  "iang_zh",
  "vn_zh",
  "o_zh",
  "p_zh",
  "uen_zh",
  "ua_zh",
  "k_zh",
  "iong_zh",
  "uai_zh",
  "er_zh"
]

class DurationDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        split, data_dir, max_ph_len=150, latent_downsample_factor=1, drop=0,
        **kwargs
    ):
        """
        :param split: 'train', 'valid', 或 'test'
        :param data_dir: 存放 tsv 文件的目录路径 (如 ./data/final)
        :param max_ph_len: 最大音素序列长度
        :param latent_downsample_factor: 时长缩放系数 (匹配 VAE 时间轴下采样)
        """
        super().__init__()
        self.split = split
        self.max_ph_len = max_ph_len
        self.downsample_factor = latent_downsample_factor
        self.drop = drop
        
        # 🎯 直接指向对应的 tsv 文件
        manifest_path = os.path.join(data_dir, f'{split}.tsv')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"找不到数据集清单文件: {manifest_path}")
            
        df = pd.read_csv(manifest_path, sep='\t')

        # 对测试集同名文件添加编号后缀防覆盖
        if split == 'test':
            df = self.add_name_num(df)
            
        self.dataset = df
        self.dataset.reset_index(drop=True, inplace=True)
        print(f'[{split}] 数据集加载成功: {manifest_path}, 样本数: {len(self.dataset)}')

    def add_name_num(self, df):
        """给同名的音频数据加后缀，例如 name_0, name_1"""
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
        data = self.dataset.iloc[idx]
        item = {}
        
        # 1. 纯文本和音符特征加载 (轻量化)
        ph = ast.literal_eval(data['ph'])
        ph_durs = ast.literal_eval(data['ph_durs'])
        ep_pitches = ast.literal_eval(data['ep_pitches'])
        ep_notedurs = ast.literal_eval(data['ep_notedurs'])
        ep_types = ast.literal_eval(data['ep_types'])
        
        # 2. 映射音素为 ID
        ph2id = {p: i for i, p in enumerate(ph_set)}
        ph = [ph2id.get(p, 59) for p in ph] 
        
        # 3. 时长下采样映射 (匹配 VAE)
        if self.downsample_factor > 1:
            ph_durs = [float(d) / self.downsample_factor for d in ph_durs]

        # 4. 长度对齐 (Padding / Truncation)
        if len(ph) > self.max_ph_len:
            start_tok = np.random.randint(len(ph) - self.max_ph_len + 1)
            end_tok = start_tok + self.max_ph_len
            
            ph = ph[start_tok:end_tok]
            ph_durs = ph_durs[start_tok:end_tok]
            ep_pitches = ep_pitches[start_tok:end_tok]
            ep_notedurs = ep_notedurs[start_tok:end_tok]
            ep_types = ep_types[start_tok:end_tok]
            
        elif len(ph) < self.max_ph_len:
            pad_n = self.max_ph_len - len(ph)
            ph          += [59] * pad_n    # ph PAD = 475
            ph_durs     += [0] * pad_n    # 时长 PAD = 0
            ep_pitches  += [0]  * pad_n    # pitch PAD = 0
            ep_notedurs += [0.0] * pad_n    # 音符时值 PAD = 0.0
            ep_types    += [4]   * pad_n    # type PAD = 4

        # 5. 组装张量数据
        item['ph'] = np.array(ph, dtype=np.int64)
        item['pitches'] = np.array(ep_pitches, dtype=np.int64)
        item['notedurs'] = np.array(ep_notedurs, dtype=np.float32)
        item['notetypes'] = np.array(ep_types, dtype=np.int64)
        item['ph_durs'] = np.array(ph_durs, dtype=np.float32)
        
        if self.split == 'test':
            item['f_name'] = data['item_name']
            
        return item

    def __len__(self):
        return len(self.dataset)

# 确保这里的传参和 yaml 中的目标类严格对应
class DurTrainDataset(DurationDataset):
    def __init__(self, dataset_cfg):
        super().__init__('train', **dataset_cfg)

class DurValidationDataset(DurationDataset):
    def __init__(self, dataset_cfg):
        super().__init__('valid', **dataset_cfg)

class DurTestDataset(DurationDataset):
    def __init__(self, dataset_cfg):
        super().__init__('test', **dataset_cfg)
