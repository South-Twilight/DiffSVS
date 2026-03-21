import os
import json
import pandas as pd
import csv
import unicodedata

# ANSI 转义序列，用于在终端中输出带颜色的文本
RED = '\033[91m'
RESET = '\033[0m'  # 重置颜色

# 尝试将路径的字符标准化为文件系统兼容的格式
def normalize_path(path):
    # 规范化路径，去除非标准字符
    return unicodedata.normalize('NFC', path)

def save_df_to_tsv(dataframe, path):
    dataframe.to_csv(
        path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )
def generate(input_file, output_file, latent_path):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取所有可能的键，假设所有条目的键都相同
    all_keys = set()
    for item in data:
        all_keys.update(item.keys())
    all_keys = list(all_keys)
    
    # 将 'wav_fn' 替换为 'audio_path'
    all_keys.append('mel_path')
    if 'wav_fn' in all_keys:
        all_keys.remove('wav_fn')
        all_keys.append('audio_path')  # 使用 'audio_path' 代替 'wav_fn'

    manifest = {key: [] for key in all_keys}

    missing_audio_files = []  # 用于记录不存在的音频文件
    missing_mel_files = []    # 🌟 新增：用于记录不存在的 mel 特征文件

    print(f'总共{len(data)}个条目')
    for item in data:
        tmp = {}
        audio_value = item.get('wav_fn', '')
        
        # 对路径进行规范化处理
        audio_value = normalize_path(audio_value)
        
        # 1. 检查 audio_path 是否存在
        if not os.path.exists(audio_value):
            missing_audio_files.append(audio_value)
            continue
        
        # 构建 mel_path 路径并规范化
        mel_value = os.path.join(
            latent_path,
            item.get('item_name') + ".npy"
        )
        mel_value = normalize_path(mel_value)
        
        # 🌟 2. 新增：检查 mel_path 是否存在
        if not os.path.exists(mel_value):
            missing_mel_files.append(mel_value)
            continue
        
        # 只有在 audio 和 mel 都存在的情况下，才加入到 manifest 中
        for key in all_keys:
            if key == 'mel_path':
                value = mel_value
            elif key == 'audio_path':
                value = audio_value
            else:
                value = item.get(key, None)
            tmp[key] = value
            manifest[key].append(value)

    # 输出不存在的文件信息
    if missing_audio_files:
        print(RED + f"以下音频文件不存在 ({len(missing_audio_files)} 个):" + RESET)
        for f in missing_audio_files[:10]: # 建议只打印前几个避免刷屏
            print(f"  - {f}")
        if len(missing_audio_files) > 10: print("  ...")

    # 🌟 新增：输出不存在的 mel 文件信息
    if missing_mel_files:
        print(RED + f"以下 mel(latent) 特征文件不存在 ({len(missing_mel_files)} 个):" + RESET)
        for f in missing_mel_files[:10]:
            print(f"  - {f}")
        if len(missing_mel_files) > 10: print("  ...")

    print(f"有效数据共 {len(manifest['audio_path'])} 条。")
    print(f"正在将清单写入 {output_file} ...")
    save_df_to_tsv(pd.DataFrame.from_dict(manifest), output_file)

if __name__ == '__main__':
    input_file = '/data7/tyx/DiffSVS/data/preprocess/meta.json'
    output_file = '/data7/tyx/DiffSVS/data/preprocess/data.tsv'
    latent_path = '/data7/tyx/DiffSVS/data/feat_extract/latent'
    generate(input_file, output_file, latent_path)
