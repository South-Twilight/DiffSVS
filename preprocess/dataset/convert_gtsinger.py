import json
import os
from tqdm import tqdm

def process_gtsinger_meta(input_path, output_path, path_prefix):
    """
    修改 GTSinger metadata:
    1. 给 wav_fn 添加前缀
    2. 删除 speech_fn
    """
    print(f"Loading metadata from: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 判断数据结构是 List 还是 Dict
    if isinstance(data, list):
        items = data
        print(f"Detected format: List (Total {len(items)} items)")
        iterator = items
    elif isinstance(data, dict):
        items = data.values()
        print(f"Detected format: Dict (Total {len(items)} items)")
        iterator = items
    else:
        raise ValueError("Unsupported JSON format. Must be List or Dict.")

    modified_count = 0

    # --- 核心处理循环 ---
    for item in tqdm(iterator):
        # 1. 修改 wav_fn (添加前缀)
        if "wav_fn" in item:
            original_wav = item["wav_fn"]
            # 使用 os.path.join 智能拼接，防止多重斜杠
            # 如果 original_wav 已经是绝对路径，join 会忽略 prefix，所以这里强制拼接
            # 假设 original_wav 是相对路径 (如 "Chinese/ZH-...")
            
            # 移除可能存在的开头的斜杠，防止 join 失效
            clean_rel_path = original_wav.lstrip(os.sep).lstrip('/')
            
            item["wav_fn"] = os.path.join(path_prefix, clean_rel_path)

        # 2. 删除 speech_fn
        if "speech_fn" in item:
            del item["speech_fn"]
            
        modified_count += 1

    # --- 保存结果 ---
    print(f"Saving modified metadata to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        # 使用 ensure_ascii=False 保证中文可读
        # 使用 indent=4 保持格式整洁 (若想压缩体积可去掉 indent)
        json.dump(data, f_out, ensure_ascii=False, indent=4)

    print("Done.")

# ==========================================
# 运行配置
# ==========================================
if __name__ == "__main__":
    # 输入文件 (GTSinger Chinese 的原始 meta.json)
    INPUT_FILE = "/data7/tyx/DiffSVS/data/gtsinger/meta.json" 
    
    # 输出文件 (修改后的文件)
    OUTPUT_FILE = "/data7/tyx/DiffSVS/data/gtsinger/gtsinger.json"
    
    # 需要添加的前缀路径
    # 例如：最终路径变成 /data7/tyx/dataset/GTSinger/Chinese/...
    WAV_PREFIX = "/data7/tyx/dataset/GTSinger" 

    process_gtsinger_meta(INPUT_FILE, OUTPUT_FILE, WAV_PREFIX)
