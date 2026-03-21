import json
import os
import glob
from tqdm import tqdm

# ==========================================
# 1. 配置区域
# ==========================================

REQUIRED_KEYS = [
    "item_name", "txt", "ph", "ph_durs", "word_durs", 
    "ep_pitches", "ep_notedurs", "ep_types", "ph2words",
    "mix_tech", "falsetto_tech", "breathy_tech", 
    "pharyngeal_tech", "vibrato_tech", "glissando_tech", "tech",
    "wav_fn", "language", "singer", 
    "emotion", "singing_method", "pace", "range"
]

DEFAULTS = {
    "emotion": "happy",
    "singing_method": "pop",
    "pace": "moderate",
    "range": "medium",
    "language": "Chinese"
}

# ==========================================
# 2. 核心逻辑
# ==========================================

def get_items_from_json(data):
    """
    智能解析 JSON 数据，将其统一转换为 List of Objects
    处理情况：
    1. [ {...}, ... ] -> 直接返回
    2. { "item_name": ... } -> 包装成列表返回
    3. { "opencpop_xxx": { "item_name": ... } } -> 提取 value 返回 (解决你当前的问题)
    """
    if isinstance(data, list):
        return data
    
    if isinstance(data, dict):
        # 检查是否是单一数据对象 (含有 item_name)
        if "item_name" in data:
            return [data]
        
        # 检查是否是字典包裹的情况 (Key是ID, Value是数据)
        # 例如: {"opencpop_2050001886": {"item_name":...}}
        # 我们取所有的 values
        return list(data.values())
    
    return []

def merge_datasets_to_flat_list(input_dirs, output_file):
    merged_list = []
    total_files = 0
    
    print(f"Target Output: {output_file}")
    
    for d in input_dirs:
        if not os.path.exists(d):
            print(f"[Warning] Directory not found: {d}")
            continue
            
        files = glob.glob(os.path.join(d, "*.json"))
        # 排除自身
        files = [f for f in files if os.path.basename(f) != os.path.basename(output_file)]
        
        print(f"Processing {d} ... Found {len(files)} files.")
        
        for fpath in tqdm(files, desc=f"Loading {os.path.basename(d)}"):
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                # --- 关键步骤：智能解包 ---
                items_to_add = get_items_from_json(raw_data)
                
                for item in items_to_add:
                    # 再次确认这是一个有效的数据对象
                    if not isinstance(item, dict): continue
                    
                    # --- 字段清洗 ---
                    cleaned_item = {}
                    for key in REQUIRED_KEYS:
                        if key in item:
                            cleaned_item[key] = item[key]
                        elif key in DEFAULTS:
                            cleaned_item[key] = DEFAULTS[key]
                        else:
                            cleaned_item[key] = [] # 列表类型默认空
                    
                    # 强制删除 speech_fn (如果混进去了)
                    if "speech_fn" in cleaned_item:
                        del cleaned_item["speech_fn"]
                    
                    merged_list.append(cleaned_item)
                    total_files += 1
                    
            except Exception as e:
                print(f"[Error] Failed to read {fpath}: {e}")

    # ==========================================
    # 3. 保存为 List 格式
    # ==========================================
    print("="*30)
    print(f"Merging Complete.")
    print(f"Total Items: {len(merged_list)}")
    print(f"Saving to {output_file} ...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # indent=4 方便阅读，生成的是 [ {}, {} ] 格式
        json.dump(merged_list, f, ensure_ascii=False, indent=4)
        
    print("Done.")

# ==========================================
# 运行入口
# ==========================================
if __name__ == "__main__":
    INPUT_DIRS = [
        "/data7/tyx/DiffSVS/data/preprocess/score_data/opencpop",
        "/data7/tyx/DiffSVS/data/preprocess/score_data/gtsinger",
        "/data7/tyx/DiffSVS/data/preprocess/score_data/acesinger",
        "/data7/tyx/DiffSVS/data/preprocess/score_data/m4singer",
    ]
    
    OUTPUT_FILE = "/data7/tyx/DiffSVS/data/preprocess/meta.json"

    merge_datasets_to_flat_list(INPUT_DIRS, OUTPUT_FILE)