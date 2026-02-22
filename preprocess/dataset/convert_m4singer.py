import json
import os
from tqdm import tqdm
from pypinyin import pinyin, Style

# ==========================================
# 1. 配置与辅助函数
# ==========================================

def get_tech_string(is_glissando):
    return "6" if is_glissando else "0"

# ==========================================
# 2. 核心转换逻辑
# ==========================================

def convert_m4singer_item(meta_item, wav_root_prefix="m4singer"):
    # 1. 解析基础信息
    # item_name: "Alto-1#newboy#0000"
    raw_name = meta_item['item_name']
    parts = raw_name.split('#')
    if len(parts) == 3:
        singer, song, fid = parts
    else:
        # 容错处理
        singer, song, fid = "m4singer", "unknown", raw_name

    # 构造新的 item_name
    safe_folder = f"{singer}_{song}"
    new_item_name = f"m4singer_{safe_folder}_{fid}"
    
    # 构造 wav_fn
    # 假设文件结构: m4singer/Alto-1#newboy/0000.wav
    wav_fn = f"{wav_root_prefix}/{singer}#{song}/{fid}.wav"

    # 提取原始序列数据
    src_phs = meta_item['phs']           # 音素列表
    src_txt = meta_item['txt']           # 纯汉字文本 (无 SP/AP)
    src_slur = meta_item['is_slur']      # 0/1
    src_ph_dur = meta_item['ph_dur']     # 秒
    src_notes = meta_item['notes']       # MIDI
    src_note_dur = meta_item['notes_dur']# 秒

    # 初始化结果字典
    seq = {
        "item_name": new_item_name,
        "txt": [],
        "ph": [],
        "ph_durs": [],
        "word_durs": [],
        "ep_pitches": [],
        "ep_notedurs": [],
        "ep_types": [],
        "ph2words": [],
        # Tech Lists
        "mix_tech": [], "falsetto_tech": [], "breathy_tech": [],
        "pharyngeal_tech": [], "vibrato_tech": [], "glissando_tech": [],
        "tech": [],
        # Metadata
        "wav_fn": wav_fn,
        "language": "Chinese",
        "singer": singer,
        "emotion": "happy",
        "singing_method": "pop",
        "pace": "moderate",
        "range": "medium"
    }

    ph_ptr = 0  # 音素指针
    txt_ptr = 0 # 汉字指针
    word_idx = 0 # 输出的字索引

    # 获取汉字总数
    total_chars = len(src_txt)

    # --- 主循环：遍历所有音素 ---
    # 我们以“字”为核心进行推进，需要不断消费音素
    
    while ph_ptr < len(src_phs):
        curr_ph = src_phs[ph_ptr]

        # Case A: 处理休止符 (SP/AP)
        if curr_ph in ['<SP>', '<AP>']:
            p_dur = src_ph_dur[ph_ptr]
            n_dur = src_note_dur[ph_ptr]
            
            # 填充数据
            seq["txt"].append(f"<{curr_ph[1:-1]}>") # 去掉原有<>再包一层，或者直接用
            # M4Singer meta 里的已经是 <AP>, <SP>
            # 目标格式也是 <AP>, <SP> (包裹在 txt 和 ph 中)
            
            # 修正：如果已经是 <AP>，再加 <> 变成 <<AP>> 就不对了
            # 假设 meta 里是 "<AP>"，我们要存 "<AP>"
            token = curr_ph 
            
            seq["txt"][-1] = token # 覆盖上面的 append
            seq["word_durs"].append(round(p_dur, 4))
            
            seq["ph"].append(token) # 无后缀
            seq["ph_durs"].append(round(p_dur, 4))
            seq["ph2words"].append(word_idx)
            seq["ep_pitches"].append(0) # Rest pitch 0
            seq["ep_notedurs"].append(round(n_dur, 5))
            seq["ep_types"].append(1) # Rest
            
            # Techs
            for k in ["mix", "falsetto", "breathy", "pharyngeal", "vibrato", "glissando"]:
                seq[f"{k}_tech"].append(0)
            seq["tech"].append("0")
            
            ph_ptr += 1
            word_idx += 1
            continue

        # Case B: 处理汉字
        # 确保还有汉字可匹配
        if txt_ptr >= total_chars:
            # 异常情况：音素还有，但字没了（可能是尾部的噪音或对齐错误）
            # 简单跳过或视为 AP
            ph_ptr += 1
            continue

        char = src_txt[txt_ptr]
        
        # 1. 计算当前字应该包含哪些音素
        # 即使 M4Singer 给出了音素，我们也需要 Pypinyin 来校验边界，防止错位
        std_init = pinyin(char, style=Style.INITIALS, strict=True)[0][0]
        # M4Singer 的数据里 'y', 'w' 也是声母，需要兼容
        
        current_word_indices = []
        
        # [Step 1] 尝试匹配声母
        # 逻辑：当前音素不是 AP/SP，且 (是标准声母 OR 是 y/w OR 它是该字唯一的音素)
        if curr_ph in ['y', 'w'] or (std_init != "" and curr_ph == std_init):
            current_word_indices.append(ph_ptr)
            ph_ptr += 1
        
        # [Step 2] 匹配韵母 (必须有)
        if ph_ptr < len(src_phs):
            # 只要不是 AP/SP，基本上就是韵母
            if src_phs[ph_ptr] not in ['<SP>', '<AP>']:
                current_word_indices.append(ph_ptr)
                ph_ptr += 1
        
        # [Step 3] 匹配转音 (Slur)
        # 检查 is_slur 标记。如果下一个音素 is_slur==1，说明它是当前字的延续
        while ph_ptr < len(src_phs):
            is_slur_flag = src_slur[ph_ptr]
            next_ph = src_phs[ph_ptr]
            
            # 只有当它是 Slur (1) 且不是特殊符号时，才归入当前字
            if is_slur_flag == 1 and next_ph not in ['<SP>', '<AP>']:
                current_word_indices.append(ph_ptr)
                ph_ptr += 1
            else:
                break
        
        # 2. 填充数据
        if not current_word_indices:
            # 异常：没匹配到任何音素 (不应该发生)
            txt_ptr += 1
            continue

        total_word_dur = sum([src_ph_dur[i] for i in current_word_indices])
        
        # 目标格式要求 txt 用 <> 包裹特殊符，普通字不用
        seq["txt"].append(char)
        seq["word_durs"].append(round(total_word_dur, 4))
        
        for i, idx in enumerate(current_word_indices):
            raw_ph = src_phs[idx]
            is_slur_val = src_slur[idx]
            
            # 添加 _zh 后缀
            seq["ph"].append(f"{raw_ph}_zh")
            seq["ph_durs"].append(round(src_ph_dur[idx], 4))
            seq["ph2words"].append(word_idx)
            
            # Pitch & NoteDur
            seq["ep_pitches"].append(int(src_notes[idx]))
            seq["ep_notedurs"].append(round(src_note_dur[idx], 5))
            
            # Type & Slur
            is_gliss = (is_slur_val == 1)
            
            # Type Logic: 3=Glissando, 2=Note
            seq["ep_types"].append(3 if is_gliss else 2)
            
            # Tech Flags
            seq["mix_tech"].append(0)
            seq["falsetto_tech"].append(0)
            seq["breathy_tech"].append(0)
            seq["pharyngeal_tech"].append(0)
            seq["vibrato_tech"].append(0)
            seq["glissando_tech"].append(1 if is_gliss else 0)
            
            seq["tech"].append(get_tech_string(is_gliss))
        
        txt_ptr += 1
        word_idx += 1

    return seq

# ==========================================
# 3. 批量处理入口
# ==========================================

def process_m4singer_meta(input_json_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading meta from: {input_json_path}")
    
    # M4Singer 的 meta.json 通常是一个包含所有 item 的列表，或者是一个大字典
    # 这里假设是一个 List (根据常见的 meta.json 格式)
    # 或者是您提供的那种单行 JSON 格式的文本文件
    
    with open(input_json_path, 'r', encoding='utf-8') as f:
        # 尝试读取整个文件为 JSON
        try:
            data = json.load(f)
            # 如果是字典 {"item_id": data}，转为列表
            if isinstance(data, dict):
                items = data.values()
            elif isinstance(data, list):
                items = data
            else:
                items = [data] # 单个对象
        except:
            # 如果是 JSONL (每行一个 JSON)
            f.seek(0)
            items = [json.loads(line) for line in f if line.strip()]

    print(f"Found {len(items)} items. Processing...")
    
    success_count = 0
    
    for item in tqdm(items):
        try:
            seq_data = convert_m4singer_item(item, "/data7/tyx/dataset/m4singer")
            
            # 提取 uid 用于文件名
            # m4singer_Singer_Song_ID
            out_name = f"{seq_data['item_name']}.json"
            
            out_file = os.path.join(output_dir, out_name)
            with open(out_file, 'w', encoding='utf-8') as f_out:
                json.dump(seq_data, f_out, ensure_ascii=False, indent=4)
            
            success_count += 1
        except Exception as e:
            print(f"Error processing {item.get('item_name', 'unknown')}: {e}")

    print(f"Done. Success: {success_count}")

if __name__ == "__main__":
    # 输入：M4Singer 的原始 meta.json 文件路径
    INPUT_META = "/data7/tyx/dataset/m4singer/meta.json" 
    
    # 输出：转换后的 JSON 文件夹
    OUTPUT_DIR = "/data7/tyx/DiffSVS/data/m4singer"
    
    process_m4singer_meta(INPUT_META, OUTPUT_DIR)
