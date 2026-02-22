import json
import os
from tqdm import tqdm
from pypinyin import pinyin, Style

# ==========================================
# 1. 配置与辅助函数
# ==========================================

# Tech ID Mapping (GTSinger Standard)
TECH_MAP_ID = {
    "mix": 1,
    "falsetto": 2,
    "breathy": 3,
    "pharyngeal": 4,
    "vibrato": 5,
    "glissando": 6
}

def note_to_midi(note_str):
    """将 Opencpop 音高转为 MIDI 编号，rest/None 归于 0"""
    if not note_str or note_str.lower() == 'rest':
        return 0
    note_name = note_str.split('/')[0]
    mapping = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 
               'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
    try:
        name = note_name[:-1]
        octave = int(note_name[-1])
        # 标准 MIDI C4 = 60
        return 12 * (octave + 1) + mapping[name]
    except:
        return 0

def get_tech_string(is_glissando):
    """
    根据技巧状态生成字符串。Opencpop 只包含 Glissando (ID 6)。
    """
    if is_glissando:
        return "6"
    return "0"

# ==========================================
# 2. 核心转换逻辑
# ==========================================

def convert_opencpop_line(line, wav_root):
    parts = line.strip().split('|')
    if len(parts) < 7: return None, None
    
    # 解析原始数据
    uid, lyrics, ph_raw, pitch_raw, note_dur_raw, ph_dur_raw, slur_raw = parts
    phonemes = ph_raw.split()
    pitches = pitch_raw.split()
    note_durs = [float(x) for x in note_dur_raw.split()]
    ph_durs = [float(x) for x in ph_dur_raw.split()]
    slurs = slur_raw.split()
    
    # 初始化结果字典 (严格按照目标格式)
    seq = {
        "item_name": f"opencpop#{uid}",
        "txt": [],
        "ph": [],
        "ph_durs": [],
        "word_durs": [],
        "ep_pitches": [],
        "ep_notedurs": [],
        "ep_types": [],
        "ph2words": [],
        # Tech Lists
        "mix_tech": [],
        "falsetto_tech": [],
        "breathy_tech": [],
        "pharyngeal_tech": [],
        "vibrato_tech": [],
        "glissando_tech": [],
        "tech": [],
        # Metadata
        "wav_fn": os.path.join(wav_root, f"{uid}.wav"),
        "language": "Chinese",
        "singer": "opencpop",
        "emotion": "happy",
        "singing_method": "pop",
        "pace": "moderate",
        "range": "medium"
    }

    ph_ptr = 0  # Opencpop 原始音素指针
    word_idx = 0 # 当前字索引

    # --- 遍历歌词字符 ---
    for char in lyrics:
        
        # A. 优先消费 SP/AP (休止符)
        while ph_ptr < len(phonemes) and phonemes[ph_ptr] in ['SP', 'AP']:
            token = phonemes[ph_ptr]
            p_dur = ph_durs[ph_ptr]
            n_dur = note_durs[ph_ptr]
            midi = note_to_midi(pitches[ph_ptr])
            
            # 【修正点 1】: txt 必须用 <> 包裹
            seq["txt"].append(f"<{token}>")
            seq["word_durs"].append(round(p_dur, 4))
            
            # 【修正点 2】: ph 也必须用 <> 包裹 (且无 _zh 后缀)
            seq["ph"].append(f"<{token}>") 
            seq["ph_durs"].append(round(p_dur, 4))
            seq["ph2words"].append(word_idx)
            seq["ep_pitches"].append(midi)
            seq["ep_notedurs"].append(round(n_dur, 5))
            
            # Type: 1 = Rest
            seq["ep_types"].append(1)
            
            # Techs (全0)
            for k in ["mix", "falsetto", "breathy", "pharyngeal", "vibrato", "glissando"]:
                seq[f"{k}_tech"].append(0)
            seq["tech"].append("0")
            
            ph_ptr += 1
            word_idx += 1

        # B. 处理汉字
        if ph_ptr >= len(phonemes): break

        # 1. 确定当前字的标准声母和韵母
        # strict=True: 'y', 'w' 等被视为空声母
        std_init = pinyin(char, style=Style.INITIALS, strict=True)[0][0]
        std_final = pinyin(char, style=Style.FINALS, strict=True)[0][0].replace('ü', 'v')

        # 2. 识别 Opencpop 实际归属音素索引
        current_word_indices = []
        
        # [Step 1: 尝试匹配声母]
        # 逻辑：如果 Opencpop 当前音素是 'y'/'w' 或者与标准声母一致，则消费它作为声母
        curr_op_ph = phonemes[ph_ptr]
        if curr_op_ph in ['y', 'w'] or std_init != "":
            current_word_indices.append(ph_ptr)
            ph_ptr += 1
        
        # [Step 2: 匹配韵母]
        # 只要没有越界，下一个应该就是韵母 (或者唯一的音素)
        if ph_ptr < len(phonemes):
            current_word_indices.append(ph_ptr)
            last_captured_final = phonemes[ph_ptr] # 记住韵母的样子 (如 'ai')
            ph_ptr += 1
            
            # [Step 3: 贪婪匹配转音延伸 (Melisma Check)]
            # 例如 "ai ai ai"，只要后续音素跟刚才的韵母一样，且不是休止符，就全部归入该字
            while ph_ptr < len(phonemes):
                next_ph = phonemes[ph_ptr]
                if next_ph == last_captured_final and next_ph not in ['SP', 'AP']:
                    current_word_indices.append(ph_ptr)
                    ph_ptr += 1
                else:
                    break
        
        # 3. 填充序列数据
        # 计算字总时长
        total_word_dur = sum([ph_durs[i] for i in current_word_indices])
        seq["txt"].append(char)
        seq["word_durs"].append(round(total_word_dur, 4))
        
        for i, idx in enumerate(current_word_indices):
            raw_ph = phonemes[idx]
            
            # 加 _zh 后缀
            seq["ph"].append(f"{raw_ph}_zh")
            seq["ph_durs"].append(round(ph_durs[idx], 4))
            seq["ph2words"].append(word_idx)
            
            # 音高与时值
            seq["ep_pitches"].append(note_to_midi(pitches[idx]))
            seq["ep_notedurs"].append(round(note_durs[idx], 5))
            
            # Type & Slur
            is_gliss = (slurs[idx] == "1")
            
            # Type Logic: 3=Glissando, 2=Note
            seq["ep_types"].append(3 if is_gliss else 2)
            
            # Tech Flags
            seq["mix_tech"].append(0)
            seq["falsetto_tech"].append(0)
            seq["breathy_tech"].append(0)
            seq["pharyngeal_tech"].append(0)
            seq["vibrato_tech"].append(0)
            seq["glissando_tech"].append(1 if is_gliss else 0)
            
            # Tech String
            seq["tech"].append(get_tech_string(is_gliss))
            
        word_idx += 1

    # C. 处理尾部残留 SP/AP
    while ph_ptr < len(phonemes) and phonemes[ph_ptr] in ['SP', 'AP']:
        token = phonemes[ph_ptr]
        p_dur = ph_durs[ph_ptr]
        n_dur = note_durs[ph_ptr]
        midi = note_to_midi(pitches[ph_ptr])
        
        # 【修正点 1】: txt 用 <> 包裹
        seq["txt"].append(f"<{token}>")
        seq["word_durs"].append(round(p_dur, 4))
        
        # 【修正点 2】: ph 用 <> 包裹
        seq["ph"].append(f"<{token}>") 
        seq["ph_durs"].append(round(p_dur, 4))
        seq["ph2words"].append(word_idx)
        seq["ep_pitches"].append(midi)
        seq["ep_notedurs"].append(round(n_dur, 5))
        
        seq["ep_types"].append(1)
        
        for k in ["mix", "falsetto", "breathy", "pharyngeal", "vibrato", "glissando"]:
            seq[f"{k}_tech"].append(0)
        seq["tech"].append("0")
        
        ph_ptr += 1
        word_idx += 1

    return uid, seq

# ==========================================
# 3. 批量处理
# ==========================================

def process_opencpop_batch(input_path, output_dir, wav_root):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Reading from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    success_count = 0
    error_count = 0
    
    print(f"Start processing {len(lines)} lines...")
    
    for line in tqdm(lines):
        line = line.strip()
        if not line: continue
        
        try:
            uid, json_data = convert_opencpop_line(line, wav_root)
            
            if uid and json_data:
                # File name: opencpop_2001000001.json
                out_file = os.path.join(output_dir, f"opencpop_{uid}.json")
                with open(out_file, 'w', encoding='utf-8') as f_out:
                    # 使用 indent=4 方便查看结构，生产环境可去掉 indent
                    json.dump(json_data, f_out, ensure_ascii=False, indent=4)
                success_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            print(f"Error processing line {line[:20]}...: {e}")
            error_count += 1

    print("="*30)
    print(f"Success: {success_count}")
    print(f"Failed:  {error_count}")
    print(f"Output:  {output_dir}")

# ==========================================
# Entry Point
# ==========================================
if __name__ == "__main__":
    # Input File
    INPUT_TXT = "/data7/tyx/dataset/opencpop/segments/transcriptions.txt"
    
    # Output Directory
    OUTPUT_DIR = "/data7/tyx/DiffSVS/data/opencpop"
    
    # Wav Root (For hardcoded path generation)
    WAV_ROOT = "/data7/tyx/dataset/opencpop/segments/wavs"

    process_opencpop_batch(INPUT_TXT, OUTPUT_DIR, WAV_ROOT)
