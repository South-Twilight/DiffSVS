import json
import os
import copy
import glob
import re
from tqdm import tqdm

def generate_acesinger_labels_flat(opencpop_json_root, acesinger_wav_root, output_dir):
    """
    基于扁平化 wav 目录结构生成 AceSinger 标注
    wav结构: acesinger_{singer_tag}#{uid}.wav (例如: acesinger_1crimon#2097003617.wav)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 1. 扫描所有 wav 文件
    print(f"Scanning audio files in: {acesinger_wav_root}")
    # 获取所有 .wav 文件
    wav_files = glob.glob(os.path.join(acesinger_wav_root, "*.wav"))
    
    print(f"Found {len(wav_files)} wav files.")

    success_count = 0
    missing_template_count = 0

    # 2. 遍历每个音频文件
    for wav_path in tqdm(wav_files):
        try:
            filename = os.path.basename(wav_path) 
            name_no_ext = os.path.splitext(filename)[0]
            
            # --- 解析文件名 ---
            # 格式要求: acesinger_{singer_tag}#{uid}
            # 使用正则精准提取
            match = re.match(r'^acesinger_(.*?)#(\d+)$', name_no_ext)
            if not match:
                print(f"[Skip] Invalid filename format: {filename}")
                continue
            
            singer_tag = match.group(1)  # e.g., '1crimon'
            uid = match.group(2)         # e.g., '2097003617'
            
            target_singer_name = f"acesinger_{singer_tag}"

            # 3. 寻找对应的 Opencpop 模板
            template_path = os.path.join(opencpop_json_root, f"opencpop_{uid}.json")
            
            if not os.path.exists(template_path):
                # print(f"Template not found for UID: {uid}")
                missing_template_count += 1
                continue
            
            # 4. 读取并修改
            with open(template_path, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
            
            new_data = copy.deepcopy(template_data)
            
            # --- 修改元数据 ---
            # Item Name: acesinger_1crimon#2097003617
            new_data["item_name"] = f"{target_singer_name}#{uid}"
            
            # Singer: acesinger_1crimon
            new_data["singer"] = target_singer_name
            
            # Wav Fn: 绝对路径 (直接使用扫描到的路径)
            new_data["wav_fn"] = wav_path
            
            # 清理 speech_fn (如果模板中残留)
            if "speech_fn" in new_data:
                del new_data["speech_fn"]

            # 5. 保存
            # 输出文件名: acesinger_1crimon#2097003617.json
            out_filename = f"{target_singer_name}#{uid}.json"
            out_path = os.path.join(output_dir, out_filename)
            
            with open(out_path, 'w', encoding='utf-8') as f_out:
                json.dump(new_data, f_out, ensure_ascii=False, indent=4)
            
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")

    print("="*30)
    print(f"Processing Complete.")
    print(f"Total Generated: {success_count}")
    print(f"Missing Templates: {missing_template_count}")
    print(f"Output Directory: {output_dir}")

# ==========================================
# 入口
# ==========================================
if __name__ == "__main__":
    # Opencpop 完美 JSON 目录
    OPENCPOP_JSON_DIR = "/data7/tyx/DiffSVS/data/preprocess/score_data/opencpop"
    
    # AceSinger 扁平化 Wav 目录
    ACESINGER_WAV_ROOT = "/data7/tyx/espnet/egs2/acesinger/svs1/wav_dump"
    
    # 输出目录
    OUTPUT_DIR = "/data7/tyx/DiffSVS/data/preprocess/score_data/acesinger"

    generate_acesinger_labels_flat(OPENCPOP_JSON_DIR, ACESINGER_WAV_ROOT, OUTPUT_DIR)
