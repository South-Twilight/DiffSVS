import json
import math
import os

def validate_svs_entry(data):
    """
    校验单条数据的合法性，返回 (是否合法, 错误列表, 警告列表)
    """
    errors = []
    warnings = []
    
    # 1. 检查核心字段
    required_keys = ["item_name", "txt", "ph", "ph_durs", "word_durs", "ph2words", "wav_fn"]
    for key in required_keys:
        if key not in data:
            errors.append(f"缺失核心字段: {key}")
    
    if errors:
        return False, errors, warnings 

    # 2. 长度一致性校验
    num_txt = len(data["txt"])
    num_ph = len(data["ph"])

    if len(data["word_durs"]) != num_txt:
        errors.append(f"word_durs 长度 ({len(data['word_durs'])}) 与 txt 长度 ({num_txt}) 不一致")

    ph_aligned_keys = [
        "ph_durs", "ep_pitches", "ep_notedurs", "ep_types", "ph2words", 
        "mix_tech", "falsetto_tech", "breathy_tech", "pharyngeal_tech", 
        "vibrato_tech", "glissando_tech", "tech"
    ]
    
    for key in ph_aligned_keys:
        if key in data and len(data[key]) != num_ph:
            errors.append(f"{key} 长度 ({len(data[key])}) 与 ph 长度 ({num_ph}) 不一致")

    # 3. 映射逻辑校验 (ph2words)
    if data["ph2words"]:
        max_word_idx = max(data["ph2words"])
        min_word_idx = min(data["ph2words"])
        if max_word_idx >= num_txt:
            errors.append(f"ph2words 包含越界索引 {max_word_idx}，txt 只有 {num_txt} 个词")
        if min_word_idx < 0:
            errors.append(f"ph2words 包含非法的负数索引 {min_word_idx}")
            
        # ⚠️ 警告级检查：映射是否严格单调递增
        # 如果音素对应的词索引突然变小，说明对齐发生了“时光倒流”，这在正常语流中是不可能的
        if data["ph2words"] != sorted(data["ph2words"]):
            warnings.append(f"ph2words 映射非单调递增 (出现错位乱序): {data['ph2words']}")

    # 4. 时长数学校验
    sum_ph_dur = sum(data["ph_durs"])
    sum_word_dur = sum(data["word_durs"])
    if not math.isclose(sum_ph_dur, sum_word_dur, abs_tol=1e-3):
        errors.append(f"时长不匹配: 音素总时长 ({sum_ph_dur:.4f}s) != 词总时长 ({sum_word_dur:.4f}s)")

    # 5. 特殊 Token 对齐 (SP/AP)
    # ⚠️ 警告级检查：静音/呼吸标记是否混入了实际发音的音素
    for i, word in enumerate(data["txt"]):
        if word in ["<SP>", "<AP>"]:
            mapped_phs = [data["ph"][j] for j, w_idx in enumerate(data["ph2words"]) if w_idx == i]
            # 如果对应音素里包含了类似 "a", "m", "g_zh" 等非静音符号，说明对齐切分有问题
            if not all(ph in ["<SP>", "<AP>", "sp", "ap"] for ph in mapped_phs):
                warnings.append(f"休止符/呼吸符 '{word}' (索引 {i}) 错误地对齐到了发音音素: {mapped_phs}")

    # 6. 文件路径校验
    # ⚠️ 警告级检查：如果在本机跑，顺便查一下音频文件存不存在
    wav_path = data["wav_fn"]
    if not os.path.exists(wav_path):
        warnings.append(f"音频文件在当前路径下找不到: {wav_path}")

    return len(errors) == 0, errors, warnings


def parse_json_file(file_path):
    """
    智能解析 JSON 文件
    """
    entries = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                content = json.load(f)
                if isinstance(content, list):
                    entries.extend(content)
                elif isinstance(content, dict):
                    if "item_name" in content and "txt" in content and "ph" in content:
                        entries.append(content)
                    else:
                        entries.extend(list(content.values()))
            except json.JSONDecodeError:
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
    except Exception as e:
        return None, str(e)
        
    return entries, None


def scan_dataset_directory(root_dir):
    """
    扫描整个目录并收集所有 Error 和 Warning
    """
    if not os.path.isdir(root_dir):
        print(f"❌ 找不到指定的目录: {root_dir}")
        return

    print(f"🚀 开始深度扫描目录: {root_dir}")
    print("-" * 60)
    
    stats = {"files_scanned": 0, "entries_checked": 0, "valid_entries": 0}
    error_logs = []
    warning_logs = []
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith('.json'):
                continue
                
            file_path = os.path.join(dirpath, filename)
            stats["files_scanned"] += 1
            
            entries, parse_error = parse_json_file(file_path)
            
            if parse_error:
                error_logs.append({
                    "file": file_path,
                    "item_name": "N/A",
                    "errors": [f"文件解析失败或损坏: {parse_error}"]
                })
                continue
                
            if not entries:
                continue

            for data in entries:
                stats["entries_checked"] += 1
                item_name = data.get("item_name", "Unknown_Item")
                
                is_valid, errors, warnings = validate_svs_entry(data)
                
                # 记录错误
                if not is_valid:
                    error_logs.append({
                        "file": file_path,
                        "item_name": item_name,
                        "errors": errors
                    })
                else:
                    stats["valid_entries"] += 1
                
                # 记录警告 (不论是否有错误，只要有警告就记录)
                if warnings:
                    warning_logs.append({
                        "file": file_path,
                        "item_name": item_name,
                        "warnings": warnings
                    })

    # ================= 输出最终报告 =================
    print("\n📊 扫描与校验报告")
    print("=" * 60)
    print(f"📁 扫描的 JSON 文件总数 : {stats['files_scanned']}")
    print(f"📝 校验的数据条目总数   : {stats['entries_checked']}")
    print(f"✅ 格式合法的数据条目   : {stats['valid_entries']}")
    print(f"❌ 存在 ERROR 的条目    : {len(error_logs)}")
    print(f"⚠️ 存在 WARNING 的条目  : {len(warning_logs)}")
    print("=" * 60)
    
    if error_logs or warning_logs:
        report_filename = "dataset_inspection_report.txt"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write("=== 歌声合成数据集深度检测报告 ===\n\n")
            
            # --- 写入致命错误 (Errors) ---
            if error_logs:
                f.write("🔴 【第一部分：致命错误 (ERRORS)】\n")
                f.write("说明：这些错误通常会导致训练直接崩溃，必须被修复或从数据集中剔除。\n\n")
                for log in error_logs:
                    f.write(f"📂 来源文件: {log['file']}\n")
                    f.write(f"🔖 数据项名: {log['item_name']}\n")
                    for err in log['errors']:
                        f.write(f"   ❌ {err}\n")
                    f.write("-" * 50 + "\n")
            
            # --- 写入异常警告 (Warnings) ---
            if warning_logs:
                f.write("\n\n🟡 【第二部分：潜在异常警告 (WARNINGS)】\n")
                f.write("说明：这些数据格式合法，能跑通训练，但存在逻辑瑕疵，可能会严重影响模型的发音质量（吐字不清/电音/爆音等）。\n\n")
                for log in warning_logs:
                    f.write(f"📂 来源文件: {log['file']}\n")
                    f.write(f"🔖 数据项名: {log['item_name']}\n")
                    for warn in log['warnings']:
                        f.write(f"   ⚠️ {warn}\n")
                    f.write("-" * 50 + "\n")
                    
        print(f"\n🚨 检测完成！详细的排查报告已生成至: {report_filename}")
        print("建议打开该 txt 文件，优先处理 ERROR，再审阅 WARNING 是否可以接受。")
    else:
        print("\n🎉 完美！扫描的所有数据集文件极其健康，没有任何错误或警告！")


if __name__ == "__main__":
    # ==========================================
    # 在这里填入你的数据集文件夹的【根目录】路径
    # 例如：ROOT_DIR = "/data/my_datasets"
    # ==========================================
    ROOT_DIR = "/data7/tyx/DiffSVS/data/gtsinger" 
    
    scan_dataset_directory(ROOT_DIR)