import json
import os

def minify_json(input_path, output_path):
    print(f"Reading: {input_path}")
    
    # 1. 读取原始数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} items. Writing to {output_path}...")

    # 2. 写入压缩格式
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(
            data, 
            f, 
            ensure_ascii=False,  # 关键点1：中文直接显示，不转义为 \uXXXX (更小且可读)
            separators=(',', ':') # 关键点2：去除逗号和冒号后面的空格
            # indent 参数不写，默认就是无缩进
        )

    # 3. 对比大小
    old_size = os.path.getsize(input_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"Done.")
    print(f"Original: {old_size:.2f} MB")
    print(f"Minified: {new_size:.2f} MB")
    print(f"Saved:    {old_size - new_size:.2f} MB")

if __name__ == "__main__":
    INPUT_FILE = "/data7/tyx/DiffSVS/data/meta.json"
    OUTPUT_FILE = "/data7/tyx/DiffSVS/data/meta_min.json"
    
    minify_json(INPUT_FILE, OUTPUT_FILE)