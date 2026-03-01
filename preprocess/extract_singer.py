import os
import pandas as pd
import glob

def generate_singer_txt(data_dir, output_file="singer.txt"):
    """
    从数据目录中的所有 tsv 文件里提取唯一的 singer，
    并生成固定格式的 singer.txt，强制 'unknown' 为第一项。
    """
    # 找到目录下所有的 tsv 文件 (比如 train.tsv, valid.tsv, test.tsv)
    tsv_files = glob.glob(os.path.join(data_dir, '*.tsv'))
    
    if not tsv_files:
        print(f"❌ 在 {data_dir} 目录下没有找到任何 .tsv 文件！")
        return

    unique_singers = set()
    
    for tsv_path in tsv_files:
        print(f"正在读取: {os.path.basename(tsv_path)} ...")
        try:
            df = pd.read_csv(tsv_path, sep='\t', low_memory=False)
            
            # 检查是否存在 singer 列
            if 'singer' in df.columns:
                # 提取所有非空的歌手名，转为字符串并去除首尾空格
                singers = df['singer'].dropna().astype(str).str.strip().tolist()
                unique_singers.update(singers)
            else:
                print(f"⚠️ 警告: {os.path.basename(tsv_path)} 中没有 'singer' 列。")
                
        except Exception as e:
            print(f"❌ 读取 {tsv_path} 时出错: {e}")

    # 清理掉可能不小心被当成歌手的空字符串或原本就有的 'unknown'
    unique_singers.discard("")
    unique_singers.discard("unknown")
    unique_singers.discard("nan") # pandas 读取空值时转换的字符串
    
    # 为了保证每次生成的字典顺序完全一致（极度重要！），进行排序
    sorted_singers = sorted(list(unique_singers))
    
    # 🌟 强制把 unknown 放在第一位 (id=0)
    final_singer_list = ['unknown'] + sorted_singers
    
    # 写入 singer.txt
    output_path = os.path.join(data_dir, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        for singer in final_singer_list:
            f.write(f"{singer}\n")
            
    print("-" * 30)
    print(f"✅ 提取成功！共找到 {len(final_singer_list) - 1} 名真实歌手。")
    print(f"✅ 已保存至: {output_path}")
    print("前 5 个歌手(含 unknown) 预览:")
    for i, s in enumerate(final_singer_list[:5]):
        print(f"  ID {i}: {s}")

if __name__ == "__main__":
    # 👇 把这里改成你存放 tsv 文件的真实路径
    YOUR_DATA_DIR = "./data/final" 
    
    generate_singer_txt(YOUR_DATA_DIR)