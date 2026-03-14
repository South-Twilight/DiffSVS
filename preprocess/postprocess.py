import os
import re
import ast
import logging
import pandas as pd
from glob import glob

# ==========================================
# 0. 基础配置与日志
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# 配置参数
INPUT_FILE = './data/preprocess/data.tsv'
OUTPUT_FILE = './data/postprocess/data.tsv'

# 列定义
FILL_NO_COLS = ['emotion', 'range', 'pace', 'singing_method']
TECH_COLS = [
    'mix_tech', 'falsetto_tech', 'vibrato_tech', 'breathy_tech', 
    'pharyngeal_tech', 'bubble_tech', 'strong_tech', 'glissando_tech'
]

# 期望的表头顺序
DESIRED_ORDER = [
    'item_name', 
    'word_durs', 'ph_durs', 'ep_notedurs', 
    'txt', 'ph', 'ph2words', 'ep_pitches', 'ep_types', 
    'audio_path', 'mel_path', 
    'prompt_mel_paths',
    'language', 'singer', 'emotion', 'range', 'pace', 'singing_method', 'technique', 'duration',
    'mix_tech', 'falsetto_tech', 'vibrato_tech', 'breathy_tech', 
    'pharyngeal_tech', 'bubble_tech', 'strong_tech', 'glissando_tech'
]

# ==========================================
# 1. 辅助与工具函数
# ==========================================
def safe_literal_eval(val):
    """安全地解析字符串格式的列表"""
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return val
    return val

def calculate_prompt_mel(mel_path, max_prompts=10):
    """根据 mel_path 寻找同目录下最接近的音频作为 prompt_mel (当前未在主流中使用，保留备用)"""
    if not isinstance(mel_path, str) or not os.path.exists(mel_path):
        return None

    base_dir, filename = os.path.split(mel_path)
    match = re.search(r'(\d+)(?:_mel)?\.npy$', filename)
    if not match:
        return None

    original_number = int(match.group(1))
    candidate_files = glob(os.path.join(base_dir, '*.npy'))

    def extract_number(file):
        m = re.search(r'(\d+)(?:_mel)?\.npy$', file)
        return int(m.group(1)) if m else None

    valid_files = [(f, extract_number(f)) for f in candidate_files]
    valid_files = [(f, num) for f, num in valid_files if num is not None and f != mel_path]

    if not valid_files:
        return None

    valid_files.sort(key=lambda x: abs(x[1] - original_number))
    return [f for f, _ in valid_files[:max_prompts]]

# ==========================================
# 2. 核心数据处理函数
# ==========================================
def fill_missing_values(df):
    """填充基础缺失值和技巧列"""
    logger.info("正在填充缺失值...")
    
    # 填充常规文本缺失值
    for col in FILL_NO_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('no').replace('', 'no')
            
    # 填充技巧列 (如果为空则填充 [2, 2, ...])
    def fill_tech(row, col):
        val = row.get(col)
        ph_list = row.get('ph', [])
        ph_len = len(ph_list) if isinstance(ph_list, list) else 0
        
        if pd.isna(val) or val == '':
            return [2] * ph_len
        return safe_literal_eval(val)

    for col in TECH_COLS:
        if col in df.columns:
            df[col] = df.apply(lambda r: fill_tech(r, col), axis=1)
            
    return df

def generate_features(df):
    """生成衍生特征：duration, language, technique"""
    logger.info("正在生成衍生特征 (duration, language, technique)...")
    
    # 计算总时长
    if 'ph_durs' in df.columns:
        df['duration'] = df['ph_durs'].apply(lambda x: sum(x) if isinstance(x, list) else 0)

    # 推理语言
    def infer_language(row):
        lang = row.get('language')
        if pd.isna(lang) or lang == '':
            item_name = str(row.get('item_name', ''))
            return 'English' if '业余' in item_name or '专业' in item_name else 'Chinese'
        return lang
    
    if 'language' in df.columns or 'item_name' in df.columns:
        df['language'] = df.apply(infer_language, axis=1)

    # 聚合 Technique (替代原先缓慢的 iterrows)
    if 'technique' not in df.columns:
        def aggregate_techs(row):
            techs = [col.split('_')[0] for col in TECH_COLS 
                     if col in row and isinstance(row[col], list) and 1 in row[col]]
            return ' and '.join(techs) if techs else 'no'
        
        df['technique'] = df.apply(aggregate_techs, axis=1)

    return df

def filter_invalid_data(df):
    """异常数据清洗过滤"""
    logger.info("正在执行数据过滤清洗...")
    initial_len = len(df)
    
    # 1. 过滤名字带 '#歌词' 的数据
    if 'item_name' in df.columns:
        df = df[~df['item_name'].str.contains('#歌词', na=False)]

    # 2. 过滤负数异常
    for col in ['word_durs', 'ph_durs', 'ep_notedurs']:
        if col in df.columns:
            df = df[df[col].apply(lambda x: all(i >= 0 for i in x) if isinstance(x, list) else True)]

    # 3. 过滤时长与音素长度
    df['ph_len'] = df['ph'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    mask_duration = (df['duration'] >= 2.0) & (df['duration'] <= 20.0)
    mask_ph = (df['ph_len'] >= 2) & (df['ph_len'] <= 100)
    df = df[mask_duration & mask_ph]
    
    # 删除辅助列
    df = df.drop(columns=['ph_len'])
    
    logger.info(f"清洗完成: 删除了 {initial_len - len(df)} 条异常数据。")
    # 关键：清洗后重置索引，保证后续按 position 访问时不越界
    df = df.reset_index(drop=True)
    return df


def add_prompt_column(df, src_col='mel_path', out_col='prompt_mel_paths', max_prompts=1, pattern=r'(\d+)(?:_mel)?\.npy$'):
    """
    基于 TSV 中已有的 *.npy 特征路径，按编号就近为每一行生成 prompt 列。
    不再对目录做 glob 扫描，只在「同一目录、同一 TSV」内部互相选最近编号，复杂度约 O(N log N)。

    只要你的 latent 也是按 001.npy / 002.npy 或 001_mel.npy 这种命名，就可以直接把 src_col
    改成 latent 的列名来用：

        df = add_prompt_column(df, src_col='latent_path', out_col='prompt_latent_paths')
    """
    if src_col not in df.columns:
        logger.warning(f"add_prompt_column: 源列 {src_col} 不存在，跳过 prompt 生成")
        return df

    # 预先为每一行解析出 base_dir 和 数字编号（不访问文件系统）
    base_dirs = []
    numbers = []
    paths = df[src_col].tolist()
    for p in paths:
        if isinstance(p, str):
            base_dir, filename = os.path.split(p)
            m = re.search(pattern, filename)
            if m:
                base_dirs.append(base_dir)
                numbers.append(int(m.group(1)))
            else:
                base_dirs.append(None)
                numbers.append(None)
        else:
            base_dirs.append(None)
            numbers.append(None)

    df['_prompt_dir'] = base_dirs
    df['_prompt_num'] = numbers

    # 初始化结果列：默认每行是空列表
    prompts = [[] for _ in range(len(df))]

    # 按目录 + singer 分组（如果有 singer 列的话），在组内根据编号排序后，
    # 为每一行挑选最近编号的邻居作为 prompt，避免不同歌手互相当 prompt。
    if 'singer' in df.columns:
        tmp = df[['_prompt_dir', '_prompt_num', src_col, 'singer']].reset_index()
        group_keys = ['_prompt_dir', 'singer']
    else:
        tmp = df[['_prompt_dir', '_prompt_num', src_col]].reset_index()
        group_keys = ['_prompt_dir']

    for _, group in tmp.groupby(group_keys):
        # 丢弃没有合法编号的行
        group = group.dropna(subset=['_prompt_num'])
        if len(group) <= 1:
            continue

        group_sorted = group.sort_values('_prompt_num')
        nums = group_sorted['_prompt_num'].to_numpy()
        files = group_sorted[src_col].to_numpy()
        indices = group_sorted['index'].to_numpy()
        n = len(nums)

        for i in range(n):
            # 这里实现 max_prompts == 1 的高效邻居选择（覆盖目前需求）
            prev_dist = nums[i] - nums[i - 1] if i > 0 else float('inf')
            next_dist = nums[i + 1] - nums[i] if i < n - 1 else float('inf')

            if prev_dist == float('inf') and next_dist == float('inf'):
                continue
            if prev_dist <= next_dist:
                j = i - 1
            else:
                j = i + 1

            prompts[indices[i]] = [files[j]] if max_prompts >= 1 else []

    df[out_col] = prompts
    df.drop(columns=['_prompt_dir', '_prompt_num'], inplace=True)
    return df

def reorder_and_save(df, output_path):
    """调整表头顺序并落盘"""
    ordered_cols = [col for col in DESIRED_ORDER if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in ordered_cols]
    
    final_columns = ordered_cols + remaining_cols
    df = df[final_columns]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, sep='\t')
    logger.info(f"🎉 处理完成！数据已保存至: {output_path}")

# ==========================================
# 3. 主程序入口
# ==========================================
def main():
    logger.info(f"📥 开始读取数据: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, sep='\t', header=0, low_memory=False)
    
    # 1. 基础解析 (String -> List)
    list_cols = ['ph', 'ph_durs', 'word_durs', 'ep_notedurs']
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(safe_literal_eval)
    
    # 2. 核心处理管线 (Pipeline)
    df = fill_missing_values(df)
    df = generate_features(df)
    df = filter_invalid_data(df)

    # 3. 可选：基于 *.npy 为每条样本生成 prompt 列
    # 如果你要用 latent 做 prompt，只需要把 src_col 改成 "latent_path"
    # 比如：df = add_prompt_column(df, src_col='latent_path', out_col='prompt_latent_paths')
    if 'mel_path' in df.columns:
        df = add_prompt_column(df, src_col='mel_path', out_col='prompt_mel_paths', max_prompts=1)
    
    # 4. 打印统计报告
    total_duration = df['duration'].sum()
    logger.info("========================================")
    logger.info(f"📊 最终剩余有效行数: {len(df)}")
    logger.info(f"⏱️ 最终剩余总时长: {total_duration / 3600:.2f} 小时 ({total_duration:.2f} 秒)")
    logger.info("========================================")
    
    # 5. 整理并保存
    reorder_and_save(df, OUTPUT_FILE)

if __name__ == "__main__":
    main()
