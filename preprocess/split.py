import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_dataset(
    input_tsv, output_dir, test_ids, valid_ids=None,
    other_ratios=(0.9, 0.05, 0.05),
    random_seed=42
):
    """
    划分数据集：
    - opencpop/acesinger: 按照给定的 ID 列表严格划分 test (和 valid)。
    - 其他数据集: 按 other_ratios (Train, Valid, Test) 的比例随机划分。
    """
    logger.info(f"📥 正在读取最终数据集: {input_tsv}")
    df = pd.read_csv(input_tsv, sep='\t', header=0, low_memory=False)
    total_rows = len(df)
    
    # ---------------------------------------------------------
    # 1. 区分特殊数据集 (Opencpop/Acesinger) 和 其他数据集
    # ---------------------------------------------------------
    # 只要 item_name 里包含 opencpop 或 acesinger 就归入固定划分组
    is_fixed_group = df['item_name'].str.contains('opencpop|acesinger', case=False, na=False)
    df_fixed = df[is_fixed_group].copy()
    df_other = df[~is_fixed_group].copy()
    
    logger.info(f"🔍 识别到 {len(df_fixed)} 条 Opencpop/Acesinger 数据，{len(df_other)} 条其他数据。")

    # ---------------------------------------------------------
    # 2. 处理 Opencpop/Acesinger (固定划分)
    # ---------------------------------------------------------
    # 使用正则快速匹配 item_name 中是否包含指定的 ID
    test_pattern = '|'.join(test_ids)
    is_fixed_test = df_fixed['item_name'].str.contains(test_pattern, na=False)
    
    test_fixed = df_fixed[is_fixed_test]
    remaining_fixed = df_fixed[~is_fixed_test]
    
    # 如果你也提供了 opencpop 的固定验证集 ID 列表
    if valid_ids and len(valid_ids) > 0:
        valid_pattern = '|'.join(valid_ids)
        is_fixed_valid = remaining_fixed['item_name'].str.contains(valid_pattern, na=False)
        valid_fixed = remaining_fixed[is_fixed_valid]
        train_fixed = remaining_fixed[~is_fixed_valid]
    else:
        # 如果没有指定固定的 valid，从剩余的固定组里随机抽 5% 作为验证集，其余作为训练集
        logger.info("未提供 fixed 验证集 ID，将从 Opencpop/Acesinger 剩余数据中随机抽取 5% 作为验证集。")
        train_fixed, valid_fixed = train_test_split(remaining_fixed, test_size=0.05, random_state=random_seed)

    # ---------------------------------------------------------
    # 3. 处理其他数据集 (随机划分)
    # ---------------------------------------------------------
    train_ratio, valid_ratio, test_ratio = other_ratios
    # sklearn 的 train_test_split 需要分两步来切分三份
    if len(df_other) > 0:
        # 第一次切分：分离出 Train，剩下的是 Valid + Test
        test_valid_size = valid_ratio + test_ratio
        train_other, temp_other = train_test_split(df_other, test_size=test_valid_size, random_state=random_seed)
        
        # 第二次切分：把剩下的按比例切成 Valid 和 Test
        relative_test_size = test_ratio / test_valid_size
        valid_other, test_other = train_test_split(temp_other, test_size=relative_test_size, random_state=random_seed)
    else:
        # 如果没有其他数据，就建空表
        train_other = pd.DataFrame(columns=df.columns)
        valid_other = pd.DataFrame(columns=df.columns)
        test_other = pd.DataFrame(columns=df.columns)

    # ---------------------------------------------------------
    # 4. 合并两部分数据并打乱顺序 (Shuffle)
    # ---------------------------------------------------------
    final_train = pd.concat([train_fixed, train_other]).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    final_valid = pd.concat([valid_fixed, valid_other]).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    final_test = pd.concat([test_fixed, test_other]).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # ---------------------------------------------------------
    # 5. 落盘保存
    # ---------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    final_train.to_csv(os.path.join(output_dir, 'train.tsv'), index=False, sep='\t')
    final_valid.to_csv(os.path.join(output_dir, 'valid.tsv'), index=False, sep='\t')
    final_test.to_csv(os.path.join(output_dir, 'test.tsv'), index=False, sep='\t')

    # 打印最终报告
    logger.info("========================================")
    logger.info("✅ 数据集划分完成！")
    logger.info(f"📊 训练集 (Train): {len(final_train)} 条 ({len(final_train)/total_rows:.1%})")
    logger.info(f"📊 验证集 (Valid): {len(final_valid)} 条 ({len(final_valid)/total_rows:.1%})")
    logger.info(f"📊 测试集 (Test) : {len(final_test)} 条 ({len(final_test)/total_rows:.1%})")
    logger.info(f"📁 文件已保存至目录: {output_dir}")
    logger.info("========================================")


if __name__ == "__main__":
    # --- 1. 配置路径 ---
    INPUT_TSV = "./data/postprocess/data_test.tsv"  # 你刚才最后一步生成的纯净文件
    OUTPUT_DIR = "./data/final_test"        # 切分后的三个文件存放位置
    
    # --- 2. 配置 Opencpop/Acesinger 的官方测试集 ID 列表 ---
    # 把你需要分到测试集的 ID 填在这里（注意：只需要填数字部分即可，脚本会自动匹配）
    OPENCPOP_TEST_IDS = [
        "204400", 
        "208600",
        "209200",
        "209300",
        "210000",
    ]
    
    # (可选) 如果 Opencpop 也有官方验证集 ID，填在这里；如果没有，留空 []，脚本会自动抽取
    OPENCPOP_VALID_IDS = [] 
    
    # --- 3. 配置其他数据集的随机划分比例 (Train, Valid, Test) ---
    OTHER_DATA_RATIO = (0.90, 0.05, 0.05)
    
    split_dataset(
        input_tsv=INPUT_TSV,
        output_dir=OUTPUT_DIR,
        test_ids=OPENCPOP_TEST_IDS,
        valid_ids=OPENCPOP_VALID_IDS,
        other_ratios=OTHER_DATA_RATIO
    )