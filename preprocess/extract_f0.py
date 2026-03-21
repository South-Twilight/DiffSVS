import os
import glob
import torch
import librosa
import logging
import numpy as np
import pandas as pd
import pyworld as pw
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

SR = 44100
HOP_SIZE = 2048   # 直接定义你希望的 F0 帧移
FRAME_PERIOD_MS = HOP_SIZE / SR * 1000.0


def process_single_file(task, target_dir):
    audio_path = task.get('audio_path')
    try:
        if audio_path is None:
            return False, "task 中缺少 audio_path"

        # 用音频文件名直接命名，不再依赖 latent_path
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        f0_save_path = os.path.join(target_dir, f"{base_name}_f0.npy")

        if os.path.exists(f0_save_path):
            return True, None

        # 1. 读取音频
        wav, sr = librosa.load(audio_path, sr=SR)

        # 2. 直接按固定 frame_period 提取 F0
        f0, t = pw.harvest(
            wav.astype(np.float64),
            sr,
            frame_period=FRAME_PERIOD_MS
        )
        f0 = f0.astype(np.float32)

        # 3. 可选：填充再恢复清音区
        f0_series = pd.Series(f0)
        f0_filled = f0_series.replace(0, np.nan).ffill().bfill().fillna(0).values.astype(np.float32)
        uv = (f0 > 0).astype(np.float32)

        f0_final = f0_filled * uv

        # 4. 保存
        np.save(f0_save_path, f0_final)
        return True, None

    except Exception as e:
        return False, f"[{audio_path}] 报错: {str(e)}"


def extract_f0_fast(data_dir, target_dir, max_workers=None):
    tsv_files = glob.glob(os.path.join(data_dir, '*.tsv'))
    if not tsv_files:
        print(f"❌ 未找到任何 .tsv 文件于 {data_dir}")
        return

    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)

    print(f"🚀 启动多进程模式，使用 {max_workers} 个 CPU 核心并行处理...")

    for tsv_path in tsv_files:
        print(f"\n📂 正在处理: {os.path.basename(tsv_path)}")
        df = pd.read_csv(tsv_path, sep='\t', low_memory=False)

        if 'audio_path' not in df.columns:
            print(f"❌ {tsv_path} 中不存在 audio_path 列")
            continue

        tasks = df.to_dict('records')
        success_count = 0
        error_count = 0

        executor = ProcessPoolExecutor(max_workers=max_workers)
        futures = []

        try:
            for task in tasks:
                futures.append(executor.submit(process_single_file, task, target_dir))

            with tqdm(total=len(tasks), desc="提取 F0") as pbar:
                for future in as_completed(futures):
                    success, error_msg = future.result()
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        logging.error(error_msg)
                    pbar.update(1)

            print(f"✅ {os.path.basename(tsv_path)} 处理完成！成功: {success_count}, 失败: {error_count}")
            executor.shutdown(wait=True)

        except KeyboardInterrupt:
            print("\n🛑 接收到中止信号 (Ctrl+C)！正在终止所有任务...")
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            os._exit(1)


if __name__ == "__main__":
    YOUR_DATA_DIR = "./data/final"
    TARGET_DIR = "./data/feat_extract/f0"

    os.makedirs(TARGET_DIR, exist_ok=True)
    extract_f0_fast(YOUR_DATA_DIR, TARGET_DIR, max_workers=16)
