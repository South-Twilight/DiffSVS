#!/usr/bin/env python3
"""统计 TSV 中各数据集的条数与总时长（秒）。

数据集名：item_name 去掉 '#' 及后缀后，取第一个 '_' 之前的部分
（如 m4singer_Tenor-7_... -> m4singer；opencpop#id -> opencpop）。
"""
import argparse

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tsv",
        default="/data5/tyx/DiffSVS/data/postprocess/data.tsv",
        help="TSV 路径",
    )
    args = p.parse_args()

    df = pd.read_csv(args.tsv, sep="\t", dtype=str, low_memory=False)
    if "duration" not in df.columns:
        raise SystemExit("缺少 duration 列")

    dur = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)

    if "item_name" in df.columns:
        # 先去掉 '#' 及之后（如 opencpop#id），再取第一个 '_' 之前的部分（如 m4singer_* -> m4singer）
        left = df["item_name"].astype(str).str.split("#", n=1).str[0]
        ds = left.str.split("_", n=1).str[0]
    elif "singer" in df.columns:
        ds = df["singer"].astype(str)
    else:
        raise SystemExit("需要 item_name 或 singer 列以区分数据集")

    g = dur.groupby(ds, sort=True)
    counts = g.size()
    totals = g.sum()

    print(f"文件: {args.tsv}")
    print(f"数据集数: {len(counts)}")
    print()
    print(f"{'dataset':<32} {'count':>12} {'duration_s':>16}")
    for name in counts.index:
        print(f"{name:<32} {int(counts[name]):>12} {float(totals[name]):>16.4f}")
    print("-" * 64)
    print(f"{'TOTAL':<32} {int(counts.sum()):>12} {float(totals.sum()):>16.4f}")


if __name__ == "__main__":
    main()
