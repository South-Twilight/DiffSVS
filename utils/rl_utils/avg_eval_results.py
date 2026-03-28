import argparse
import json
import os
from typing import Dict, List, Tuple
 
 
def parse_args():
    p = argparse.ArgumentParser(description="Compute per-metric mean over utterances from final_results.*.txt")
    p.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="目录路径（包含 final_results.gpu.txt / final_results.cpu.txt）",
    )
    p.add_argument(
        "--out_json",
        type=str,
        default="avg_eval_results.json",
        help="输出 json 文件名（写到 eval_dir 下）",
    )
    return p.parse_args()
 
 
def _load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows
 
 
def _mean_metrics(rows: List[Dict]) -> Tuple[int, Dict[str, float]]:
    sums: Dict[str, float] = {}
    cnts: Dict[str, int] = {}
    for r in rows:
        for k, v in r.items():
            if k == "key":
                continue
            if isinstance(v, (int, float)):
                sums[k] = float(sums.get(k, 0.0)) + float(v)
                cnts[k] = int(cnts.get(k, 0)) + 1
    means = {k: (sums[k] / max(cnts.get(k, 0), 1)) for k in sums.keys()}
    return len(rows), means
 
 
def main():
    args = parse_args()
    eval_dir = args.eval_dir.rstrip("/")
    gpu_path = os.path.join(eval_dir, "final_results.gpu.txt")
    cpu_path = os.path.join(eval_dir, "final_results.cpu.txt")
 
    summary: Dict[str, Dict] = {}
    for tag, path in [("gpu", gpu_path), ("cpu", cpu_path)]:
        if not os.path.isfile(path):
            print(f"[{tag}] missing: {path}")
            continue
        rows = _load_jsonl(path)
        n, means = _mean_metrics(rows)
        print(f"[{tag}] utterances={n}")
        for k in sorted(means.keys()):
            print(f"  {k}: {means[k]:.6f}")
        summary[tag] = {"utterances": int(n), "means": means}

    out_path = os.path.join(eval_dir, args.out_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote summary json: {out_path}")
 
 
if __name__ == "__main__":
    main()

