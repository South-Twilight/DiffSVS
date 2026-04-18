"""
LoRA merge utility:
- Load base DiffSVS model from config + base ckpt
- Load LoRA adapter (PEFT save_pretrained directory)
- Merge adapter into backbone weights
- Export merged full-model ckpt for infer.py
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from peft import PeftModel

# Ensure project root (contains `ldm/`) is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ldm.util import instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base DiffSVS checkpoint")
    parser.add_argument("--config", type=str, required=True, help="训练/推理配置文件路径（支持 base_model_config 复用）")
    parser.add_argument("--base_ckpt", type=str, required=True, help="基础模型 ckpt 路径（训练 LoRA 时使用的 base 权重）")
    parser.add_argument("--adapter_dir", type=str, required=True, help="LoRA adapter 目录（包含 adapter_config.json）")
    parser.add_argument("--output_ckpt", type=str, default="", help="导出 merged ckpt 路径（默认写到 adapter_dir 同级）")
    return parser.parse_args()


def load_base_model(config_path: str, base_ckpt: str, device: torch.device):
    config = OmegaConf.load(config_path)
    base_model_config = config.get("base_model_config", None) if hasattr(config, "get") else None
    if base_model_config:
        base_cfg = OmegaConf.load(base_model_config)
        config = OmegaConf.merge(base_cfg, config)

    model = instantiate_from_config(config.model)
    ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt, strict=False)
    model = model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load base model
    model = load_base_model(args.config, args.base_ckpt, device)

    # 2) Load LoRA adapter onto backbone, then merge
    model.model = PeftModel.from_pretrained(model.model, args.adapter_dir)
    merged_backbone = model.model.merge_and_unload()
    model.model = merged_backbone

    # 3) Decide output path
    if args.output_ckpt:
        output_ckpt = args.output_ckpt
    else:
        adapter_dir = Path(args.adapter_dir).resolve()
        base_name = Path(args.base_ckpt).stem
        output_ckpt = str(adapter_dir.parent / f"merged_from-{base_name}_{adapter_dir.name}.pt")

    os.makedirs(str(Path(output_ckpt).parent), exist_ok=True)

    # 4) Save full model state dict for infer.py compatibility
    torch.save({"state_dict": model.state_dict()}, output_ckpt)
    print(f"Merged checkpoint saved: {output_ckpt}")


if __name__ == "__main__":
    main()
