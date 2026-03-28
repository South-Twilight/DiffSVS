"""
DiffSVS 推理脚本：加载 config/ckpt，读 manifest 乐谱，CFM 采样得到 latent，
除以 scale_factor 后直接送 VAE 解码为波形并保存。结果保存到 outputs/{model}/epoch={epoch}/。
"""
import os
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed import init_process_group
from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import OmegaConf
from tqdm import tqdm
import soundfile as sf

from ldm.models.diffusion.cfm1_audio_sampler import CFMSampler
from ldm.util import instantiate_from_config
from ldm.dataset.diffsvs_dataset import PHN_PAD_ID
from ldm.dataset.diffsvs_infer_dataset import (
    DiffSVSInferConditionDataset as DiffSVSEvalDataset,
    infer_condition_collate_fn as eval_collate_fn,
)


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="DiffSVS inference")
    parser.add_argument("--config", type=str, default="configs/diff_cfm.yaml", help="模型 config")
    parser.add_argument("--ckpt", type=str, required=True, help="模型权重路径")
    parser.add_argument("--manifest_path", type=str, default="data/final_test/test.tsv")
    parser.add_argument("--ddim_steps", type=int, default=50, help="ODE 积分步数")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--scale", type=float, default=4.0, help="CFG 引导系数")
    parser.add_argument("--scales", type=str, default="", help="多尺度，如 2.0-4.0-6.0")
    parser.add_argument("--save_dir", type=str, default="", help="覆盖时的根目录，默认不填则用 outputs/{model}/epoch={epoch}")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--max_eval", type=int, default=50, help="最多评测条数，0 表示全部")
    parser.add_argument("--max_duration", type=float, default=20.0, help="过滤超过该时长的样本")
    parser.add_argument(
        "--use_teacher_forcing",
        action="store_true",
        help="使用 GT latent + GT f0（若存在）进行 teacher forcing 推理，跳过 CFM 采样",
    )
    return parser.parse_args()


def safe_path(path):
    os.makedirs(Path(path).parent, exist_ok=True)
    return path


def get_file_logger(log_path: str, logger_name: str) -> logging.Logger:
    """
    为推理过程创建/复用文件 logger（仅写文件，不污染控制台）。
    多进程下请确保日志文件名包含 rank，避免写冲突。
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 可能会被重复调用（多 scale / 多 batch），先清理旧 handler
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    os.makedirs(Path(log_path).parent, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def normalize_loudness(wav, target_loudness=-23):
    rms = np.sqrt(np.mean(wav.astype(np.float64) ** 2) + 1e-8)
    loudness = 20 * np.log10(rms)
    gain = target_loudness - loudness
    return wav * (10 ** (gain / 20))


def cond_to_device(cond, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in cond.items()}


def maybe_load_prompt_latent(prompt_latent_path, device, max_prompt_len=100):
    if prompt_latent_path is None:
        return None, None
    prompt_np = np.load(prompt_latent_path).astype(np.float32)  # [C, T_p]
    prompt_t = torch.from_numpy(prompt_np).unsqueeze(0).to(device)  # [1, C, T_p]
    if prompt_t.shape[2] > max_prompt_len:
        prompt_t = prompt_t[:, :, :max_prompt_len]
    return prompt_t, prompt_np


def save_audio_pair(logger, save_dir, item_name, pred_wav, gt_wav_path):
    wav = normalize_loudness(pred_wav.squeeze().cpu().numpy(), -23)
    pred_path = safe_path(os.path.join(save_dir, f"{item_name}_pred.wav"))
    sf.write(pred_path, wav, 44100, subtype="PCM_16")
    logger.info(
        "Saved pred wav: %s (len=%d, rms=%.6f)",
        pred_path,
        len(wav),
        float(np.sqrt(np.mean(wav.astype(np.float64) ** 2) + 1e-12)),
    )

    try:
        gt_wav, _ = sf.read(gt_wav_path)
        gt_wav = normalize_loudness(gt_wav, -23)
    except Exception:
        gt_wav = np.zeros(1000, dtype=np.float32)
    gt_path = safe_path(os.path.join(save_dir, f"{item_name}_gt.wav"))
    sf.write(gt_path, gt_wav, 44100, subtype="PCM_16")
    logger.info("Saved gt wav: %s (len=%d)", gt_path, len(gt_wav))
    return pred_path


# 推理用 Dataset / collate：ldm.dataset.diffsvs_infer_dataset

# ==========================================
# 模型加载与生成
# ==========================================
def load_model_and_sampler(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    model.eval()
    num_timesteps = getattr(model, "num_timesteps", config.model.params.get("timesteps", 1000))
    sampler = CFMSampler(model, num_timesteps=num_timesteps)
    sampling_cfg = config.model.params.get("sampling_config", None)
    if sampling_cfg is not None:
        sampling_cfg = OmegaConf.to_container(sampling_cfg, resolve=True)
    else:
        sampling_cfg = {}
    return sampler, sampling_cfg


@torch.no_grad()
def run_inference(rank, args):
    if args.num_gpus > 1:
        init_process_group(
            backend="nccl",
            init_method="tcp://localhost:54321",
            world_size=args.num_gpus,
            rank=rank,
        )
    device = torch.device(f"cuda:{rank}")
    sampler, sampling_cfg = load_model_and_sampler(args.config, args.ckpt, device)
    model = sampler.model

    dataset = DiffSVSEvalDataset(
        args.manifest_path,
        max_eval=args.max_eval,
        max_duration=args.max_duration,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=eval_collate_fn,
        sampler=DistributedSampler(dataset, shuffle=False) if args.num_gpus > 1 else None,
    )
    if rank == 0:
        loader = tqdm(loader, desc="infer")

    scales = [float(x) for x in args.scales.split("-")] if args.scales else [args.scale]
    csv_rows = []
    wavscp_rows = []

    # 运行级别日志（写到输出根目录）
    run_log_path = os.path.join(args.save_dir, f"infer_rank{rank}.log")
    run_logger = get_file_logger(run_log_path, logger_name=f"infer_rank{rank}")
    run_logger.info(
        "Start inference: config=%s ckpt=%s manifest=%s ddim_steps=%d n_samples=%d scales=%s use_teacher_forcing=%s",
        args.config, args.ckpt, args.manifest_path, args.ddim_steps, args.n_samples, scales, args.use_teacher_forcing
    )
    run_logger.info(
        "save_dir=%s max_eval=%d max_duration=%.3f num_gpus=%d",
        args.save_dir, args.max_eval, args.max_duration, args.num_gpus
    )

    for batch_idx, (cond, names, audio_paths, latent_paths, f0_paths, prompt_mel_paths) in enumerate(loader):
        item_name = names[0]
        gt_wav_path = audio_paths[0]
        latent_path = latent_paths[0] if latent_paths[0] else None
        f0_path = f0_paths[0] if f0_paths and f0_paths[0] else None
        prompt_latent_path = prompt_mel_paths[0] if prompt_mel_paths and prompt_mel_paths[0] else None

        run_logger.info(
            "[batch_idx=%d] item_name=%s gt_wav=%s latent_path=%s f0_path=%s prompt_latent_path=%s",
            batch_idx, item_name, gt_wav_path, latent_path, f0_path, prompt_latent_path
        )

        cond = cond_to_device(cond, device)

        for scale in scales:
            # 每个 scale 的音频输出目录下单独写日志
            save_dir = os.path.join(args.save_dir, f"scale_{scale}_steps={args.ddim_steps}")
            logger = run_logger
            logger.info("Begin scale=%s ddim_steps=%d", scale, args.ddim_steps)

            if args.use_teacher_forcing:
                # 直接读取 GT latent（未归一化的 128 维 [mean, scale]），并可选加载 GT f0，跳过 CFM 采样
                if latent_path is None:
                    raise ValueError("teacher_forcing 模式需要 test.tsv 中包含 latent 路径（如 latent_path 或 mel_path 列）。")
                latent_np = np.load(latent_path).astype(np.float32)  # [128, T]
                z_raw = torch.from_numpy(latent_np).unsqueeze(0).to(device)
                mean, scale_param = torch.chunk(z_raw, 2, dim=1)
                if f0_path is not None and os.path.isfile(f0_path):
                    f0_np = np.load(f0_path).astype(np.float32)
                    f0_t = torch.from_numpy(f0_np).view(1, 1, -1).to(device)
                    cond["f0_gt"] = F.interpolate(
                        f0_t, size=mean.shape[-1], mode="linear", align_corners=False
                    )
                else:
                    f0_np = None
                logger.info(
                    "teacher_forcing: latent_np_shape=%s mean_shape=%s scale_shape=%s f0_loaded=%s",
                    latent_np.shape, tuple(mean.shape), tuple(scale_param.shape), bool(f0_np is not None)
                )
                latents_vae, _ = model.first_stage_model.vae_sample(mean, scale_param)
                wav_preds = model.first_stage_model.decode(latents_vae)
            else:
                # 用 frontend 预测时长得到 latent 帧数，与 apply_model 内 infer 逻辑一致
                padding_mask = cond["phn"] == PHN_PAD_ID
                pred_dur_log = model.frontend(
                    cond["phn"], cond["notedurs"], cond["pitches"], cond["notetypes"], padding_mask
                )
                dur_pred = torch.clamp(torch.round(torch.exp(pred_dur_log) - 1), min=1).long()
                latent_length = int(dur_pred.sum().item())
                latent_channels = 128  # 与 config data.params.latent_channels / VAE 输出一致
                shape = (latent_channels, latent_length)
                logger.info(
                    "cfm sampling: latent_length=%d latent_channels=%d start_shape=%s",
                    latent_length, latent_channels, shape
                )

                uc_cond = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in cond.items()}
                uc_cond["spk_id"] = torch.zeros_like(cond["spk_id"])
                uc_cond["infer"] = True

                # dataset/collate 已经可能提供 cond["prompt_latent"]（含 padding）
                # 只有在 cond 未提供 prompt_latent 时才 fallback 从 prompt_latent_path 读取。
                if cond.get("prompt_latent", None) is None and prompt_latent_path is not None:
                    # 若提供了 prompt latent，则加载并在时间维上裁剪到最多 100 帧
                    # 与训练集 DiffSVSDataset 中 max_prompt_len=100 的逻辑对齐：
                    # - 训练时随机裁一段长度为 100
                    # - 推理时简单使用前 100 帧作为 prompt 参考
                    prompt_t, prompt_np = maybe_load_prompt_latent(
                        prompt_latent_path, device, max_prompt_len=100
                    )
                    _, _, T_p = prompt_t.shape
                    logger.info(
                        "prompt: prompt_np_shape=%s orig_T_p=%d used_T_p=%d max_prompt_len=%d",
                        prompt_np.shape,
                        prompt_np.shape[1],
                        prompt_t.shape[2],
                        100,
                    )
                    cond["prompt_latent"] = prompt_t
                    uc_cond["prompt_latent"] = prompt_t

                start_code = torch.randn(args.n_samples, latent_channels, latent_length, device=device)
                logger.info(
                    "start_code: shape=%s n_samples=%d unconditional_guidance_scale=%s",
                    tuple(start_code.shape), args.n_samples, scale
                )

                samples_ddim, _, f0_pred = sampler.sample_cfg(
                    cond=cond,
                    batch_size=args.n_samples,
                    shape=shape,
                    timesteps=args.ddim_steps,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc_cond,
                    x_latent=start_code,
                    sampler_type=sampling_cfg.get("sampler_type", "ode"),
                    sigma=sampling_cfg.get("sigma", 0.0),
                    sigma_schedule=sampling_cfg.get("sigma_schedule", "constant"),
                    sigma_min=sampling_cfg.get("sigma_min", 0.0),
                    noise_start_t=sampling_cfg.get("noise_start_t", 0.05),
                    noise_stop_t=sampling_cfg.get("noise_stop_t", 0.95),
                )
                logger.info(
                    "cfm done: samples_ddim_shape=%s f0_pred_type=%s",
                    tuple(samples_ddim.shape), type(f0_pred).__name__
                )
                # 采样结果除以 scale_factor 后即为 128 维 [mean(64), scale(64)]，直接送 VAE
                z_raw = samples_ddim / model.scale_factor
                mean, scale_param = torch.chunk(z_raw, 2, dim=1)
                # latents_vae, _ = model.first_stage_model.vae_sample(mean, scale_param)
                latents_vae = mean
                wav_preds = model.first_stage_model.decode(latents_vae)
                logger.info('\n\n')

            # 将 ddim_steps 体现在 scale 目录中，便于区分同一 ckpt 下不同采样步数的结果
            for wav_t in wav_preds:
                pred_path = save_audio_pair(logger, save_dir, item_name, wav_t, gt_wav_path)
                csv_rows.append({"audio_path": pred_path, "name": item_name, "scale": scale})
                wavscp_rows.append(f"{item_name} {os.path.abspath(pred_path)}")
                logger.info("Append csv row: name=%s scale=%s", item_name, scale)

    if rank == 0:
        csv_path = os.path.join(args.save_dir, "inference_results.csv")
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        wavscp_path = os.path.join(args.save_dir, "wav.scp")
        with open(wavscp_path, "w", encoding="utf-8") as f:
            for line in wavscp_rows:
                f.write(line + "\n")
        print(f"推理完成，wavscp: {wavscp_path}")
        print(f"推理完成，结果清单: {csv_path}")


if __name__ == "__main__":
    args = parse_args()
    # 默认保存路径：exp_outputs/{model}/epoch={epoch}，便于区分不同 ckpt
    if not args.save_dir:
        ckpt_path = Path(args.ckpt)
        # 1) 模型目录：优先从 checkpoints 上一级目录获取 run 名
        model_part = None
        for parent in ckpt_path.parents:
            if parent.name == "checkpoints" and parent.parent is not None:
                # logs/{run}/checkpoints/(trainstep_checkpoints)/xxx.ckpt -> {run}
                model_part = parent.parent.name
                break
        if model_part is None:
            # 兜底：直接用 ckpt 所在目录名
            model_part = ckpt_path.parent.name or "default"

        # 2) ckpt 标识：仅使用 epoch/step，采样步数下沉到 scale 目录名中
        epoch_part = ckpt_path.stem  # 例如 epoch=000064 或 epoch=000049-step=000040000
        args.save_dir = os.path.join("exp_outputs", model_part, epoch_part)
    else:
        args.save_dir = args.save_dir.rstrip("/")
    
    make_dirs(args.save_dir)

    if args.num_gpus > 1:
        mp.spawn(run_inference, nprocs=args.num_gpus, args=(args,))
    else:
        run_inference(0, args)
