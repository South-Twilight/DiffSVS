import argparse
import logging
import os
import random
from datetime import datetime
import json
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import soundfile as sf
import wandb

from ldm.util import instantiate_from_config
from ldm.dataset.diffsvs_dataset import PHN_PAD_ID
from ldm.dataset.diffsvs_infer_dataset import (
    DiffSVSInferConditionDataset as DiffSVSEvalDataset,
    infer_condition_collate_fn as eval_collate_fn,
)
from torch.utils.data import DataLoader
from ldm.models.diffusion.cfm1_audio_sampler import CFMSampler
from utils.rl_utils.reward_utils import load_reward_yaml_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def normalize_loudness(wav: np.ndarray, target_loudness: float = -23.0) -> np.ndarray:
    """Normalize wave loudness in the same way as infer.py (for debug audio inspection)."""
    wav = wav.astype(np.float32)
    rms = np.sqrt(np.mean(wav.astype(np.float64) ** 2) + 1e-8)
    loudness = 20 * np.log10(rms)
    gain = target_loudness - loudness
    return wav * (10.0 ** (gain / 20.0))


def save_debug_waves(
    *,
    debug_save_dir: str,
    base_name: str,
    group_data: dict,
    sample_rate: int = 44100,
):
    """Save generated waves so you can verify wave[i] aligns with reward[i]/advantage[i]."""
    if "wave" not in group_data:
        return

    wave: torch.Tensor = group_data["wave"]  # [G, T]
    reward = group_data.get("reward", None)

    # Ensure output dir exists
    os.makedirs(debug_save_dir, exist_ok=True)

    wave_cpu = wave.detach().cpu()
    rewards_cpu = reward.detach().cpu() if isinstance(reward, torch.Tensor) else None

    G = wave_cpu.size(0)
    for g in range(G):
        wav_g = wave_cpu[g].numpy()
        wav_norm = normalize_loudness(wav_g, -23.0)

        r_str = f"{float(rewards_cpu[g]):.4f}" if rewards_cpu is not None else "na"
        out_path = os.path.join(
            debug_save_dir,
            # Simple & sortable: keep only g/r for group-wise inspection
            f"{base_name}_g{g}_r{r_str}.wav",
        )
        sf.write(out_path, wav_norm, sample_rate, subtype="PCM_16")


@torch.no_grad()
def decode_latent_to_wave(
    *,
    vae,
    latent: "np.ndarray | torch.Tensor",
    device: torch.device,
) -> torch.Tensor:
    """
    将 latent（[C, T] = [mean(64), scale(64)]）解码为波形。
    - 支持输入 np.ndarray 或 torch.Tensor
    - latent 语义应与 infer.py teacher_forcing 一致：未乘/未除 scale_factor 的 [mean, scale]
    返回: wave [1, T_wav]
    """
    if isinstance(latent, np.ndarray):
        z_raw = torch.from_numpy(latent.astype(np.float32))
    elif isinstance(latent, torch.Tensor):
        z_raw = latent.detach().to(torch.float32)
    else:
        raise ValueError(f"latent must be np.ndarray or torch.Tensor, got {type(latent).__name__}")

    if z_raw.dim() == 3:
        # [G, C, T] -> 取第一个
        z_raw = z_raw[0]
    if z_raw.dim() != 2:
        raise ValueError(f"latent must be [C,T] or [G,C,T], got {tuple(z_raw.shape)}")

    z_raw = z_raw.unsqueeze(0).to(device)  # [1, C, T]
    mean, scale_param = torch.chunk(z_raw, 2, dim=1)
    if hasattr(vae, "vae_sample"):
        latents_vae, _ = vae.vae_sample(mean, scale_param)
    else:
        latents_vae = mean
    wave = vae.decode(latents_vae).squeeze(1)  # [1, T_wav]
    return wave

# ============================================================================
# 【1. 冻结前端域 (Frozen Frontend)】
# ============================================================================
class FrozenFrontend:
    """
    绝对静止的前端，严禁保留计算图。
    负责提取 prompt_latent，并经过 Duration 扩展对齐 Music/Phoneme 联合特征。
    """
    def __init__(self, policy_model, *, duration_min: int = 1):
        self.policy_model = policy_model
        self.duration_min = int(duration_min)
        # 仅冻结并固定 frontend，避免在这里全局切换 policy_model 模式
        self.policy_model.frontend.eval()
        for p in self.policy_model.frontend.parameters():
            p.requires_grad = False
            
    @torch.no_grad()
    def process(self, batch_cond):
        """
        输入: 带有 padding 的 batch condition
        输出: 对齐后的特征 aligned_conditions 以及无 padding 的有效长度 cond_lens
        """
        phn = batch_cond["phn"].long()
        midi = batch_cond["pitches"].long()
        notedurs = batch_cond["notedurs"].float()
        notetypes = batch_cond["notetypes"].long()
        padding_mask = (phn == PHN_PAD_ID)
        
        # 冻结 frontend 情况下无需开启 train mode，始终 eval 
        pred_dur_log = self.policy_model.frontend(phn, notedurs, midi, notetypes, padding_mask)
        dur_raw = torch.round(torch.exp(pred_dur_log) - 1).long()  # 允许为 0（用于计算真实长度）

        # dur_gt：用于 apply_model 在 infer=False 分支直接 length_regulator
        # （apply_model 内部会再 clamp(min=1)；这里统一按 duration_min 下界构造）
        dur_raw = torch.clamp(dur_raw, min=self.duration_min).long()

        # cond_lens：用于 unbatch / latent padding mask 的“真实有效帧长”
        # 必须与 apply_model 的 length_regulator 对齐：这里用 dur_gt 后的 phn_aligned 计数，
        # 确保非 padding 帧长度与 apply_model 内部截断/填充一致。
        phn_aligned = self.policy_model.length_regulator(
            phn, dur_raw, padding_value=int(PHN_PAD_ID)
        )
        cond_lens_valid = (phn_aligned != PHN_PAD_ID).sum(dim=1).clamp(min=1)

        aligned_conditions = {
            "phn": phn,
            "pitches": midi,
            "notedurs": notedurs,
            "notetypes": notetypes,
            "dur_gt": dur_raw,
            "spk_id": batch_cond.get("spk_id"),
            # 可选：f0_gt/prompt 在部分数据源可能不存在
            "f0_gt": batch_cond.get("f0_gt", None),
            "prompt_latent": batch_cond.get("prompt_latent", None),
            "infer": False,
        }

        return {
            "aligned_conditions": aligned_conditions,
            "cond_lens": cond_lens_valid,  # [B] 真实有效帧长
        }


# ============================================================================
# 【2. 经验采样域 (Rollout Engine)】
# ============================================================================
class RolloutEngine:
    """
    严格 @torch.no_grad()
    动作空间: 纯 Latent Space。
    采样过程: ODE -> SDE -> ODE (生成 target_latent)。
    数据捕获: 针对 Girsanov 定理，在 SDE 阶段子采样并保存所需的 (x_t, t, dt, v_old, noise)。
    """
    def __init__(
        self,
        policy_model,
        *,
        G: int = 8,
        timesteps: int = 25,
        noise_start_t: float = 0.05,
        noise_stop_t: float = 0.95,
        sigma: float = 0.7,
        sigma_schedule: str = "flow_grpo",
        sampler_type: str = "sde",
        score_denom_eps: float = 1.0e-4,
        lora_manager=None,
        debug_sde_diff: bool = False,
        debug_sde_dir: str = "debug_outputs/grpo_sde_vis",
        debug_sde_max_groups: int = 4,
    ):
        self.policy_model = policy_model
        self.G = int(G)
        self.timesteps = int(timesteps)
        self.noise_start_t = float(noise_start_t)
        self.noise_stop_t = float(noise_stop_t)
        self.score_denom_eps = float(score_denom_eps)
        self.sigma = float(sigma)
        self.sigma_schedule = str(sigma_schedule)
        self.sampler_type = str(sampler_type)
        self.sampler = CFMSampler(policy_model, num_timesteps=policy_model.num_timesteps)
        self.lora_manager = lora_manager
        if self.lora_manager is None:
            raise ValueError("RolloutEngine requires lora_manager for actor/ref LoRA switching.")
        self.debug_sde_diff = bool(debug_sde_diff)
        self.debug_sde_dir = str(debug_sde_dir)
        self.debug_sde_max_groups = int(debug_sde_max_groups)
        self._debug_group_counter = 0

    def _pairwise_l2(self, x_flat: torch.Tensor) -> torch.Tensor:
        # x_flat: [G, D]
        diff = x_flat.unsqueeze(1) - x_flat.unsqueeze(0)
        return torch.sqrt(torch.clamp((diff * diff).sum(dim=-1), min=0.0))

    def _save_sde_diff_debug(
        self,
        *,
        prompt_idx: int,
        t_cur: torch.Tensor,
        sigma_t: torch.Tensor,
        dt: torch.Tensor,
        noise: torch.Tensor,
        stochastic_increment: torch.Tensor,
    ):
        if not self.debug_sde_diff:
            return
        if self._debug_group_counter >= self.debug_sde_max_groups:
            return
        self._debug_group_counter += 1

        os.makedirs(self.debug_sde_dir, exist_ok=True)
        group_tag = f"group_{self._debug_group_counter:03d}_p{prompt_idx:03d}"
        group_dir = os.path.join(self.debug_sde_dir, group_tag)
        os.makedirs(group_dir, exist_ok=True)

        # Flatten for pairwise distance analysis
        noise_flat = noise.detach().float().cpu().reshape(noise.size(0), -1)
        inc_flat = stochastic_increment.detach().float().cpu().reshape(stochastic_increment.size(0), -1)
        d_noise = self._pairwise_l2(noise_flat)
        d_inc = self._pairwise_l2(inc_flat)

        # "是不是一样" 检查：与第一个样本比较最大绝对差
        noise_max_abs_diff_vs0 = float((noise_flat - noise_flat[0:1]).abs().max().item())
        inc_max_abs_diff_vs0 = float((inc_flat - inc_flat[0:1]).abs().max().item())

        summary = {
            "prompt_idx": int(prompt_idx),
            "G": int(noise.size(0)),
            "t_cur": float(t_cur.detach().cpu().item()),
            "dt": float(dt.detach().cpu().item()),
            "sigma_t": float(sigma_t.detach().cpu().item()),
            "noise_pairwise_l2_mean": float(d_noise.mean().item()),
            "noise_pairwise_l2_max": float(d_noise.max().item()),
            "increment_pairwise_l2_mean": float(d_inc.mean().item()),
            "increment_pairwise_l2_max": float(d_inc.max().item()),
            "noise_max_abs_diff_vs_sample0": noise_max_abs_diff_vs0,
            "increment_max_abs_diff_vs_sample0": inc_max_abs_diff_vs0,
            "noise_identical_all_G": bool(noise_max_abs_diff_vs0 < 1e-8),
            "increment_identical_all_G": bool(inc_max_abs_diff_vs0 < 1e-8),
        }
        with open(os.path.join(group_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # Save raw matrices for further analysis
        np.save(os.path.join(group_dir, "noise_pairwise_l2.npy"), d_noise.numpy())
        np.save(os.path.join(group_dir, "increment_pairwise_l2.npy"), d_inc.numpy())

        # Optional visualization (if matplotlib exists)
        try:
            import matplotlib.pyplot as plt  # type: ignore

            def save_heatmap(mat: np.ndarray, title: str, out_png: str):
                plt.figure(figsize=(5, 4))
                plt.imshow(mat, aspect="auto")
                plt.colorbar()
                plt.title(title)
                plt.xlabel("sample_j")
                plt.ylabel("sample_i")
                plt.tight_layout()
                plt.savefig(out_png, dpi=150)
                plt.close()

            save_heatmap(d_noise.numpy(), "Pairwise L2 of recovered noise", os.path.join(group_dir, "noise_pairwise_l2.png"))
            save_heatmap(d_inc.numpy(), "Pairwise L2 of stochastic increment", os.path.join(group_dir, "increment_pairwise_l2.png"))
        except Exception:
            # No matplotlib or plotting failed; summary/npy are enough.
            pass
        
    @torch.no_grad()
    def rollout(self, frontend_output):
        aligned_cond = frontend_output["aligned_conditions"]
        cond_lens = frontend_output["cond_lens"]
        B = cond_lens.size(0)
        device = cond_lens.device
        
        unbatched_data = []
        
        # 将 Batch 拆开，逐 Prompt 进行独立采样，彻底杜绝时间维 padding 污染。
        # 由于我们针对每个 Prompt 并行了 G 条轨迹，GPU 利用率依然极高。
        for b in range(B):
            T_exact = int(cond_lens[b].item())
            T_exact = max(T_exact, 1)
            
            # 1. 取出当前 b 的条件，并扩展为 G 份
            cond_b = {}
            for k, v in aligned_cond.items():
                if isinstance(v, torch.Tensor):
                    cond_b[k] = v[b:b+1].repeat_interleave(self.G, dim=0)
                else:
                    cond_b[k] = v
                    
            C_mel = self.policy_model.channels if getattr(self.policy_model, "channels", 0) > 0 else self.policy_model.mel_dim
            shape = (self.G, C_mel, T_exact)
            
            # 组内不共享的初始噪声（每条轨迹各自独立）
            x_start = torch.randn(self.G, C_mel, T_exact, device=device)
            
            wrapper = self.sampler.ode_wrapper(cond_b)
            
            # 启用 LoRA 确保 SDE 探索轨迹来自 Actor
            self.lora_manager.enable()
            
            # 2. 调用 sampler 提供的 sample_loop
            x_final, traj = self.sampler._sample_loop(
                wrapper=wrapper,
                shape=shape,
                x_latent=x_start,  # 传入共享初始噪声
                timesteps=self.timesteps,
                sampler_type=self.sampler_type,
                sigma=self.sigma,
                sigma_schedule=self.sigma_schedule,
                noise_start_t=self.noise_start_t,
                noise_stop_t=self.noise_stop_t,
                score_denom_eps=self.score_denom_eps,
            )
            
            t_span = torch.linspace(0.0, 1.0, self.timesteps, device=device)
            # Determine SDE phase indices
            sde_indices = []
            for i in range(len(t_span) - 1):
                if self.noise_start_t <= t_span[i] <= self.noise_stop_t:
                    sde_indices.append(i)
                    
            # Randomly select one time step to save for RL update
            save_idx = random.choice(sde_indices) if sde_indices else 0
            t_cur = t_span[save_idx]
            t_next = t_span[save_idx + 1]
            dt = t_next - t_cur
            dt_sqrt = torch.sqrt(torch.clamp(dt, min=0.0))
            
            x_t = traj[save_idx]
            
            # 收集 Actor 的速度场
            v_old = wrapper(t_cur, x_t, None)
            
            sigma_t = self.sampler._get_sigma_t(
                t_cur,
                sigma=torch.tensor(self.sigma, device=device),
                sigma_schedule=self.sigma_schedule,
            )
            one_minus_t = torch.clamp(1.0 - t_cur, min=self.score_denom_eps)
            score = (t_cur / one_minus_t) * v_old - (x_t / one_minus_t)
            drift = v_old + 0.5 * (sigma_t ** 2) * score
            
            x_next = traj[save_idx + 1]
            
            # 此时 x_t 已经是精确长度 T_exact，无需额外的 padding mask！
            if (sigma_t > 0).item() and dt_sqrt.item() > 0:
                noise = (x_next - x_t - drift * dt) / (sigma_t * dt_sqrt)
            else:
                noise = torch.zeros_like(x_t)
            
            # debug
            stochastic_increment = x_next - x_t - drift * dt
            self._save_sde_diff_debug(
                prompt_idx=b,
                t_cur=t_cur,
                sigma_t=sigma_t,
                dt=dt,
                noise=noise,
                stochastic_increment=stochastic_increment,
            )
            
            # 获取 v_ref
            self.lora_manager.disable()
            
            ref_wrapper = self.sampler.ode_wrapper(cond_b)
            v_ref = ref_wrapper(t_cur, x_t, None)
            
            # 重新开启 LoRA
            self.lora_manager.enable()
            
            group_dict = {
                "cond": cond_b,
                "t": t_cur,
                "dt": dt,
                "x_t": x_t.clone(),
                "noise": noise.clone(),
                "v_old": v_old.clone(),
                "v_ref": v_ref.clone(),
                "x_final": x_final.clone(),
            }
            unbatched_data.append(group_dict)
            
        return unbatched_data


# ============================================================================
# 【3. 奖励评估域 (Reward Engine)】
# ============================================================================
class RewardEngine:
    """
    支持 singmos_pro / mcd / f0 / spk_similarity，按权重混合总奖励并用其计算组内 advantage。
    """
    def __init__(
        self,
        vae=None,
        vae_scale_factor=1.0,
        *,
        enabled_rewards=None,
        reward_weights=None,
        adv_eps: float = 1e-8,
        debug_return_wave: bool = False,
        use_gpu: bool = True,
        cache_dir: str = "versa_cache",
        mcd_f0_cfg=None,
        spk_cfg=None,
        pseudo_mos_cfg=None,
        mcd_exp_k: float = 0.08,
        f0_exp_k: float = 0.01,
    ):
        self.vae = vae
        self.vae_scale_factor = float(vae_scale_factor)
        self.adv_eps = float(adv_eps)
        self.debug_return_wave = bool(debug_return_wave)
        self.use_gpu = bool(use_gpu)
        self.cache_dir = str(cache_dir)
        self.mcd_exp_k = float(mcd_exp_k)
        self.f0_exp_k = float(f0_exp_k)

        if enabled_rewards is None:
            enabled_rewards = ["singmos_pro"]
        self.enabled_rewards = {str(x) for x in enabled_rewards}
        default_weights = {
            "singmos_pro": 1.0,
            "mcd": 0.0,
            "f0": 0.0,
            "spk_similarity": 0.0,
        }
        if reward_weights is not None:
            if OmegaConf.is_config(reward_weights):
                reward_weights = OmegaConf.to_container(reward_weights, resolve=True)
            if isinstance(reward_weights, dict):
                for k, v in reward_weights.items():
                    default_weights[str(k)] = float(v)
        self.reward_weights = default_weights

        self.mcd_f0_cfg = mcd_f0_cfg if isinstance(mcd_f0_cfg, dict) else {}
        self.spk_cfg = spk_cfg if isinstance(spk_cfg, dict) else {}
        self.pseudo_mos_cfg = pseudo_mos_cfg if isinstance(pseudo_mos_cfg, dict) else {}

        if self.vae is not None:
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False

        self._pseudo_mos_metric = None
        self._mcd_f0_metric = None
        self._speaker_metric = None
        self._speaker_model = None
        self._pseudo_predictor_dict = None
        self._pseudo_predictor_fs = None

        # 动态导入，避免影响当前已有训练环境
        if "singmos_pro" in self.enabled_rewards:
            try:
                from modules.versa_reward.pesudo_mos import pseudo_mos_metric, pseudo_mos_setup
                self._pseudo_mos_metric = pseudo_mos_metric
                predictor_types = self.pseudo_mos_cfg.get("predictor_types", ["singmos_pro"])
                predictor_args = self.pseudo_mos_cfg.get("predictor_args", {})
                self._pseudo_predictor_dict, self._pseudo_predictor_fs = pseudo_mos_setup(
                    predictor_types, predictor_args=predictor_args, cache_dir=self.cache_dir, use_gpu=self.use_gpu
                )
            except Exception as e:
                logger.warning("RewardEngine: init singmos_pro failed, fallback to zeros. err=%s", str(e))

        if ("mcd" in self.enabled_rewards) or ("f0" in self.enabled_rewards):
            try:
                from modules.versa_reward.mcd_f0 import mcd_f0
                self._mcd_f0_metric = mcd_f0
            except Exception as e:
                logger.warning("RewardEngine: init mcd_f0 failed, fallback to zeros. err=%s", str(e))

        if "spk_similarity" in self.enabled_rewards:
            try:
                from modules.versa_reward.spk_similarity import speaker_metric, speaker_model_setup
                self._speaker_metric = speaker_metric
                self._speaker_model = speaker_model_setup(
                    model_tag=self.spk_cfg.get("model_tag", "default"),
                    model_path=self.spk_cfg.get("model_path", None),
                    model_config=self.spk_cfg.get("model_config", None),
                    use_gpu=self.use_gpu,
                )
            except Exception as e:
                logger.warning("RewardEngine: init spk_similarity failed, fallback to zeros. err=%s", str(e))

    @torch.no_grad()
    def _decode_wave(self, x_final: torch.Tensor) -> torch.Tensor:
        G = x_final.size(0)
        if self.vae is not None:
            z_raw = x_final / self.vae_scale_factor
            mean, _ = torch.chunk(z_raw, 2, dim=1)
            wave = self.vae.decode(mean).squeeze(1)
        else:
            wave = torch.randn(G, x_final.size(2) * 256, device=x_final.device)
        return wave

    def _to_numpy(self, x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().float().numpy()

    @torch.no_grad()
    def compute_reward_and_advantage(self, prompt_group_data):
        x_final = prompt_group_data["x_final"]  # [G, C, T]
        G = x_final.size(0)
        wave = self._decode_wave(x_final)       # [G, Tw]
        assert wave.dim() == 2 and wave.size(0) == G

        device = wave.device
        vae_sr = int(getattr(self.vae, "sampling_rate", 44100)) if self.vae is not None else 44100
        wave_np = self._to_numpy(wave)

        # 参考波形：ref_wave = 整句 GT latent 解码，供 mcd/f0 与（无 prompt 时）spk 回退
        ref_wave = prompt_group_data.get("ref_wave", None)
        ref_wave_np = None
        if isinstance(ref_wave, torch.Tensor):
            ref_wave_t = ref_wave.to(device).float()
            if ref_wave_t.dim() == 1:
                ref_wave_t = ref_wave_t.unsqueeze(0).repeat_interleave(G, dim=0)
            elif ref_wave_t.dim() == 2 and ref_wave_t.size(0) == 1:
                ref_wave_t = ref_wave_t.repeat_interleave(G, dim=0)
            ref_wave_np = self._to_numpy(ref_wave_t)

        # spk_similarity：优先用外部注入的 spk_ref_wave（通常来自 cond.prompt_latent 解码），与「目标说话人」条件一致
        spk_ref_wave = prompt_group_data.get("spk_ref_wave", None)
        spk_ref_np = None
        if isinstance(spk_ref_wave, torch.Tensor):
            spk_ref_t = spk_ref_wave.to(device).float()
            if spk_ref_t.dim() == 1:
                spk_ref_t = spk_ref_t.unsqueeze(0).repeat_interleave(G, dim=0)
            elif spk_ref_t.dim() == 2 and spk_ref_t.size(0) == 1:
                spk_ref_t = spk_ref_t.repeat_interleave(G, dim=0)
            spk_ref_np = self._to_numpy(spk_ref_t)

        components = {}

        # mcd/f0 是同一次特征提取得到的，避免重复计算
        mcd_f0_out = None
        if (("mcd" in self.enabled_rewards) or ("f0" in self.enabled_rewards)) and (self._mcd_f0_metric is not None) and (ref_wave_np is not None):
            try:
                mcd_f0_out = self._mcd_f0_metric(
                    wave_np,
                    ref_wave_np,
                    fs=vae_sr,
                    f0min=float(self.mcd_f0_cfg.get("f0min", 40)),
                    f0max=float(self.mcd_f0_cfg.get("f0max", 810)),
                    mcep_shift=int(self.mcd_f0_cfg.get("mcep_shift", 5)),
                    mcep_fftl=int(self.mcd_f0_cfg.get("mcep_fftl", 1024)),
                    mcep_dim=int(self.mcd_f0_cfg.get("mcep_dim", 39)),
                    mcep_alpha=float(self.mcd_f0_cfg.get("mcep_alpha", 0.466)),
                    seq_mismatch_tolerance=float(self.mcd_f0_cfg.get("seq_mismatch_tolerance", 0.1)),
                    power_threshold=float(self.mcd_f0_cfg.get("power_threshold", -20)),
                    dtw=bool(self.mcd_f0_cfg.get("dtw", True)),
                    batch_workers=self.mcd_f0_cfg.get("batch_workers", None),
                )
            except Exception as e:
                logger.warning("RewardEngine: mcd_f0 batch compute failed, fallback to 0. err=%s", str(e))
                mcd_f0_out = None

        # === 1) 先拿 raw 指标 ===
        raw_mos = torch.zeros(G, device=device)
        raw_sim = torch.zeros(G, device=device)
        raw_mcd = torch.zeros(G, device=device)
        raw_f0 = torch.zeros(G, device=device)

        if "singmos_pro" in self.enabled_rewards:
            if self._pseudo_mos_metric is not None and self._pseudo_predictor_dict is not None:
                try:
                    out = self._pseudo_mos_metric(
                        wave_np,
                        fs=vae_sr,
                        predictor_dict=self._pseudo_predictor_dict,
                        predictor_fs=self._pseudo_predictor_fs,
                        use_gpu=self.use_gpu,
                    )
                    raw_mos = torch.as_tensor(out["singmos_pro"], dtype=torch.float32, device=device).reshape(G)
                except Exception:
                    raw_mos = torch.zeros(G, device=device)

        if isinstance(mcd_f0_out, dict) and ("mcd" in mcd_f0_out):
            mcd_arr = np.asarray(mcd_f0_out["mcd"])
            if mcd_arr.size == 1:
                mcd_arr = np.repeat(mcd_arr.reshape(1), G, axis=0)
            mcd_arr = np.nan_to_num(mcd_arr, nan=0.0, posinf=0.0, neginf=0.0)
            raw_mcd = torch.as_tensor(mcd_arr, dtype=torch.float32, device=device).reshape(G)

        if isinstance(mcd_f0_out, dict) and ("f0rmse" in mcd_f0_out):
            f0_arr = np.asarray(mcd_f0_out["f0rmse"])
            if f0_arr.size == 1:
                f0_arr = np.repeat(f0_arr.reshape(1), G, axis=0)
            f0_arr = np.nan_to_num(f0_arr, nan=0.0, posinf=0.0, neginf=0.0)
            raw_f0 = torch.as_tensor(f0_arr, dtype=torch.float32, device=device).reshape(G)

        if "spk_similarity" in self.enabled_rewards:
            spk_gt_np = spk_ref_np if spk_ref_np is not None else ref_wave_np
            if self._speaker_metric is not None and self._speaker_model is not None and spk_gt_np is not None:
                try:
                    out = self._speaker_metric(self._speaker_model, wave_np, spk_gt_np, fs=vae_sr)
                    raw_sim = torch.as_tensor(out["spk_similarity"], dtype=torch.float32, device=device).reshape(G)
                except Exception as e:
                    logger.warning("RewardEngine: spk_similarity batch compute failed, fallback to 0. err=%s", str(e))
                    raw_sim = torch.zeros(G, device=device)

        # === 2) 按要求映射到 [0, 1] 且越大越好 ===
        mos_reward = torch.clamp((raw_mos - 1.0) / 4.0, min=0.0, max=1.0)
        sim_reward = torch.clamp(raw_sim, min=0.0, max=1.0)
        mcd_reward = torch.clamp(torch.exp(-self.mcd_exp_k * raw_mcd), min=0.0, max=1.0)
        f0_reward = torch.clamp(torch.exp(-self.f0_exp_k * raw_f0), min=0.0, max=1.0)

        # === 3) 权重融合（从配置 reward_weights 读取）===
        rw = self.reward_weights if isinstance(self.reward_weights, dict) else {}
        w_mos = float(rw.get("mos", rw.get("singmos_pro", 1.0)))
        w_sim = float(rw.get("sim", rw.get("spk_similarity", 0)))
        w_mcd = float(rw.get("mcd", 0))
        w_f0 = float(rw.get("f0", 0))
        total_reward = (
            w_mos * mos_reward
            + w_sim * sim_reward
            + w_mcd * mcd_reward
            + w_f0 * f0_reward
        )

        # === 4) Advantage（GRPO）===
        mean = total_reward.mean()
        std = total_reward.std(unbiased=False)
        safe_std = torch.clamp(std, min=0.02)
        advantage = (total_reward - mean) / safe_std

        # 记录归一化后的子 reward（便于日志直接分析）
        components["singmos_pro"] = mos_reward
        components["spk_similarity"] = sim_reward
        components["mcd"] = mcd_reward
        components["f0"] = f0_reward

        prompt_group_data["reward"] = total_reward
        prompt_group_data["advantage"] = advantage
        prompt_group_data["reward_components"] = {k: v.detach() for k, v in components.items()}
        if self.debug_return_wave:
            prompt_group_data["wave"] = wave
            prompt_group_data["wave_lengths"] = torch.tensor([wave.shape[1]] * G, dtype=torch.long, device=device)
        return prompt_group_data


# ============================================================================
# 【3.5 CPU 卸载与经验回放模块 (CPU Offload Buffer)】
# ============================================================================
class CPUOffloadBuffer:
    def __init__(self):
        self.memory_list = []
        
    def add(self, prompt_group_data):
        """将 GPU 数据执行 .detach().cpu() 后存入"""
        cpu_data = {}
        for k, v in prompt_group_data.items():
            if isinstance(v, torch.Tensor):
                cpu_data[k] = v.detach().cpu()
            elif isinstance(v, dict):
                # 针对 cond 字典
                cpu_data[k] = {kk: vv.detach().cpu() if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
            else:
                cpu_data[k] = v
        self.memory_list.append(cpu_data)
        
    def clear(self):
        self.memory_list.clear()
        
    def __len__(self):
        return len(self.memory_list)
    
    def get_all(self):
        return self.memory_list


# ============================================================================
# 【4. 核心训练域 (Trainer Engine)】
# ============================================================================
class TrainerEngine:
    """
    梯度开启，动态 LoRA 切换
    Actor Model (LoRA ON), Reference Model (LoRA OFF)
    纯净数据输入（无 padding），执行维度塌缩计算 Loss。
    """
    def __init__(
        self,
        policy_model,
        a_sde=0.7,
        kl_coef=0.01,
        clip_range=0.2,
        score_denom_eps=1e-4,
        lora_manager=None,
    ):
        self.policy_model = policy_model
        self.a_sde = float(a_sde) # 严格对齐 Rollout 的 Sigma
        self.kl_coef = float(kl_coef)
        self.clip_range = float(clip_range)
        self.score_denom_eps = float(score_denom_eps)
        self.lora_manager = lora_manager
        
    def optimize_policy(self, group_data):
        """
        按照数学公式，严格翻译计算 Surrogate Loss 和 KL Loss。
         group_data 已解 Batch（无 padding）。
        """
        device = next(self.policy_model.parameters()).device
        
        # 将数据移回 GPU
        t_cur = group_data['t'].to(device)
        dt = group_data['dt'].to(device)
        x_t = group_data['x_t'].to(device)
        noise = group_data['noise'].to(device)
        v_old = group_data['v_old'].to(device)
        v_ref = group_data['v_ref'].to(device)
        advantage = group_data['advantage'].to(device)
        
        # Step 1: Constants (与 RolloutEngine/Sampler 的 one_minus_t clamp 对齐)
        one_minus_t = torch.clamp(1.0 - t_cur, min=self.score_denom_eps)
        # t_cur 理论上来自噪声窗口 (t>0)，为避免数值除零只用极小兜底
        t_safe = torch.clamp(t_cur, min=1e-12)
        sigma_t = self.a_sde * torch.sqrt((1.0 - t_cur) / t_safe)
        w_score = t_cur / one_minus_t
        
        # Step 2: Forward
        # 由于 group_data['cond'] 形状与 x_t 的 batch 维一致，这里直接调用 apply_model。
        cond_for_apply = {}
        for k, v in group_data["cond"].items():
            if isinstance(v, torch.Tensor):
                cond_for_apply[k] = v.to(device)
            else:
                cond_for_apply[k] = v
        # apply_model 内部会把 t 传给 DiffusionWrapper，需要 long 时间索引；与 CFMSampler.Wrapper 一致使用 t*1000。
        t_idx_val = int(float(t_cur.detach().cpu().item()) * 1000.0)
        t_idx = torch.full((x_t.size(0),), t_idx_val, dtype=torch.long, device=device)
        
        if self.lora_manager is not None:
            self.lora_manager.enable()
        v_theta, _, _, _ = self.policy_model.apply_model(x_t, t_idx, cond_for_apply)
            
        mu_theta = v_theta + 0.5 * (sigma_t ** 2) * (w_score * v_theta - x_t / one_minus_t)
        mu_old = v_old + 0.5 * (sigma_t ** 2) * (w_score * v_old - x_t / one_minus_t)
        
        # Step 3: Log Ratio
        delta_mu = mu_old - mu_theta
        diff = delta_mu * dt
        
        # 降维塌缩 sum(dim=[1,2])
        num = diff ** 2 + 2 * diff * sigma_t * torch.sqrt(torch.clamp(dt, min=0.0)) * noise
        log_r = -num.sum(dim=[1, 2]) / (2 * (sigma_t ** 2) * dt)
        r_t = torch.exp(log_r)
        
        # Step 4: Surrogate Loss
        L_surr1 = r_t * advantage
        L_surr2 = torch.clamp(r_t, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantage
        L_pg = -torch.min(L_surr1, L_surr2).mean()
        
        # Step 5: KL Loss
        W_KL = (dt / 2) * ((self.a_sde / 2 + 1 / self.a_sde) ** 2) * w_score
        L_KL = (W_KL * ((v_theta - v_ref) ** 2).sum(dim=[1, 2])).mean()
        
        # Step 6: Total
        loss = L_pg + self.kl_coef * L_KL
        
        metrics = {
            "loss/L_pg": float(L_pg.item()),
            "loss/L_KL": float(L_KL.item()),
            "loss/total": float(loss.item()),
            "metrics/v_theta_v_ref_mse": float(((v_theta - v_ref) ** 2).mean().item()),
        }
        return loss, metrics


# ============================================================================
# 【主循环管线】
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="模型 config")
    parser.add_argument("--ckpt", type=str, required=True, help="模型权重路径")
    parser.add_argument("--manifest_path", type=str, default="data/final_test/test.tsv")
    parser.add_argument("--max_eval", type=int, default=1, help="最多评测条数，0 表示全部")
    parser.add_argument("--max_duration", type=float, default=20.0, help="过滤超过该时长的样本")
    # 训练参数默认从 YAML(grpo.grpo_train) 读取；CLI 仅作为可选覆盖
    parser.add_argument("--batch_size", type=int, default=None, help="训练 DataLoader batch size (override YAML)")
    parser.add_argument("--epochs", type=int, default=None, help="训练总 Epoch 数 (override YAML)")
    parser.add_argument("--lr", type=float, default=None, help="Actor 学习率 (override YAML)")
    parser.add_argument("--save_every_steps", type=int, default=None, help="每过 N 步保存一次 Checkpoint (override YAML)")
    parser.add_argument("--output_dir", type=str, default=None, help="Checkpoint 输出目录 (override YAML)")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb 项目名称 (override YAML)")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb 运行名称 (override YAML)")
    parser.add_argument("--skip_train", action="store_true", help="仅执行 rollout/reward，不执行训练、wandb 与 checkpoint")
    parser.add_argument("--resume_run_dir", type=str, default=None, help="恢复训练时指定的运行目录（包含 training_state_latest.pt）")
    args, _ = parser.parse_known_args()
    return args

def cond_to_device(cond, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in cond.items()}


def main():
    # 1. 解析参数 & 准备设备
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 加载模型 (Policy & Ref)
    config = OmegaConf.load(args.config)
    # 支持在 GRPO 配置里通过 base_model_config 复用基础模型配置。
    base_model_config = config.get("base_model_config", None) if hasattr(config, "get") else None
    if base_model_config:
        base_cfg = OmegaConf.load(base_model_config)
        config = OmegaConf.merge(base_cfg, config)
    policy_model = instantiate_from_config(config.model)
    
    if os.path.exists(args.ckpt):
        policy_ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        policy_model.load_state_dict(policy_ckpt["state_dict"] if "state_dict" in policy_ckpt else policy_ckpt, strict=False)
    
    policy_model.to(device)
    
    # 直接复用 VAE（first_stage_model 里包含 decode），只冻结参数即可
    vae = policy_model.first_stage_model
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
        
    # 3. LoRA 设置
    from peft import get_peft_model, LoraConfig
    lora_config = LoraConfig(
        r=16,                  # 秩数，16 能够在保持表达能力的同时极大地节省显存
        lora_alpha=32,         # 缩放系数
        target_modules=[
            # Attention 模块
            "qkv_audio", 
            "qkv_cond", 
            "proj_audio", 
            "proj_cond",
            # ConvMLP 模块 (PEFT 原生支持 nn.Conv1d)
            "conv1", 
            "conv2"
        ],
        lora_dropout=0.05,
        bias="none",
    )
    # 我们只对主干模型(Backbone/MMDiT) 应用 LoRA，避免污染 Frontend 结构
    # 将 Backbone 转为带有 Lora 的模型
    policy_model.model = get_peft_model(policy_model.model, lora_config)
    policy_model.model.print_trainable_parameters()
    # 默认保持 eval；仅在真正执行一次参数更新时短暂切到 train
    policy_model.model.eval()
    
    # 定义 LoraManager 以便开关
    class LoraManager:
        def __init__(self, lora_model):
            self.model = lora_model
            
        def enable(self):
            self.model.enable_adapter_layers()
            
        def disable(self):
            self.model.disable_adapter_layers()
            
    lora_manager = LoraManager(policy_model.model)
    
    # 3. 实例化四大引擎
    grpo_cfg = config.get("grpo", {}) if hasattr(config, "get") else {}
    frontend_cfg = grpo_cfg.get("frontend", {}) if hasattr(grpo_cfg, "get") else {}
    rollout_cfg = grpo_cfg.get("rollout", {}) if hasattr(grpo_cfg, "get") else {}
    trainer_cfg = grpo_cfg.get("trainer", {}) if hasattr(grpo_cfg, "get") else {}
    reward_cfg = grpo_cfg.get("reward", {}) if hasattr(grpo_cfg, "get") else {}
    grpo_train_cfg = grpo_cfg.get("grpo_train", {}) if hasattr(grpo_cfg, "get") else {}

    debug_return_wave = bool(reward_cfg.get("debug_return_wave", False))
    debug_save_dir = reward_cfg.get("debug_save_dir", "debug_outputs/grpo_audio")

    train_batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(grpo_train_cfg.get("data_batch_size", grpo_train_cfg.get("batch_size", 1)))
    )
    train_epochs = int(args.epochs) if args.epochs is not None else int(grpo_train_cfg.get("epochs", 10))
    train_lr = float(args.lr) if args.lr is not None else float(grpo_train_cfg.get("lr", 1e-5))
    accum_steps = int(grpo_train_cfg.get("grad_accum_steps", grpo_train_cfg.get("accum_steps", 1)))
    accum_steps = max(1, accum_steps)
    flush_accum_on_epoch_end = bool(grpo_train_cfg.get("flush_accum_on_epoch_end", True))
    save_every_steps = int(args.save_every_steps) if args.save_every_steps is not None else int(grpo_train_cfg.get("save_every_steps", 50))
    output_dir = str(args.output_dir) if args.output_dir is not None else str(grpo_train_cfg.get("output_dir", "exp_outputs/grpo_checkpoints"))
    skip_train = bool(args.skip_train) or bool(grpo_train_cfg.get("skip_train", False))
    resume_run_dir = args.resume_run_dir if args.resume_run_dir is not None else grpo_train_cfg.get("resume_run_dir", None)

    wandb_project = args.wandb_project if args.wandb_project is not None else grpo_train_cfg.get("wandb_project", "DiffSVS-GRPO")
    wandb_run_name = args.wandb_run_name if args.wandb_run_name is not None else grpo_train_cfg.get("wandb_run_name", "grpo_test_run")
    log_reward_advantage = bool(grpo_train_cfg.get("log_reward_advantage", True))
    grpo_logdir_cfg = grpo_train_cfg.get("logdir", None)

    frozen_frontend = FrozenFrontend(
        policy_model,
        duration_min=frontend_cfg.get("duration_min", 1),
    )
    rollout_engine = RolloutEngine(
        policy_model,
        G=rollout_cfg.get("G", 4),
        timesteps=rollout_cfg.get("timesteps", 25),
        noise_start_t=rollout_cfg.get("noise_start_t", 0.05),
        noise_stop_t=rollout_cfg.get("noise_stop_t", 0.95),
        sigma=rollout_cfg.get("sigma", 0.7),
        sigma_schedule=rollout_cfg.get("sigma_schedule", "flow_grpo"),
        sampler_type=rollout_cfg.get("sampler_type", "sde"),
        lora_manager=lora_manager,
        debug_sde_diff=rollout_cfg.get("debug_sde_diff", False),
        debug_sde_dir=rollout_cfg.get("debug_sde_dir", "debug_outputs/grpo_sde_vis"),
        debug_sde_max_groups=rollout_cfg.get("debug_sde_max_groups", 4),
    )
    reward_yaml_path = reward_cfg.get("reward_config_path", None)
    reward_yaml_cfg = load_reward_yaml_config(str(reward_yaml_path)) if reward_yaml_path else {}
    enabled_rewards = reward_cfg.get("enabled_rewards", None)
    reward_weights = reward_cfg.get("reward_weights", None)
    mcd_f0_cfg = reward_cfg.get("mcd_f0_cfg", None) or reward_yaml_cfg.get("mcd_f0", None)
    spk_cfg = reward_cfg.get("spk_cfg", None) or reward_yaml_cfg.get("speaker", None)
    pseudo_cfg = reward_yaml_cfg.get("pseudo_mos", None) if isinstance(reward_yaml_cfg, dict) else None
    pseudo_mos_cfg = reward_cfg.get("pseudo_mos_cfg", None) or pseudo_cfg
    cache_dir = reward_cfg.get("cache_dir", "/root/.cache/torch/hub")

    reward_engine = RewardEngine(
        vae=vae,
        vae_scale_factor=policy_model.scale_factor,
        enabled_rewards=enabled_rewards,
        reward_weights=reward_weights,
        adv_eps=reward_cfg.get("adv_eps", 1e-8),
        debug_return_wave=reward_cfg.get("debug_return_wave", False),
        use_gpu=reward_cfg.get("use_gpu", True),
        cache_dir=cache_dir if cache_dir is not None else "versa_cache",
        mcd_f0_cfg=mcd_f0_cfg,
        spk_cfg=spk_cfg,
        pseudo_mos_cfg=pseudo_mos_cfg,
        mcd_exp_k=reward_cfg.get("mcd_exp_k", 0.08),
        f0_exp_k=reward_cfg.get("f0_exp_k", 0.01),
    )
    logger.info("Using RewardEngine (weighted multi-reward), reward_yaml=%s", str(reward_yaml_path))
    buffer = None
    optimizer = None
    trainer_engine = None
    rl_update_batch_size = int(
        grpo_train_cfg.get(
            "rl_update_group_batch_size",
            trainer_cfg.get("rl_update_batch_size", 1),
        )
    )
    rl_update_batch_size = max(1, rl_update_batch_size)
    if not skip_train:
        buffer = CPUOffloadBuffer()
        # 确保创建 optimizer 时 LoRA adapter 真正处于可训练状态（否则 trainable_params 可能为空）
        lora_manager.enable()
        # 获取需要更新的参数：MMDiT 中的 LoRA 参数
        trainable_params = [p for p in policy_model.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=train_lr)
        trainer_engine = TrainerEngine(
            policy_model=policy_model,
            a_sde=trainer_cfg.get("a_sde", rollout_cfg.get("sigma", 0.7)),  # TrainerEngine a_sde 与 RolloutEngine sigma 严格对齐
            kl_coef=trainer_cfg.get("kl_coef", 0.01),
            clip_range=trainer_cfg.get("clip_range", 0.2),
            score_denom_eps=trainer_cfg.get("score_denom_eps", rollout_cfg.get("score_denom_eps", 1.0e-4)),
            lora_manager=lora_manager,
        )
        logger.info("RL update threshold (buffer size): %d", rl_update_batch_size)
        logger.info("Gradient accumulation steps: %d", accum_steps)
    else:
        logger.info("skip_train=True: training/wandb/checkpoint are disabled.")

    # 运行目录管理：
    # - 默认每次新建 run_dir
    # - 指定 resume_run_dir 时复用该目录并尝试恢复训练状态
    run_dir = None
    checkpoints_dir = None
    latest_state_path = None
    start_epoch = 0
    effective_run_name = wandb_run_name
    if not skip_train:
        os.makedirs(output_dir, exist_ok=True)
        if resume_run_dir:
            run_dir = str(resume_run_dir)
            os.makedirs(run_dir, exist_ok=True)
            logger.info("Resume mode: using existing run_dir=%s", run_dir)
        else:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            config_name = os.path.basename(str(args.config)) if args.config else "grpo_config"
            config_stem = os.path.splitext(config_name)[0]
            run_name_safe = str(config_stem).replace(" ", "_")
            run_dir = os.path.join(output_dir, f"{ts}_{run_name_safe}")
            os.makedirs(run_dir, exist_ok=True)
            logger.info("New run dir created: %s", run_dir)
        effective_run_name = os.path.basename(os.path.normpath(run_dir))
        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        latest_state_path = os.path.join(run_dir, "training_state_latest.pt")

        if resume_run_dir and os.path.exists(latest_state_path):
            state = torch.load(latest_state_path, map_location="cpu")
            model_sd = state.get("model_state_dict", None)
            opt_sd = state.get("optimizer_state_dict", None)
            if model_sd is not None:
                policy_model.model.load_state_dict(model_sd, strict=False)
            if opt_sd is not None:
                optimizer.load_state_dict(opt_sd)
            start_epoch = int(state.get("start_epoch", 0))
            logger.info(
                "Resumed state loaded: start_epoch=%d, global_step=%d, accum_counter=%d",
                start_epoch,
                int(state.get("global_step", 0)),
                int(state.get("accum_counter", 0)),
            )
        elif resume_run_dir:
            logger.warning("resume_run_dir is set but no state file found: %s", latest_state_path)

        def save_training_state(*, start_epoch_to_save: int, global_step_to_save: int, accum_counter_to_save: int):
            state_obj = {
                "start_epoch": int(start_epoch_to_save),
                "global_step": int(global_step_to_save),
                "accum_counter": int(accum_counter_to_save),
                "model_state_dict": policy_model.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(state_obj, latest_state_path)

    reward_adv_logpath = None
    if log_reward_advantage:
        if run_dir is not None:
            if grpo_logdir_cfg is None or str(grpo_logdir_cfg).strip() == "":
                reward_adv_logdir = os.path.join(run_dir, "logdir")
            else:
                ld = str(grpo_logdir_cfg)
                reward_adv_logdir = ld if os.path.isabs(ld) else os.path.join(run_dir, ld)
        else:
            os.makedirs(output_dir, exist_ok=True)
            ts_skip = datetime.now().strftime("%Y%m%d-%H%M%S")
            skip_base = os.path.join(output_dir, f"skip_train_{ts_skip}")
            os.makedirs(skip_base, exist_ok=True)
            if grpo_logdir_cfg is None or str(grpo_logdir_cfg).strip() == "":
                reward_adv_logdir = os.path.join(skip_base, "logdir")
            else:
                ld = str(grpo_logdir_cfg)
                reward_adv_logdir = ld if os.path.isabs(ld) else os.path.join(skip_base, ld)
        os.makedirs(reward_adv_logdir, exist_ok=True)
        reward_adv_logpath = os.path.join(reward_adv_logdir, "reward_advantage.log")
        logger.info("Reward/advantage batch log: %s", reward_adv_logpath)

    # 4. 初始化 WandB
    print(f"\n--- 初始化 WandB (Project: {wandb_project}, Name: {effective_run_name}) ---")
    if (not skip_train) and wandb_project:
        # 将 wandb 本地文件放到当前 run_dir，便于与 ckpt 一一对应
        os.environ["WANDB_DIR"] = run_dir
        wandb.init(
            project=wandb_project,
            name=effective_run_name,
            dir=run_dir,
            config=OmegaConf.to_container(config, resolve=True) if hasattr(config, "get") else {}
        )
    
    # 5. 模拟 DataLoader
    dataset = DiffSVSEvalDataset(
        args.manifest_path,
        max_eval=args.max_eval,
        max_duration=args.max_duration,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True, # 训练时打乱
        collate_fn=eval_collate_fn,
    )
    
    # 6. 训练大循环
    logger.info("Starting training loop (Framework Setup Complete)")
    global_step = 0
    accum_counter = 0
    if not skip_train:
        if resume_run_dir and latest_state_path and os.path.exists(latest_state_path):
            state = torch.load(latest_state_path, map_location="cpu")
            global_step = int(state.get("global_step", 0))
            accum_counter = int(state.get("accum_counter", 0))
        optimizer.zero_grad()
    
    for epoch in range(start_epoch, train_epochs):
        logger.info(f"========== 开始 Epoch {epoch + 1}/{train_epochs} ==========")
        for step, (cond, names, _, latent_paths, _, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
            cond = cond_to_device(cond, device)

            # --- 阶段 1: 收集 (Rollout & Reward) ---
            # rollout/reward 阶段默认使用 eval 模式；
            # 训练更新后已立刻切回 eval，这里无需每个 iter 重复设置。
            # (1) 冻结前端特征提取
            frontend_out = frozen_frontend.process(cond)
            
            # (2) 经验采样
            rollout_prompts_data = rollout_engine.rollout(frontend_out)
            
            # (3) 奖励与优势计算
            ra_log_records = []
            for prompt_group_idx, group_data in enumerate(rollout_prompts_data):
                # GT 整句参考（latent_paths）：供 mcd/f0 与 spk 的回退参考
                gt_ref_wave = None
                try:
                    if isinstance(latent_paths, (list, tuple)) and prompt_group_idx < len(latent_paths):
                        gt_latent_path = latent_paths[prompt_group_idx]
                    else:
                        gt_latent_path = ""
                    if isinstance(gt_latent_path, str) and gt_latent_path and os.path.isfile(gt_latent_path):
                        gt_latent_np = np.load(gt_latent_path).astype(np.float32)  # [128, T]
                        gt_ref_wave = decode_latent_to_wave(vae=vae, latent=gt_latent_np, device=device)
                        group_data["ref_wave"] = gt_ref_wave
                except Exception as e:
                    logger.warning("RewardEngine: GT latent decode failed, fallback to 0. err=%s", str(e))
                    pass
                # spk_similarity：与 prompt 参考说话人一致，使用 cond.prompt_latent 解码为 wav（而非整句 GT）
                try:
                    cond_for_spk = group_data.get("cond", None)
                    p_lat = cond_for_spk.get("prompt_latent", None) if isinstance(cond_for_spk, dict) else None
                    if isinstance(p_lat, torch.Tensor) and p_lat.numel() > 0 and vae is not None:
                        group_data["spk_ref_wave"] = decode_latent_to_wave(vae=vae, latent=p_lat, device=device)
                except Exception as e:
                    logger.warning("RewardEngine: prompt_latent decode for spk_ref_wave failed. err=%s", str(e))
                group_data_with_adv = reward_engine.compute_reward_and_advantage(group_data)
                if reward_adv_logpath is not None:
                    r_cpu = group_data_with_adv["reward"].detach().cpu()
                    a_cpu = group_data_with_adv["advantage"].detach().cpu()
                    comp = group_data_with_adv.get("reward_components", {})
                    if isinstance(names, (list, tuple)) and prompt_group_idx < len(names):
                        item_name = str(names[prompt_group_idx])
                    else:
                        item_name = str(names)
                    rec = {
                        "epoch": epoch + 1,
                        "dataloader_step": step,
                        "global_step": global_step,
                        "prompt_group_idx": prompt_group_idx,
                        "item_name": item_name,
                        "reward": r_cpu.tolist(),
                        "advantage": a_cpu.tolist(),
                        "reward_mean": float(r_cpu.mean().item()),
                        "advantage_mean": float(a_cpu.mean().item()),
                        "reward_std": float(r_cpu.std(unbiased=False).item()),
                        "advantage_std": float(a_cpu.std(unbiased=False).item()),
                    }
                    # 记录各子 reward（如果存在）
                    if isinstance(comp, dict):
                        for k_sub, v_sub in comp.items():
                            if isinstance(v_sub, torch.Tensor):
                                v_cpu = v_sub.detach().cpu()
                                rec[f"reward#_{k_sub}"] = v_cpu.tolist()
                    ra_log_records.append(rec)
                
                if debug_return_wave:
                    if isinstance(names, (list, tuple)):
                        base_name = str(names[prompt_group_idx])
                    else:
                        base_name = str(names)
                    # 保存 prompt 参考音频（来自 cond.prompt_latent），每个 utterance 一条
                    try:
                        cond_for_prompt = group_data.get("cond", None)
                        if isinstance(cond_for_prompt, dict):
                            p_lat = cond_for_prompt.get("prompt_latent", None)
                        else:
                            p_lat = None
                        if isinstance(p_lat, torch.Tensor) and (vae is not None):
                            p_wave = decode_latent_to_wave(vae=vae, latent=p_lat, device=device)
                            os.makedirs(debug_save_dir, exist_ok=True)
                            vae_sr = int(getattr(vae, "sampling_rate", 44100))
                            p_wav = p_wave.detach().cpu().squeeze(0).numpy()
                            p_wav = normalize_loudness(p_wav, -23.0)
                            sf.write(
                                os.path.join(debug_save_dir, f"{base_name}_prompt.wav"),
                                p_wav,
                                vae_sr,
                                subtype="PCM_16",
                            )
                    except Exception:
                        pass
                    # 额外保存 GT latent 重建音频（每个 utterance 一条）
                    if isinstance(gt_ref_wave, torch.Tensor) and gt_ref_wave.numel() > 0:
                        try:
                            os.makedirs(debug_save_dir, exist_ok=True)
                            vae_sr = int(getattr(vae, "sampling_rate", 44100)) if vae is not None else 44100
                            gt_wav = gt_ref_wave.detach().cpu().squeeze(0).numpy()
                            gt_wav = normalize_loudness(gt_wav, -23.0)
                            sf.write(
                                os.path.join(debug_save_dir, f"{base_name}_gt_recon.wav"),
                                gt_wav,
                                vae_sr,
                                subtype="PCM_16",
                            )
                        except Exception:
                            pass
                    save_debug_waves(
                        debug_save_dir=debug_save_dir,
                        base_name=base_name,
                        group_data=group_data_with_adv,
                    )
                    # 避免把波形长期塞进 CPU buffer（debug 用完就删）
                    group_data_with_adv.pop("wave", None)
                    group_data_with_adv.pop("wave_lengths", None)

                # 放入 CPU 卸载区
                if not skip_train:
                    buffer.add(group_data_with_adv)

            if reward_adv_logpath is not None and ra_log_records:
                with open(reward_adv_logpath, "a", encoding="utf-8") as _raf:
                    for _rec in ra_log_records:
                        _raf.write(json.dumps(_rec, ensure_ascii=False) + "\n")

            # --- 阶段 2: 优化 (Optimize) ---
            if skip_train:
                continue
            # 仅当 buffer 累计到指定阈值时，才执行一次 RL 更新。
            if len(buffer) >= rl_update_batch_size:
                # 只训练主干 LoRA 模块；frontend/vae 始终保持 eval，避免模式污染
                policy_model.model.train()
                experiences = buffer.get_all()
                
                batch_metrics = {
                    "loss/L_pg": 0.0,
                    "loss/L_KL": 0.0,
                    "loss/total": 0.0,
                    "reward/mean": 0.0,
                    "reward/advantage_mean": 0.0,
                    "metrics/v_theta_v_ref_mse": 0.0,
                }
                reward_comp_sums = {}

                for exp in experiences:
                    loss, metrics = trainer_engine.optimize_policy(exp)
                    # 先对当前 buffer 内经验求均值，再按 accum_steps 缩放做跨步累积
                    (loss / (len(experiences) * accum_steps)).backward()
                    
                    # 累加 metrics 用于日志记录
                    batch_metrics["loss/L_pg"] += metrics["loss/L_pg"]
                    batch_metrics["loss/L_KL"] += metrics["loss/L_KL"]
                    batch_metrics["loss/total"] += metrics["loss/total"]
                    batch_metrics["reward/mean"] += exp["reward"].mean().item()
                    batch_metrics["reward/advantage_mean"] += exp["advantage"].mean().item()
                    batch_metrics["metrics/v_theta_v_ref_mse"] += metrics["metrics/v_theta_v_ref_mse"]
                    comp = exp.get("reward_components", None)
                    if isinstance(comp, dict):
                        for k_sub, v_sub in comp.items():
                            if isinstance(v_sub, torch.Tensor):
                                reward_comp_sums[k_sub] = reward_comp_sums.get(k_sub, 0.0) + float(
                                    v_sub.float().mean().item()
                                )

                # 累计一个“buffer更新窗口”
                accum_counter += 1
                stepped = False
                if accum_counter % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    # 更新完成后立刻切回 eval，避免影响后续 rollout/reward
                    policy_model.model.eval()
                    # 只有真正完成一次参数更新，才计一次训练 step
                    global_step += 1
                    stepped = True
                else:
                    # 未达到 accum_steps 时，也切回 eval 以保障后续 rollout 一致性
                    policy_model.model.eval()
                
                # 计算平均 metrics
                ne = len(experiences)
                avg_metrics = {k: v / ne for k, v in batch_metrics.items()}
                avg_metrics["train/epoch"] = epoch + 1
                reward_comp_avg = {k: s / ne for k, s in reward_comp_sums.items()}

                # 记录到 WandB
                if wandb_project and stepped:
                    wb = {
                        "train/epoch": avg_metrics["train/epoch"],
                        "loss/total": avg_metrics["loss/total"],
                        "loss/L_pg": avg_metrics["loss/L_pg"],
                        "loss/L_KL": avg_metrics["loss/L_KL"],
                        "reward/mean": avg_metrics["reward/mean"],
                        "reward/advantage_mean": avg_metrics["reward/advantage_mean"],
                        "metrics/v_theta_v_ref_mse": avg_metrics["metrics/v_theta_v_ref_mse"],
                    }
                    for k_sub, v_sub in reward_comp_avg.items():
                        wb[f"reward/{k_sub}_mean"] = v_sub
                    wandb.log(wb, step=global_step)
                logger.info(
                    "Step %d | Loss: %.6e | KL: %.6e | PG: %.6e | Reward: %.4f | v_theta_v_ref_mse=%.3e | accum %d/%d | stepped=%s",
                    global_step,
                    avg_metrics["loss/total"],
                    avg_metrics["loss/L_KL"],
                    avg_metrics["loss/L_pg"],
                    avg_metrics["reward/mean"],
                    avg_metrics["metrics/v_theta_v_ref_mse"],
                    accum_counter % accum_steps,
                    accum_steps,
                    str(stepped),
                )
                
                # --- 阶段 3: 打扫 ---
                buffer.clear()
                
                # 定期保存 Checkpoint (只保存 LoRA 权重)
                if stepped and global_step % save_every_steps == 0:
                    save_path = os.path.join(checkpoints_dir, f"grpo_step_{global_step}")
                    logger.info(f"Saving Checkpoint to {save_path}")
                    policy_model.model.save_pretrained(save_path)
                    save_training_state(
                        start_epoch_to_save=epoch,
                        global_step_to_save=global_step,
                        accum_counter_to_save=accum_counter,
                    )
            else:
                logger.info(
                    "Buffer not ready: %d/%d, skip RL update this iter.",
                    len(buffer),
                    rl_update_batch_size,
                )
        
        # epoch 末尾可选 flush：把不足 accum_steps 的残余梯度也完成一次更新
        if (not skip_train) and flush_accum_on_epoch_end and (accum_counter % accum_steps != 0):
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            policy_model.model.eval()
            global_step += 1
            if wandb_project:
                wandb.log(
                    {
                        "train/epoch": epoch + 1,
                    },
                    step=global_step,
                )
            logger.info(
                "Epoch-end flush update executed. Step %d (accum remainder consumed).",
                global_step,
            )
            save_training_state(
                start_epoch_to_save=epoch,
                global_step_to_save=global_step,
                accum_counter_to_save=accum_counter,
            )

        # Epoch 结束也保存一次
        if not skip_train:
            save_path = os.path.join(checkpoints_dir, f"grpo_epoch_{epoch+1}")
            logger.info(f"Saving Epoch Checkpoint to {save_path}")
            policy_model.model.save_pretrained(save_path)
            save_training_state(
                start_epoch_to_save=epoch + 1,
                global_step_to_save=global_step,
                accum_counter_to_save=accum_counter,
            )
        
    # 训练结束后：将 LoRA 与 policy_model 合并，并保存完整 ckpt（含 frontend + backbone）。
    # 目标：输出可直接给 infer.py --ckpt 使用的文件（torch.load 后包含 "state_dict"）。
    if not skip_train and checkpoints_dir is not None:
        base_ckpt_name = os.path.basename(str(args.ckpt))
        base_ckpt_stem = os.path.splitext(base_ckpt_name)[0]
        config_name = os.path.basename(str(args.config))
        config_stem = os.path.splitext(config_name)[0]
        lora_rank = getattr(lora_config, "r", "na")
        try:
            final_epoch = int(epoch) + 1
        except Exception:
            final_epoch = int(train_epochs)

        merged_ckpt_name = (
            f"merged_full_policy_base-{base_ckpt_stem}_cfg-{config_stem}"
            f"_ep{final_epoch:03d}_step{int(global_step):06d}_rank{lora_rank}.pt"
        )
        merged_ckpt_path = os.path.join(checkpoints_dir, merged_ckpt_name)
        try:
            if hasattr(policy_model.model, "merge_and_unload"):
                # 先把 LoRA 合并回 policy_model.model（backbone）
                policy_model.model = policy_model.model.merge_and_unload()
                # 再保存完整 policy_model（包含 frontend 等其余模块）
                torch.save({"state_dict": policy_model.state_dict()}, merged_ckpt_path)
                logger.info("Merged full policy ckpt saved to: %s", merged_ckpt_path)
            else:
                logger.warning("merge_and_unload not found on model; skip merged ckpt: %s", merged_ckpt_path)
        except Exception as e:
            logger.warning("Failed to merge LoRA into full policy and save ckpt: %s", str(e))
        
    if (not skip_train) and wandb_project:
        wandb.finish()

if __name__ == "__main__":
    main()
