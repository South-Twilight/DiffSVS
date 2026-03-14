import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from ldm.models.diffusion.cfm1_audio import CFM
from ldm.util import instantiate_from_config
from ldm.dataset.diffsvs_dataset import PHN_PAD_ID

class LengthRegulator(nn.Module):
    """根据 duration 将 Token 级别的特征扩展到 Frame 级别"""
    def forward(self, x, dur):
        out = []
        for i in range(x.size(0)):
            expanded = torch.repeat_interleave(x[i], dur[i], dim=0)
            out.append(expanded)
        return torch.nn.utils.rnn.pad_sequence(out, batch_first=True)

class BlurredBoundaryAdaptor(nn.Module):
    """适配 f=2048 极端压缩的 BBC 边界平滑器"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.blur_conv = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=3, padding=1, groups=hidden_dim
        )
        self.act = nn.SiLU()

    def forward(self, c_text, dur, is_training=False):
        if is_training:
            B, T_lat, D = c_text.shape
            boundaries = torch.cumsum(dur, dim=1)
            mask = torch.ones((B, T_lat, 1), device=c_text.device)
            for b in range(B):
                for bndry in boundaries[b]:
                    if 0 <= bndry < T_lat and torch.rand(1).item() < 0.8:
                        mask[b, bndry, :] = 0
            c_text = c_text * mask
        x = c_text.transpose(1, 2)
        x = self.blur_conv(x)
        x = self.act(x)
        return c_text + x.transpose(1, 2)


class DiffSVS_System(CFM):
    """
    MMDiT 歌声合成：audio 模态（latent）+ text 模态（music score），self-attention 做模态交互，单模态 audio 输出。
    - 输入：batch['audio'] = audio latent，batch['cond'] = 乐谱（phn, note dur, note pitch）+ dur_gt/spk_id/f0_gt
    - 流程：cond 经 frontend → LR → BBC 得到 frame 级 text 特征，与 audio latent 在 backbone 内做双流 self-attention
    - 输出：仅预测 audio 流的速度场（flow），用于 CFM 去噪得到音频 latent
    """
    def __init__(self, use_bbc=False, **kwargs):
        frontend_config = kwargs.pop("frontend_config", None)
        loss_weights = kwargs.pop("loss_weights", {})
        # 各损失项权重：默认都为 1.0，可在 YAML 中通过 model.params.loss_weights 配置
        self.loss_w_cfm = float(loss_weights.get("cfm", 1.0))
        if self.loss_w_cfm > 0.01:
            self.loss_w_cfm *= 0.9999
        self.loss_w_f0 = float(loss_weights.get("f0_uv", 1.0))
        self.loss_w_uv = float(loss_weights.get("f0_uv", 1.0))  # 与 f0 共用权重
        self.loss_w_dur = float(loss_weights.get("dur", 1.0))
        super().__init__(**kwargs)
        # 此处 self.model 仍为 DiffusionWrapper(diffusion_model=Backbone, conditioning_key)，无需替换

        if frontend_config is None:
            raise ValueError("DiffSVS_System 需要 frontend_config")

        self.frontend = instantiate_from_config(frontend_config)
        hidden_dim = frontend_config.params.get("hidden_channels", 768)
        self.length_regulator = LengthRegulator()
        self.bbc_adaptor = BlurredBoundaryAdaptor(hidden_dim=hidden_dim)
        self.use_bbc = use_bbc

        self.frontend.train()
        for p in self.frontend.parameters():
            p.requires_grad = True

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """确保整个 batch（含 audio、cond 字典）都搬到当前 device，否则 PL 对 dict batch 不会自动迁移。"""
        if batch is None:
            return batch
        if isinstance(batch, dict):
            out = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    out[k] = v.to(device)
                elif isinstance(v, dict):
                    out[k] = {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
                else:
                    out[k] = v
            return out
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def configure_optimizers(self):
        """把 frontend（含 duration 预测器）参数加入优化器，否则 duration loss 不会更新 frontend。"""
        lr = self.learning_rate
        # 必须包含 model 与 frontend，这样 loss_dur 的反向才会更新 frontend
        params = list(self.model.parameters()) + list(self.frontend.parameters())
        if getattr(self, "cond_stage_trainable", False) and hasattr(self, "cond_stage_model"):
            params += list(self.cond_stage_model.parameters())
        if getattr(self, "learn_logvar", False):
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if getattr(self, "use_scheduler", False) and hasattr(self, "scheduler_config"):
            from ldm.util import instantiate_from_config
            from torch.optim.lr_scheduler import LambdaLR
            scheduler_cfg = instantiate_from_config(self.scheduler_config)
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler_cfg.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    def get_learned_conditioning(self, c):
        """乐谱（text 模态）直接透传，在 apply_model 里经 frontend 编码为 frame 级特征。"""
        if isinstance(c, dict) and "phn" in c:
            return c
        return super().get_learned_conditioning(c) if hasattr(super(), "get_learned_conditioning") else c

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        """原版接口：x = batch['audio']（audio 模态 latent），c = batch['cond']（text 模态乐谱）。"""
        x = batch.get(self.first_stage_key)
        if x is None:
            x = super().get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        z = (x * self.scale_factor).detach()

        if self.model.conditioning_key is not None:
            key = cond_key if cond_key is not None else self.cond_stage_key
            if key in batch:
                xc = batch[key]
                c = self.get_learned_conditioning(xc)
                # 如果 batch 中提供了 prompt latent，则一并放入 cond，供 backbone 使用
                if "prompt" in batch and isinstance(c, dict):
                    prompt = batch["prompt"]
                    if bs is not None and isinstance(prompt, torch.Tensor):
                        prompt = prompt[:bs]
                    c["prompt"] = prompt
                if bs is not None and isinstance(c, dict):
                    c = {k: v[:bs] if isinstance(v, torch.Tensor) and v.dim() > 0 else v for k, v in c.items()}
                # 把 cond 里所有 tensor 放到当前 device，否则 frontend/backbone 会在 CPU 上算，显存占用低且可能 device 报错
                if c is not None and isinstance(c, dict):
                    c = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in c.items()}
            else:
                c = None
                xc = None
        else:
            c = None
            xc = None

        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        """
        audio 模态 x_noisy + text 模态 cond（music score）→ frontend 得到 frame 级 text 特征 →
        与 audio 在 backbone 内 concat 后做 self-attention，仅从 audio 流输出 flow / F0。
        """
        # text 模态：乐谱 token（phn, note dur, note pitch）
        phn = cond["phn"].long()
        midi = cond["pitches"].long()
        notedurs = cond["notedurs"].float()
        notetypes = cond["notetypes"].long()
        y_spk = cond["spk_id"]
        f0_gt = cond.get("f0_gt", None)
        infer = cond.get("infer", not self.training)
        # 可选：prompt latent，形状 [B, C, T]
        prompt_latent = cond.get("prompt", None)
        padding_mask = (phn == PHN_PAD_ID)

        pred_dur_log = self.frontend(phn, notedurs, midi, notetypes, padding_mask)

        if infer:
            dur = torch.clamp(torch.round(torch.exp(pred_dur_log) - 1), min=1).long()
        else:
            # 送入 LengthRegulator 的 dur 至少为 1，避免 repeat_interleave(..., 0) 产生空序列
            dur = cond["dur_gt"].long().clamp(min=1)

        phn = self.length_regulator(phn, dur)
        midi = self.length_regulator(midi, dur)
        if self.use_bbc:
            phn = self.bbc_adaptor(phn, dur, is_training=not infer)
        # 与 audio 帧对齐
        max_len = x_noisy.shape[2]
        if phn.shape[1] > max_len:
            phn = phn[:, :max_len]
        elif phn.shape[1] < max_len:
            phn = F.pad(phn, (0, max_len - phn.shape[1]))
        
        if midi.shape[1] > max_len:
            midi = midi[:, :max_len]
        elif midi.shape[1] < max_len:
            midi = F.pad(midi, (0, max_len - midi.shape[1]))

        cond = {
            "c_concat": {
                "phn": phn,
                "midi": midi,
                "f0_gt": f0_gt,
                "spk_id": y_spk,
                "prompt": prompt_latent,
            },
            "c_crossattn": None,
            "name": None,
            "infer": infer
        } # only concat, no crossattn
        out = self.model(x_noisy, t, **cond)

        if not infer:
            return out[0], out[1], out[2], pred_dur_log
        return out[0], out[1], out[2], pred_dur_log

    def p_losses(self, x_start, cond, t, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        t_unsqueeze = t.unsqueeze(1).unsqueeze(2).float() / self.num_timesteps
        x_noisy = t_unsqueeze * x_start + (1.0 - (1 - self.sigma_min) * t_unsqueeze) * noise
        ut = x_start - (1 - self.sigma_min) * noise

        u_pred, ret_loss_dict, f0_out, pred_dur_log = self.apply_model(x_noisy, t, cond)

        # Flow matching 主损失：只对 target 帧计算，已经在apply_model中mask掉了prompt段
        loss_cfm = self.get_loss(u_pred, ut, mean=True)

        padding_mask = cond["phn"] == PHN_PAD_ID
        dur_gt_log = torch.log1p(cond["dur_gt"].float().clamp(min=0))
        pred_dur_valid = pred_dur_log.masked_select(~padding_mask)
        dur_gt_valid = dur_gt_log.masked_select(~padding_mask)
        if pred_dur_valid.numel() > 0:
            loss_dur = F.mse_loss(pred_dur_valid, dur_gt_valid, reduction="mean")
        else:
            loss_dur = torch.tensor(0.0, device=x_start.device, dtype=x_start.dtype)

        total_loss = (
            self.loss_w_cfm * loss_cfm +
            self.loss_w_f0 * ret_loss_dict["loss_f0"] +
            self.loss_w_uv * ret_loss_dict["loss_uv"] +
            self.loss_w_dur * loss_dur
        )
        prefix = "train" if self.training else "val"
        loss_dict = {
            f"{prefix}/loss_cfm": loss_cfm.detach(),
            f"{prefix}/loss_f0_uv": ret_loss_dict["loss_f0"].detach(),
            f"{prefix}/loss_dur": loss_dur.detach(),
            f"{prefix}/total_loss": total_loss.detach(),
        }
        return total_loss, loss_dict
