import os
from pytorch_memlab import LineProfiler,profile
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps
from torchvision.utils import make_grid
try:
    from pytorch_lightning.utilities.distributed import rank_zero_only
except:
    from pytorch_lightning.utilities import rank_zero_only # torch2
from torchdyn.core import NeuralODE
from ldm.models.diffusion.cfm1_audio import Wrapper, Wrapper_cfg
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from omegaconf import ListConfig

from ldm.util import log_txt_as_img, exists, default

class CFMSampler(object):

    def __init__(self, model, num_timesteps, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.num_timesteps = num_timesteps
        self.schedule = schedule

    def _infer_device(self) -> torch.device:
        """兼容 LightningModule 与 DDP 包装模型。"""
        m = self.model
        dev = getattr(m, "device", None)
        if isinstance(dev, torch.device):
            return dev
        return next(m.parameters()).device

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def _get_sigma_t(self, t, sigma, sigma_schedule="constant", sigma_min=0.0):
        """Return time-dependent noise scale sigma_t for forward SDE."""
        if sigma_schedule == "constant":
            sigma_t = sigma
        elif sigma_schedule == "linear_decay":
            sigma_t = sigma * (1.0 - t)
        elif sigma_schedule == "cosine_decay":
            sigma_t = sigma * torch.cos(0.5 * np.pi * t)
        elif sigma_schedule == "flow_grpo":
            sigma_t = sigma * torch.sqrt((1.0 - t) / t)
        else:
            raise ValueError(f"Unsupported sigma_schedule: {sigma_schedule}")
        return torch.clamp(sigma_t, min=sigma_min)

    def _sample_loop(
        self,
        wrapper,
        shape,
        timesteps,
        x_latent=None,
        t_start=None,
        sampler_type="ode",
        sigma=0.0,
        sigma_schedule="constant",
        sigma_min=0.0,
        noise_start_t=0.05,
        noise_stop_t=0.95,
        score_denom_eps: float = 1e-4,
    ):
        """
        Forward-time sampling: t increases from 0 -> 1.

        Sandwich strategy for sampler_type="sde":
          Phase 1 (t < noise_start_t): pure ODE — avoid t→0 singularity in σ_t.
          Phase 2 (noise_start_t <= t <= noise_stop_t): SDE with score-corrected drift + noise.
          Phase 3 (t > noise_stop_t): pure ODE — avoid t→1 score singularity and tail noise.

        """
        dev = self._infer_device()
        t_span = torch.linspace(0.0, 1.0, 25 if timesteps is None else timesteps, device=dev)
        if t_start is not None:
            t_span = t_span[t_start:]

        x = torch.randn(shape, device=dev) if x_latent is None else x_latent
        traj = [x]
        if t_span.numel() <= 1:
            return x, torch.stack(traj, dim=0)

        sigma_tensor = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)

        for i in range(t_span.numel() - 1):
            t_cur = t_span[i]
            t_next = t_span[i + 1]
            dt = t_next - t_cur
            dt_sqrt = torch.sqrt(torch.clamp(dt, min=0.0))

            # Wrapper follows torchdyn signature forward(t, x, args).
            velocity = wrapper(t_cur, x, None)

            if sampler_type == "ode":
                drift = velocity
                sigma_eff = torch.zeros((), device=x.device, dtype=x.dtype)
            elif sampler_type == "sde":
                # Phase 1 & 3: ODE only (sandwich ends)
                if t_cur < noise_start_t or t_cur > noise_stop_t:
                    drift = velocity
                    sigma_eff = torch.zeros((), device=x.device, dtype=x.dtype)
                else:
                    # Phase 2: full SDE (t_cur away from 0 and 1 by construction of the window)
                    sigma_t = self._get_sigma_t(
                        t_cur,
                        sigma=sigma_tensor,
                        sigma_schedule=sigma_schedule,
                        sigma_min=sigma_min,
                    )
                    one_minus_t = torch.clamp(
                        1.0 - t_cur, min=float(score_denom_eps)
                    )
                    score = (t_cur / one_minus_t) * velocity - (x / one_minus_t)
                    drift = velocity + 0.5 * (sigma_t ** 2) * score
                    sigma_eff = sigma_t
            else:
                raise ValueError(f"Unsupported sampler_type: {sampler_type}")


            # Euler–Maruyama: only sample Gaussian noise when diffusion is active (save memory / RNG)
            if (sigma_eff > 0).item():
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = x + drift * dt + sigma_eff * dt_sqrt * noise
            traj.append(x)

        return x, torch.stack(traj, dim=0)

    def stochastic_encode(self, x_start, t, noise=None):
        x1 = x_start
        x0 = default(noise, lambda: torch.randn_like(x_start))
        t_unsqueeze = 1 - t.unsqueeze(1).unsqueeze(1).float() / self.num_timesteps
        x_noisy = t_unsqueeze * x1 + (1. - (1 - self.model.sigma_min) * t_unsqueeze) * x0
        return x_noisy

    @torch.no_grad()
    def sample(self, cond, batch_size=16, timesteps=None, shape=None, x_latent=None, t_start=None, **kwargs):
        
        # print(shape)
        
        if shape is None:
            if self.model.channels > 0:
                shape = (batch_size, self.model.channels, self.model.mel_dim, self.model.mel_length)
            else:
                shape = (batch_size, self.model.mel_dim, self.model.mel_length)
        if len(shape)==3:
            C, H, W = shape
            shape = (batch_size, C, H, W)
        else:
            C, T = shape
            shape = (batch_size, C, T) 

        wrapper = self.ode_wrapper(cond)
        x_final, traj = self._sample_loop(
            wrapper=wrapper,
            shape=shape,
            timesteps=timesteps,
            x_latent=x_latent,
            t_start=t_start,
            sampler_type=kwargs.get("sampler_type", "ode"),
            sigma=kwargs.get("sigma", 0.0),
            sigma_schedule=kwargs.get("sigma_schedule", "constant"),
            sigma_min=kwargs.get("sigma_min", 0.0),
            noise_start_t=kwargs.get("noise_start_t", 0.05),
            noise_stop_t=kwargs.get("noise_stop_t", 0.95),
            score_denom_eps=float(kwargs.get("score_denom_eps", 1e-4)),
        )
        return x_final, traj

    def ode_wrapper(self, cond):
        # self.estimator receives x, mask, mu, t, spk as arguments
        return Wrapper(self.model, cond)

    @torch.no_grad()
    def sample_cfg(
        self,
        cond,
        unconditional_guidance_scale,
        unconditional_conditioning,
        batch_size=16,
        timesteps=None,
        shape=None,
        x_latent=None,
        t_start=None,
        **kwargs,
    ):
        if shape is None:
            if self.model.channels > 0:
                shape = (batch_size, self.model.channels, self.model.mel_dim, self.model.mel_length)
            else:
                shape = (batch_size, self.model.mel_dim, self.model.mel_length)

        if len(shape)==3:
            C, H, W = shape
            shape = (batch_size, C, H, W)
        else:
            C, T = shape
            shape = (batch_size, C, T) 

        wrapper = self.ode_wrapper_cfg(cond, unconditional_guidance_scale, unconditional_conditioning)
        x_final, traj = self._sample_loop(
            wrapper=wrapper,
            shape=shape,
            timesteps=timesteps,
            x_latent=x_latent,
            t_start=t_start,
            sampler_type=kwargs.get("sampler_type", "ode"),
            sigma=kwargs.get("sigma", 0.7),
            sigma_schedule=kwargs.get("sigma_schedule", "constant"),
            sigma_min=kwargs.get("sigma_min", 0.0),
            noise_start_t=kwargs.get("noise_start_t", 0.05),
            noise_stop_t=kwargs.get("noise_stop_t", 0.95),
            score_denom_eps=float(kwargs.get("score_denom_eps", 1e-4)),
        )

        f0_preds = torch.stack(wrapper.f0_preds, dim=0)   # [T, B, ...]

        return x_final, traj, f0_preds[-1]

    def ode_wrapper_cfg(self, cond, unconditional_guidance_scale, unconditional_conditioning):
        # self.estimator receives x, mask, mu, t, spk as arguments
        return Wrapper_cfg(self.model, cond, unconditional_guidance_scale, unconditional_conditioning)
