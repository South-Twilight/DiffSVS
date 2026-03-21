import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange

from ldm.dataset.diffsvs_dataset import PHN_PAD_ID, PITCH_PAD_ID

from ldm.modules.diffusionmodules.flag_large_dit_moe import (
    Attention,
    RMSNorm,
    modulate,
    TimestepEmbedder,
    ConditionEmbedder,
)


class ConvMLP(nn.Module):
    """(B,T,C) -> Conv1d MLP -> (B,T,C)，用来替换原来的 MoE / FFN。"""
    def __init__(self, dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x = x.transpose(1, 2)
        x = self.conv2(self.act(self.conv1(x)))
        return x.transpose(1, 2)


class TransformerBlockConvMLP(nn.Module):
    """参考 TCSinger2 的 TransformerBlock，Attention + ConvMLP FFN + AdaLN。"""
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int],
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        norm_eps: float,
        qk_norm: bool,
        y_dim: int,
    ):
        super().__init__()
        self.dim = dim
        self.attention = Attention(dim, n_heads, n_kv_heads, qk_norm, y_dim)

        # 按原 FeedForward 的规则算 hidden_dim
        hidden_dim = int(2 * dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.ffn = ConvMLP(dim, hidden_dim, kernel_size=3)
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        self.attention_y_norm = RMSNorm(y_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        y: torch.Tensor,
        y_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(adaln_input).chunk(6, dim=1)

            h = x + gate_msa.unsqueeze(1) * self.attention(
                modulate(self.attention_norm(x), shift_msa, scale_msa),
                x_mask,
                freqs_cis,
                self.attention_y_norm(y),
                y_mask,
            )
            out = self.ffn(
                modulate(self.ffn_norm(h), shift_mlp, scale_mlp)
            )
            out = h + gate_mlp.unsqueeze(1) * out
        else:
            h = x + self.attention(
                self.attention_norm(x), x_mask, freqs_cis, self.attention_y_norm(y), y_mask
            )
            out = h + self.ffn(self.ffn_norm(h))

        # 为了兼容 (out, loss) 的模式，这里返回一个 0
        return out, torch.tensor(0.0, device=x.device, dtype=x.dtype)


class FinalLayer(nn.Module):
    """和 TCSinger2 的 FinalLayer 一致。"""
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6,
        )
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiffSVS_Backbone(nn.Module):

    def __init__(
        self,
        in_channels: int,          # latent 通道数
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        max_len: int = 1000,
        n_kv_heads: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        rope_scaling_factor: float = 1.0,
        ntk_factor: float = 1.0,
        num_speakers: int = 100,
        phn_vocab: int = 60,
        midi_vocab: int = 130,
        loss_weights: dict = {
            "cfm": 1.0,
            "f0_uv": 1.0,
            "dur": 1.0,
        },
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.downsample_rate = 1 # latent相对于audio的downsample rate
        kernel_size = 9

        # 仅时间嵌入（AdaLN 中不再使用说话人条件）
        self.t_embedder = TimestepEmbedder(hidden_size)

        # audio latent 投影（时间维 concat(prompt, x) 后整段一起投影）
        self.proj_in = nn.Conv1d(in_channels, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2)

        # phn / midi embedding + proj
        self.midi_embedding = nn.Embedding(midi_vocab, hidden_size)
        self.phn_embedding = nn.Embedding(phn_vocab, hidden_size)
        # phn/midi 在 DiffSVS_System 中已对齐到 latent 时间维，不再下采样
        self.midi_proj = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(),
        )
        self.phn_proj = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(),
        )
        self.prompt_proj = nn.Conv1d(
            in_channels, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.content_proj = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)

        # TCSinger2 风格：Gated Add 融合（不使用 concat + linear）
        self.gate_content = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.gate_f0 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        # F0 分支
        self.f0_proj = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(),
            nn.AvgPool1d(self.downsample_rate),
        )
        self.f0_regressor = nn.Linear(hidden_size, 1)
        self.f0_upsample = nn.Upsample(scale_factor=self.downsample_rate, mode="linear", align_corners=False)
        self.uv_classifier = nn.Linear(hidden_size, 2)

        # Transformer blocks
        self.pre_transformer = TransformerBlockConvMLP(
            -1, hidden_size, num_heads, n_kv_heads,
            multiple_of, ffn_dim_multiplier, norm_eps,
            qk_norm, hidden_size,
        )
        self.blocks = nn.ModuleList([
            TransformerBlockConvMLP(
                i, hidden_size, num_heads, n_kv_heads,
                multiple_of, ffn_dim_multiplier, norm_eps,
                qk_norm, hidden_size,
            )
            for i in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.rope_scaling_factor = rope_scaling_factor
        self.ntk_factor = ntk_factor
        # RoPE 预计算
        with torch.no_grad():
            self.register_buffer(
                "_freqs_cis",
                self.precompute_freqs_cis(
                    hidden_size // num_heads,
                    max_len,
                    rope_scaling_factor=rope_scaling_factor,
                    ntk_factor=ntk_factor,
                    device=torch.device("cpu"),
                ),
            )

        self.initialize_weights()

    def forward(self, x, t, context):
        """
        Args:
            x: noisy target latent，形状 [B, C_latent, T_lat]。
            t: diffusion timestep，形状 [B]。
            context: 字典，期望结构：
                context['c_concat'] = {
                    'phn':    [B, T_lat] 或 [B, 1, T_lat],
                    'midi':   [B, T_lat] 或 [B, 1, T_lat],
                    'f0_gt':  [B, 1, T_lat]（训练时，可选）,
                    'spk_id': [B],  # 当前 forward 不显式使用，但保留字段
                    'prompt': [B, C_latent, T_p]（可选，prompt latent，音色参考片段）
                }
                context['infer']: bool，推理模式标志。

        主要流程：
            1. 使用 Conv1d 分别处理 prompt 和 target latent：
               - prompt -> prompt_proj -> [B, T_p, H]
               - x      -> proj_in      -> [B, T_lat, H]
               在时间维拼接为 x_full = [prompt_feat, x_proj]，形状 [B, T_p + T_lat, H]。
            2. 使用 PHN_PAD_ID / PITCH_PAD_ID 在 ID 级别对 phn/midi pad 到全长，
               再各自经过 embedding+proj，最后相加得到 content（[B, T_p + T_lat, H]），不再用物理零点 F.pad。
            3. 使用 TCSinger2 风格的 Gated Add 将 content 注入到 x_full 中（不使用 concat + linear）。
               仅取 pre_transformer 输出的 target 段 feats_target = x[:, T_p:, :] 做 F0 / UV 预测，
               避免 prompt 污染 F0/UV 回归器的感受野。
            4. AdaLN 条件 adaln_input = t_embed + prompt_global：
               - t_embed: 时间嵌入 [B, H]
               - prompt_global: prompt_feat 在时间维平均得到的全局音色向量 [B, H]（无 prompt 时为 0）
               该条件用于 pre_transformer、后续所有 TransformerBlock 以及 FinalLayer。
        """
        assert isinstance(context, dict)
        acoustic = context["c_concat"]
        infer = context.get("infer", not self.training)

        phn = acoustic["phn"]
        midi = acoustic["midi"]
        f0_gt = acoustic.get("f0_gt", None)
        # spk_id 仍然保留在数据流中（方便将来扩展），但本 forward 不显式使用
        # y_spk = acoustic["spk_id"]
        prompt = acoustic.get("prompt", None)

        B, _, T_lat = x.shape
        device = x.device
        T_p = prompt.shape[2] if prompt is not None else 0
        T_total = T_p + T_lat

        # phn / midi 处理（仅对 target 长度 T_lat）
        if phn.dim() == 3:
            phn_ids = phn.squeeze(1)
        else:
            phn_ids = phn
        if midi.dim() == 3:
            midi_ids = midi.squeeze(1)
        else:
            midi_ids = midi

        # 1) phn / midi 特征（target 段）
        phn_feat = self.phn_proj(self.phn_embedding(phn_ids).transpose(1, 2)).transpose(1, 2)   # [B, T_lat, H]
        midi_feat = self.midi_proj(self.midi_embedding(midi_ids).transpose(1, 2)).transpose(1, 2)  # [B, T_lat, H]

        # 2) prompt 段的 content 占位（使用 PAD ID 通过 embedding / proj）
        if T_p > 0:
            phn_pad_ids = torch.full(
                (B, T_p), PHN_PAD_ID, dtype=phn_ids.dtype, device=device
            )
            midi_pad_ids = torch.full(
                (B, T_p), PITCH_PAD_ID, dtype=midi_ids.dtype, device=device
            )
            phn_pad_feat = self.phn_proj(self.phn_embedding(phn_pad_ids).transpose(1, 2)).transpose(1, 2)
            midi_pad_feat = self.midi_proj(self.midi_embedding(midi_pad_ids).transpose(1, 2)).transpose(1, 2)
            phn_full = torch.cat([phn_pad_feat, phn_feat], dim=1)
            midi_full = torch.cat([midi_pad_feat, midi_feat], dim=1)
            content = phn_full + midi_full  # [B, T_total, H]
        else:
            content = phn_feat + midi_feat  # [B, T_lat, H]

        content= self.content_proj(content.transpose(1, 2)).transpose(1, 2)  # [B, T_total, H]

        # 3) audio latent 投影 + prompt 局部特征 + 拼接
        #    - target 段：x -> proj_in -> [B, T_lat, H]
        #    - prompt 段：prompt -> prompt_proj -> [B, T_p, H]
        x_proj = self.proj_in(x).transpose(1, 2)  # [B, T_lat, H]
        if prompt is not None and T_p > 0:
            prompt_feat = self.prompt_proj(prompt).transpose(1, 2)  # [B, T_p, H]
            x_full = torch.cat([prompt_feat, x_proj], dim=1)        # [B, T_total, H]
            prompt_global = prompt_feat.mean(dim=1)                 # [B, H]
        else:
            x_full = x_proj                                         # [B, T_lat, H]
            prompt_global = torch.zeros(B, self.hidden_size, device=device, dtype=x_proj.dtype)

        # 4) 仅使用时间嵌入 + prompt 全局特征作为 AdaLN 条件（不显式使用 spk 信息）
        t_emb = self.t_embedder(t)                                  # [B, H]
        adaln_input = t_emb + prompt_global                         # [B, H]

        # 5) audio + content 融合（Gated Add；对齐 TCSinger2 无 Caption）
        x = x_full
        gate_content = torch.sigmoid(self.gate_content(content))  # [B, T_total, H]
        x = x + content * gate_content

        mask = torch.ones((B, x.size(1)), dtype=torch.int32, device=device)
        y_mask = mask.bool()
        freqs_cis = self._get_freqs_cis(x.size(1), device)

        # pre_transformer：纯 self-attention（y=x，y_mask 使用 bool 类型）
        x, _ = self.pre_transformer(x, mask, x, y_mask, freqs_cis, adaln_input=adaln_input)

        # 仅使用 target 段特征进行 F0 / UV 预测，避免 prompt 污染
        feats_target = x[:, T_p:, :]  # [B, T_lat, H]
        f0_pred_lat = self.f0_regressor(feats_target).transpose(1, 2)  # [B, 1, T_lat/down]
        f0_pred = self.f0_upsample(f0_pred_lat)                        # [B, 1, T_lat]
        uv_logits_lat = self.uv_classifier(feats_target)               # [B, T_lat, 2]
        uv_pred_lat = torch.argmax(uv_logits_lat, dim=-1, keepdim=True).float().permute(0, 2, 1)

        f0_loss = uv_loss = torch.tensor(0.0, device=device, dtype=x.dtype)
        if infer:
            uv_pred_orig = uv_pred_lat.repeat_interleave(self.downsample_rate, dim=2)[:, :, : f0_pred.shape[-1]]
            f0_for_cond = f0_pred.detach() * uv_pred_orig
        else:
            if f0_gt is not None:
                f0_gt_safe = f0_gt.clamp(min=0.0, max=1500.0)
                f0_gt_log = torch.log(f0_gt_safe + 1.0)
                uv_mask = (f0_gt_safe > 0).float()
                f0_loss = F.smooth_l1_loss(f0_pred * uv_mask, f0_gt_log * uv_mask)
                uv_gt = (f0_gt_safe.squeeze(1) > 0).long()
                uv_loss = F.cross_entropy(uv_logits_lat.reshape(-1, 2), uv_gt.reshape(-1))
                f0_for_cond = f0_gt_log
            else:
                uv_pred_orig = uv_pred_lat.repeat_interleave(self.downsample_rate, dim=2)[:, :, : f0_pred.shape[-1]]
                f0_for_cond = f0_pred * uv_pred_orig
            f0_pred = f0_pred * uv_pred_lat.repeat_interleave(self.downsample_rate, dim=2)[:, :, : f0_pred.shape[-1]]

        # f0 注入（对齐 TCSinger2）：prompt 段与 target 段分别过 f0_proj，再在 feature 维拼接后 gated-add
        f0_feat = self.f0_proj(f0_for_cond).transpose(1, 2)  # [B, T_lat, H]
        if T_p > 0:
            f0_pad = torch.zeros(
                (B, 1, T_p * self.downsample_rate),
                device=device,
                dtype=f0_for_cond.dtype,
            )
            f0_pad_feat = self.f0_proj(f0_pad).transpose(1, 2)  # [B, T_p, H]
            f0_full = torch.cat([f0_pad_feat, f0_feat], dim=1)   # [B, T_total, H]
        else:
            f0_full = f0_feat  # [B, T_lat, H]

        gate_f0 = torch.sigmoid(self.gate_f0(f0_full))
        x = x + f0_full * gate_f0

        for block in self.blocks:
            # 纯 self-attention：y=x，无 MoE 辅助 loss
            x, _ = block(x, mask, x, y_mask, freqs_cis, adaln_input=adaln_input)
        loss_dict = {
            "loss_f0": f0_loss,
            "loss_uv": uv_loss,
        }

        x_final = self.final_layer(x, adaln_input)
        # flow 只取 target 段（后 T_lat 帧）
        x_final_target = x_final[:, T_p:, :]
        flow_v = rearrange(x_final_target, "b t c -> b c t")
        f0_out = torch.exp(f0_for_cond).detach() - 1.0
        return flow_v, loss_dict, f0_out

    @staticmethod
    def precompute_freqs_cis(
        dim: int,
        end: int,
        theta: float = 10000.0,
        rope_scaling_factor: float = 1.0,
        ntk_factor: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        theta = theta * ntk_factor
        half = dim // 2
        freqs = 1.0 / (
            theta ** (torch.arange(0, dim, 2)[:half].float().to(device=device) / dim)
        )
        t = torch.arange(end, device=freqs.device, dtype=torch.float)
        t = t / rope_scaling_factor
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def _get_freqs_cis(self, seq_len: int, device: torch.device):
        return self._freqs_cis.to(device)[:seq_len]

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.pre_transformer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.pre_transformer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
