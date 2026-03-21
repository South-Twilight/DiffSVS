import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange
from ldm.dataset.diffsvs_dataset import PHN_PAD_ID, PITCH_PAD_ID
from ldm.modules.diffusionmodules.flag_large_dit_moe import TimestepEmbedder

# ==========================================
# 工具函数与基础模块
# ==========================================

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    # 给 shift 和 scale 在时间维度 (dim=1) 扩维，使其形状从 [B, D] 变为 [B, 1, D]
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    """
    应用旋转位置编码
    x: [B, n_heads, T, head_dim]
    freqs_cis: [T, head_dim // 2] (complex)
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)  # [1, 1, T, head_dim // 2]
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    return x_out.type_as(x)

class ConvMLP(nn.Module):
    """用于 Audio 流：带 1D 卷积的 MLP，捕捉局部平滑声学特征"""
    def __init__(self, dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv2(self.act(self.conv1(x)))
        return x.transpose(1, 2)

# ==========================================
# 核心组件：双流联合注意力机制 (Joint Attention)
# ==========================================

class JointAttention(nn.Module):
    """MMAudio / SD3 风格的联合注意力：QKV独立投影，拼接计算，拆分输出"""
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # 两条流各自独立的 QKV 投影
        self.qkv_audio = nn.Linear(dim, dim * 3, bias=False)
        self.qkv_cond = nn.Linear(dim, dim * 3, bias=False)

        # QK-Norm (极其重要，防止混合精度溢出)
        self.q_norm_audio = nn.RMSNorm(self.head_dim)
        self.k_norm_audio = nn.RMSNorm(self.head_dim)
        self.q_norm_cond = nn.RMSNorm(self.head_dim)
        self.k_norm_cond = nn.RMSNorm(self.head_dim)

        # 两条流独立的 Output 投影
        self.proj_audio = nn.Linear(dim, dim, bias=False)
        self.proj_cond = nn.Linear(dim, dim, bias=False)

    def forward(self, x_audio: torch.Tensor, x_cond: torch.Tensor, freqs_cis: torch.Tensor):
        B, T, _ = x_audio.shape

        # 1. 独立投影提取 QKV
        qkv_a = self.qkv_audio(x_audio)
        qkv_c = self.qkv_cond(x_cond)

        q_a, k_a, v_a = rearrange(qkv_a, 'b t (qkv h d) -> qkv b h t d', qkv=3, h=self.n_heads)
        q_c, k_c, v_c = rearrange(qkv_c, 'b t (qkv h d) -> qkv b h t d', qkv=3, h=self.n_heads)

        # 2. 独立 QK-Norm
        q_a, k_a = self.q_norm_audio(q_a), self.k_norm_audio(k_a)
        q_c, k_c = self.q_norm_cond(q_c), self.k_norm_cond(k_c)

        # 3. 施加同频 RoPE (核心：让音频和乐谱在相同时刻具有相同的绝对位置属性)
        q_a, k_a = apply_rotary_emb(q_a, freqs_cis), apply_rotary_emb(k_a, freqs_cis)
        q_c, k_c = apply_rotary_emb(q_c, freqs_cis), apply_rotary_emb(k_c, freqs_cis)

        # 4. 在序列维度 (dim=2) 拼接 (Cond/乐谱 在前，Audio 在后)
        q_joint = torch.cat([q_c, q_a], dim=2)
        k_joint = torch.cat([k_c, k_a], dim=2)
        v_joint = torch.cat([v_c, v_a], dim=2)

        out_joint = F.scaled_dot_product_attention(q_joint, k_joint, v_joint)

        # 5. 拆分回独立流 (注意：前半段是 Cond，后半段是 Audio 了)
        out_c = out_joint[:, :, :T, :]  # 提取前面的乐谱信息
        out_a = out_joint[:, :, T:, :]  # 提取后面的声学信息

        out_a = self.proj_audio(rearrange(out_a, 'b h t d -> b t (h d)'))
        out_c = self.proj_cond(rearrange(out_c, 'b h t d -> b t (h d)'))

        return out_a, out_c

class JointTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_mult: float = 4.0):
        super().__init__()
        self.attn = JointAttention(dim, n_heads)
        
        # Audio 流组件 (使用 ConvMLP 处理时序)
        self.norm1_a = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2_a = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn_a = ConvMLP(dim, int(dim * ffn_mult), kernel_size=3)
        self.adaLN_a = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

        # Cond 流组件 (使用 StandardMLP 节省算力)
        self.norm1_c = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2_c = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn_c = ConvMLP(dim, int(dim * ffn_mult))
        self.adaLN_c = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(self, x_audio, x_cond, freqs_cis, adaln_input):
        # 1. 调制参数生成
        shift_msa_a, scale_msa_a, gate_msa_a, shift_mlp_a, scale_mlp_a, gate_mlp_a = self.adaLN_a(adaln_input).chunk(6, dim=1)
        shift_msa_c, scale_msa_c, gate_msa_c, shift_mlp_c, scale_mlp_c, gate_mlp_c = self.adaLN_c(adaln_input).chunk(6, dim=1)

        # 2. 预归一化与调制
        audio_mod = modulate(self.norm1_a(x_audio), shift_msa_a, scale_msa_a)
        cond_mod = modulate(self.norm1_c(x_cond), shift_msa_c, scale_msa_c)

        # 3. 联合注意力
        attn_a, attn_c = self.attn(audio_mod, cond_mod, freqs_cis)

        # 4. 残差注入与 Gating
        x_audio = x_audio + gate_msa_a.unsqueeze(1) * attn_a
        x_cond = x_cond + gate_msa_c.unsqueeze(1) * attn_c

        # 5. 独立 FFN
        mlp_a = self.ffn_a(modulate(self.norm2_a(x_audio), shift_mlp_a, scale_mlp_a))
        mlp_c = self.ffn_c(modulate(self.norm2_c(x_cond), shift_mlp_c, scale_mlp_c))

        x_audio = x_audio + gate_mlp_a.unsqueeze(1) * mlp_a
        x_cond = x_cond + gate_mlp_c.unsqueeze(1) * mlp_c

        return x_audio, x_cond


# ==========================================
# 顶层架构：MMDiT 骨干网络
# ==========================================

class DiffSVS_BackboneMMDiT(nn.Module):
    """
    基于 MMAudio / SD3 双流联合注意力机制重构的 SVS 骨干网络。
    彻底解耦音频特征流与乐谱条件流，仅在 Attention 阶段进行交互。
    """
    def __init__(
        self,
        in_channels: int,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        max_len: int = 1000,
        rope_scaling_factor: float = 1.0,
        ntk_factor: float = 1.0,
        phn_vocab: int = 60,
        midi_vocab: int = 130,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.downsample_rate = 1

        # --- Embedding 层 ---
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.proj_in = nn.Conv1d(in_channels, hidden_size, kernel_size=9, padding=4)
        self.prompt_proj = nn.Conv1d(in_channels, hidden_size, kernel_size=9, padding=4)
        
        self.phn_embedding = nn.Embedding(phn_vocab, hidden_size)
        self.midi_embedding = nn.Embedding(midi_vocab, hidden_size)
        self.phn_proj = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, 9, padding=4), nn.LeakyReLU())
        self.midi_proj = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, 9, padding=4), nn.LeakyReLU())
        self.content_proj = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)

        # --- F0 / UV 预测模块 ---
        self.f0_proj = nn.Sequential(nn.Conv1d(1, hidden_size, 9, padding=4), nn.LeakyReLU(), nn.AvgPool1d(self.downsample_rate))
        self.f0_regressor = nn.Linear(hidden_size, 1)
        self.f0_upsample = nn.Upsample(scale_factor=self.downsample_rate, mode="linear", align_corners=False)
        self.uv_classifier = nn.Linear(hidden_size, 2)
        self.gate_f0 = nn.Linear(hidden_size, hidden_size, bias=True)

        # --- MMDiT 核心网络 ---
        self.pre_transformer = JointTransformerBlock(hidden_size, num_heads)
        self.blocks = nn.ModuleList([JointTransformerBlock(hidden_size, num_heads) for _ in range(depth)])
        
        # Output 层仅处理 Audio 流
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(hidden_size, in_channels)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

        # --- RoPE 预计算 ---
        with torch.no_grad():
            self.register_buffer(
                "_freqs_cis",
                self.precompute_freqs_cis(hidden_size // num_heads, max_len, rope_scaling_factor, ntk_factor)
            )

        self.initialize_weights()

    def forward(self, x, t, context):
        acoustic = context["c_concat"]
        infer = context.get("infer", not self.training)
        device = x.device
        
        prompt = acoustic.get("prompt", None)
        T_p = prompt.shape[2] if prompt is not None else 0
        B, _, T_lat = x.shape
        T_total = T_p + T_lat

        # ==================================
        # 1. 构造 Audio 流 (x_audio)
        # ==================================
        x_proj = self.proj_in(x).transpose(1, 2)
        if T_p > 0:
            prompt_feat = self.prompt_proj(prompt).transpose(1, 2)
            x_audio = torch.cat([prompt_feat, x_proj], dim=1)
            prompt_global = prompt_feat.mean(dim=1)
        else:
            x_audio = x_proj
            prompt_global = torch.zeros(B, self.hidden_size, device=device)

        adaln_input = self.t_embedder(t) + prompt_global

        # ==================================
        # 2. 构造 Condition 流 (x_cond)
        # ==================================
        phn, midi = acoustic["phn"].squeeze(1), acoustic["midi"].squeeze(1)
        phn_feat = self.phn_proj(self.phn_embedding(phn).transpose(1, 2)).transpose(1, 2)
        midi_feat = self.midi_proj(self.midi_embedding(midi).transpose(1, 2)).transpose(1, 2)
        
        if T_p > 0:
            phn_pad = torch.full((B, T_p), PHN_PAD_ID, dtype=phn.dtype, device=device)
            midi_pad = torch.full((B, T_p), PITCH_PAD_ID, dtype=midi.dtype, device=device)
            phn_pad_feat = self.phn_proj(self.phn_embedding(phn_pad).transpose(1, 2)).transpose(1, 2)
            midi_pad_feat = self.midi_proj(self.midi_embedding(midi_pad).transpose(1, 2)).transpose(1, 2)
            x_cond = torch.cat([phn_pad_feat, phn_feat], dim=1) + torch.cat([midi_pad_feat, midi_feat], dim=1)
        else:
            x_cond = phn_feat + midi_feat
            
        x_cond = self.content_proj(x_cond.transpose(1, 2)).transpose(1, 2)

        # 获取共享的同长 RoPE
        freqs_cis = self._freqs_cis.to(device)[:T_total]

        # ==================================
        # 3. 前置 Transformer 与 F0 预测
        # ==================================
        x_audio, x_cond = self.pre_transformer(x_audio, x_cond, freqs_cis, adaln_input)

        feats_target = x_audio[:, T_p:, :]
        f0_pred_lat = self.f0_regressor(feats_target).transpose(1, 2)
        f0_pred = self.f0_upsample(f0_pred_lat)
        uv_logits_lat = self.uv_classifier(feats_target)
        uv_pred_lat = torch.argmax(uv_logits_lat, dim=-1, keepdim=True).float().permute(0, 2, 1)

        f0_gt = acoustic.get("f0_gt", None)
        f0_loss = uv_loss = torch.tensor(0.0, device=device, dtype=x.dtype)
        
        if infer:
            uv_pred_orig = uv_pred_lat.repeat_interleave(self.downsample_rate, dim=2)[:, :, : f0_pred.shape[-1]]
            f0_for_cond = f0_pred.detach() * uv_pred_orig
        else:
            if f0_gt is not None:
                # 实施 Huber Loss 与上限截断 (极其重要)
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

        # ==================================
        # 4. F0 强约束注入 (仅注入 Audio 流)
        # ==================================
        f0_feat = self.f0_proj(f0_for_cond).transpose(1, 2)
        if T_p > 0:
            f0_pad = torch.zeros((B, 1, T_p * self.downsample_rate), device=device, dtype=f0_for_cond.dtype)
            f0_pad_feat = self.f0_proj(f0_pad).transpose(1, 2)
            f0_full = torch.cat([f0_pad_feat, f0_feat], dim=1)
        else:
            f0_full = f0_feat
            
        gate_f0 = torch.sigmoid(self.gate_f0(f0_full))
        x_audio = x_audio + f0_full * gate_f0

        # ==================================
        # 5. 主干网络深度交互
        # ==================================
        for block in self.blocks:
            x_audio, x_cond = block(x_audio, x_cond, freqs_cis, adaln_input)

        # ==================================
        # 6. Audio 流独立输出
        # ==================================
        x_audio_final = x_audio[:, T_p:, :]  # 切出 Target 段
        
        shift, scale = self.final_adaLN(adaln_input).chunk(2, dim=1)
        x_out = modulate(self.final_norm(x_audio_final), shift, scale)
        x_out = self.final_linear(x_out)
        
        flow_v = rearrange(x_out, "b t c -> b c t")
        f0_out = torch.exp(f0_for_cond).detach() - 1.0
        
        loss_dict = {"loss_f0": f0_loss, "loss_uv": uv_loss}
        return flow_v, loss_dict, f0_out

    @staticmethod
    def precompute_freqs_cis(dim, end, rope_scaling_factor=1.0, ntk_factor=1.0):
        theta = 10000.0 * ntk_factor
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(end, dtype=torch.float) / rope_scaling_factor
        freqs = torch.outer(t, freqs).float()
        return torch.polar(torch.ones_like(freqs), freqs)

    def initialize_weights(self):
        # 零初始化 Gate 逻辑，保证深层网络稳定启动
        for block in self.blocks:
            nn.init.constant_(block.adaLN_a[-1].weight, 0)
            nn.init.constant_(block.adaLN_a[-1].bias, 0)
            nn.init.constant_(block.adaLN_c[-1].weight, 0)
            nn.init.constant_(block.adaLN_c[-1].bias, 0)
        
        nn.init.constant_(self.pre_transformer.adaLN_a[-1].weight, 0)
        nn.init.constant_(self.pre_transformer.adaLN_a[-1].bias, 0)
        nn.init.constant_(self.pre_transformer.adaLN_c[-1].weight, 0)
        nn.init.constant_(self.pre_transformer.adaLN_c[-1].bias, 0)
        
        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)
        nn.init.constant_(self.final_linear.weight, 0)
        nn.init.constant_(self.final_linear.bias, 0)

