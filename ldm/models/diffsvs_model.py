import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 基础组件：调制、RMSNorm、MLP、ConvMLP、Timestep
# ==========================================
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.scale

class TextMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, x):
        return self.net(x)

class AudioConvMLP(nn.Module):
    def __init__(self, dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(hidden_dim, dim, kernel_size=1)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv2(self.act(self.conv1(x)))
        return x.transpose(1, 2)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

# ==========================================
# Transformer Blocks (双流与单流)
# ==========================================
class MMAudio_MMBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # 12个参数: x和c 各自的 (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 12 * dim))
        
        self.qkv_x = nn.Linear(dim, 3 * dim, bias=False)
        self.qkv_c = nn.Linear(dim, 3 * dim, bias=False)
        self.rmsnorm_q_x = RMSNorm(dim)
        self.rmsnorm_k_x = RMSNorm(dim)
        self.rmsnorm_q_c = RMSNorm(dim)
        self.rmsnorm_k_c = RMSNorm(dim)
        
        self.proj_x = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.proj_c = nn.Linear(dim, dim)                           
        
        hidden_dim = int(dim * 4)
        self.ffn_x = AudioConvMLP(dim, hidden_dim)                  
        self.ffn_c = TextMLP(dim, hidden_dim)                       

    def forward(self, x, c, c_g):
        params = self.adaLN_modulation(c_g)
        (shift_x1, scale_x1, gate_x1, shift_x2, scale_x2, gate_x2,
         shift_c1, scale_c1, gate_c1, shift_c2, scale_c2, gate_c2) = params.chunk(12, dim=1)
        
        x_mod = modulate(x, shift_x1, scale_x1)
        c_mod = modulate(c, shift_c1, scale_c1)
        
        qkv_x = self.qkv_x(x_mod).chunk(3, dim=-1)
        qkv_c = self.qkv_c(c_mod).chunk(3, dim=-1)
        
        q_x, k_x, v_x = self.rmsnorm_q_x(qkv_x[0]), self.rmsnorm_k_x(qkv_x[1]), qkv_x[2]
        q_c, k_c, v_c = self.rmsnorm_q_c(qkv_c[0]), self.rmsnorm_k_c(qkv_c[1]), qkv_c[2]
        
        q = torch.cat([q_x, q_c], dim=1)
        k = torch.cat([k_x, k_c], dim=1)
        v = torch.cat([v_x, v_c], dim=1)
        
        B, L_total, D = q.shape
        q = q.view(B, L_total, self.num_heads, D // self.num_heads).transpose(1, 2)
        k = k.view(B, L_total, self.num_heads, D // self.num_heads).transpose(1, 2)
        v = v.view(B, L_total, self.num_heads, D // self.num_heads).transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, L_total, D)
        attn_x, attn_c = attn_out.split([x.shape[1], c.shape[1]], dim=1)
        
        attn_x = self.proj_x(attn_x.transpose(1, 2)).transpose(1, 2)
        x = x + gate_x1.unsqueeze(1) * attn_x
        c = c + gate_c1.unsqueeze(1) * self.proj_c(attn_c)
        
        x_mod_mlp = modulate(x, shift_x2, scale_x2)
        c_mod_mlp = modulate(c, shift_c2, scale_c2)
        
        x = x + gate_x2.unsqueeze(1) * self.ffn_x(x_mod_mlp)
        c = c + gate_c2.unsqueeze(1) * self.ffn_c(c_mod_mlp)
        return x, c

class MMAudio_SingleBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.rmsnorm_q = RMSNorm(dim)
        self.rmsnorm_k = RMSNorm(dim)
        self.proj = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.ffn = AudioConvMLP(dim, int(dim * 4))

    def forward(self, x, c_g):
        shift_x1, scale_x1, gate_x1, shift_x2, scale_x2, gate_x2 = self.adaLN_modulation(c_g).chunk(6, dim=1)
        x_mod = modulate(x, shift_x1, scale_x1)
        qkv = self.qkv(x_mod).chunk(3, dim=-1)
        q, k, v = self.rmsnorm_q(qkv[0]), self.rmsnorm_k(qkv[1]), qkv[2]
        
        B, L, D = q.shape
        q = q.view(B, L, self.num_heads, -1).transpose(1, 2)
        k = k.view(B, L, self.num_heads, -1).transpose(1, 2)
        v = v.view(B, L, self.num_heads, -1).transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, L, D)
        attn_out = self.proj(attn_out.transpose(1, 2)).transpose(1, 2)
        x = x + gate_x1.unsqueeze(1) * attn_out
        
        x_mod_mlp = modulate(x, shift_x2, scale_x2)
        x = x + gate_x2.unsqueeze(1) * self.ffn(x_mod_mlp)
        return x

# ==========================================
# 核心主干：DiffSVS MMAudio Backbone
# ==========================================
class DiffSVS_MMAudio_Backbone(nn.Module):
    def __init__(self, in_channels=128, hidden_size=768, num_heads=12, N1=10, N2=14, num_speakers=100):
        super().__init__()
        
        self.x_proj = nn.Linear(in_channels, hidden_size)
        self.c_proj = nn.Linear(hidden_size, hidden_size)  # 假设前端特征维度也是 hidden_size
        self.time_embedder = TimestepEmbedder(hidden_size)
        self.singer_embedder = nn.Embedding(num_speakers, hidden_size)
        
        # --- 🌟 纯 Latent F0 引擎 ---
        self.pre_mm_block = MMAudio_MMBlock(hidden_size, num_heads)
        self.f0_regressor = nn.Linear(hidden_size, 1)
        self.uv_classifier = nn.Linear(hidden_size, 2)
        self.f0_proj = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=9, padding=4),
            nn.LeakyReLU()
        )
        self.gate_f0 = nn.Linear(hidden_size, hidden_size, bias=True)

        # --- 深层网络 ---
        self.mm_blocks = nn.ModuleList([MMAudio_MMBlock(hidden_size, num_heads) for _ in range(N1 - 1)])
        self.single_blocks = nn.ModuleList([MMAudio_SingleBlock(hidden_size, num_heads) for _ in range(N2)])
        
        # --- 输出层 ---
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.final_conv = nn.Conv1d(hidden_size, in_channels, kernel_size=3, padding=1)
        
        self.initialize_weights()

    def initialize_weights(self):
        # 零初始化所有 DiT block 的 AdaLN 输出层，保证训练初期为恒等映射
        for block in [self.pre_mm_block] + list(self.mm_blocks) + list(self.single_blocks):
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)
        nn.init.constant_(self.final_conv.weight, 0)
        nn.init.constant_(self.final_conv.bias, 0)

    def forward(self, x_noisy, t, c_text, y_spk, f0_gt=None, infer=False):
        """
        x_noisy: [B, C, T_lat]
        c_text: [B, T_lat, D] (已经被前端拉伸并平滑对齐好)
        """
        x = self.x_proj(x_noisy.transpose(1, 2))  # [B, T_lat, D]
        c = self.c_proj(c_text)
        c_g = self.time_embedder(t) + self.singer_embedder(y_spk)
        
        x, c = self.pre_mm_block(x, c, c_g)
        
        # ================= F0 预测与注入 =================
        f0_pred_lat = self.f0_regressor(x).transpose(1, 2)  # [B, 1, T_lat]
        uv_logits_lat = self.uv_classifier(x)               # [B, T_lat, 2]
        uv_pred_lat = torch.argmax(uv_logits_lat, dim=-1, keepdim=True).float().permute(0, 2, 1)

        f0_loss, uv_loss = 0.0, 0.0
        
        if infer:
            f0_for_cond = f0_pred_lat.detach() * uv_pred_lat
        else:
            assert f0_gt is not None, "训练时必须提供 f0_gt"
            f0_gt_log = torch.log(f0_gt + 1.0)
            uv_mask = (f0_gt > 0).float()
            
            f0_loss = F.mse_loss(f0_pred_lat * uv_mask, f0_gt_log * uv_mask)
            uv_gt_labels = (f0_gt.squeeze(1) > 0).long()
            uv_loss = F.cross_entropy(uv_logits_lat.view(-1, 2), uv_gt_labels.view(-1))
            
            f0_for_cond = f0_gt_log
            f0_pred_lat = f0_pred_lat * uv_pred_lat

        f0_feat = self.f0_proj(f0_for_cond).transpose(1, 2)
        gate = torch.sigmoid(self.gate_f0(f0_feat))
        x = x + f0_feat * gate
        # =================================================

        for block in self.mm_blocks:
            x, c = block(x, c, c_g)
        for block in self.single_blocks:
            x = block(x, c_g)
            
        shift, scale = self.final_adaLN(c_g).chunk(2, dim=1)
        x = modulate(x, shift, scale)
        flow_v = self.final_conv(x.transpose(1, 2))  # [B, C, T_lat]
        
        return flow_v, (f0_loss + uv_loss), torch.exp(f0_for_cond).detach() - 1.0
