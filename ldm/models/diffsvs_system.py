import torch
import torch.nn as nn
import torch.nn.functional as F
from ldm.models.diffusion.cfm1_audio import CFM # 请确保这里指向你本地的 CFM 基类
from ldm.util import instantiate_from_config

class LengthRegulator(nn.Module):
    """根据 duration 将 Token 级别的特征扩展到 Frame 级别"""
    def forward(self, x, dur):
        # x: [B, L, D], dur: [B, L]
        out = []
        for i in range(x.size(0)):
            # 按照预测或真实的帧数，把每个音素的特征复制拉长
            expanded = torch.repeat_interleave(x[i], dur[i], dim=0)
            out.append(expanded)
        # Pad 对齐到 Batch 内最大长度
        return torch.nn.utils.rnn.pad_sequence(out, batch_first=True)

class BlurredBoundaryAdaptor(nn.Module):
    """
    适配 f=2048 极端压缩的 BBC 边界平滑器
    因为 1 个 Latent 帧包含了约 46ms，所以只掩码 1 帧就足够覆盖协同发音了
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # 使用极小核的 Depthwise 卷积进行平滑，避免过度模糊导致吞音
        self.blur_conv = nn.Conv1d(
            hidden_dim, hidden_dim, 
            kernel_size=3, padding=1, groups=hidden_dim
        )
        self.act = nn.SiLU()

    def forward(self, c_text, dur, is_training=False):
        # c_text: [B, T_lat, D]
        if is_training:
            B, T_lat, D = c_text.shape
            # 算出每个音素的边界位置（当前帧所在的索引）
            boundaries = torch.cumsum(dur, dim=1) 
            mask = torch.ones((B, T_lat, 1), device=c_text.device)
            
            for b in range(B):
                for bndry in boundaries[b]:
                    # 防止越界
                    if 0 <= bndry < T_lat:
                        # 80% 概率触发，增加模型的抗干扰鲁棒性
                        if torch.rand(1).item() < 0.8: 
                            # 🌟 极限压缩特调：精准挖空边界所在的这 1 帧
                            mask[b, bndry, :] = 0
            
            # 乘以掩码，将边界特征强制置零
            c_text = c_text * mask

        # 用 Conv1d 把空洞“糊”起来，形成平滑的声学过渡
        x = c_text.transpose(1, 2)
        x = self.blur_conv(x)
        x = self.act(x)
        # 残差连接：保留主体特征，只在边界处柔化
        out = c_text + x.transpose(1, 2)
        return out


class DiffSVS_System(CFM):
    def __init__(self, unet_config, frontend_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 1. 实例化打工人 (MMAudio Backbone)
        self.model = instantiate_from_config(unet_config)
        
        # 2. 实例化前台 (FrontendWrapper)
        self.frontend = instantiate_from_config(frontend_config)
        
        # 3. 实例化边界处理流水线
        # 注意这里获取 frontend_config 里配置的 hidden_channels
        hidden_dim = frontend_config.params.get("hidden_channels", 768)
        self.length_regulator = LengthRegulator()
        self.bbc_adaptor = BlurredBoundaryAdaptor(hidden_dim=hidden_dim)
        
        # 🔥 全面解冻：开启端到端联合训练
        self.frontend.train()
        for param in self.frontend.parameters():
            param.requires_grad = True

    def configure_optimizers(self):
        """重写优化器，将大厨(Backbone)和前台(Frontend)的参数一并收编"""
        params = list(self.model.parameters()) + list(self.frontend.parameters())
        if self.learn_logvar:
            params.append(self.logvar)
            
        opt = torch.optim.AdamW(params, lr=self.learning_rate)
        return opt

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        """
        核心数据流转拦截器：
        文本 -> 提特征 -> 算时长 -> 拉伸 -> 模糊边界 -> 强对齐 -> 送给 DiT
        """
        # 1. 解析 DataLoader 吐出的条件字典
        ph = cond['ph'].long()               # [B, L]
        pitches = cond['pitches'].long()     # [B, L]
        notedurs = cond['notedurs'].float()  # [B, L]
        notetypes = cond['notetypes'].long() # [B, L]
        y_spk = cond['spk_id']               # [B]
        f0_gt = cond.get('f0_gt', None)
        infer = cond.get('infer', not self.training)
        
        # 根据你提供的数据结构，Padding ID 是 59
        padding_mask = (ph == 59)
        
        # 2. 🌟 一键呼叫前端，拿到连续语义特征和预测时长
        encoded_text, pred_dur_log = self.frontend(ph, notedurs, pitches, notetypes, padding_mask)
        
        # 3. 决定拉伸使用的时长 (训练用真实，推理用预测)
        if infer:
            # 预测的是 log(dur + 1)，逆运算并限制最小为 1 帧防止崩溃
            dur = torch.clamp(torch.round(torch.exp(pred_dur_log) - 1), min=1).long()
        else:
            dur = cond['dur_gt'].long() 
            
        # 4. 拉伸与 BBC 模糊处理
        c_text = self.length_regulator(encoded_text, dur)
        c_text = self.bbc_adaptor(c_text, dur, is_training=not infer)
        
        # 5. ⚠️ 强制时间轴对齐 (极度重要)
        # 真实数据的音频经过 VAE 压缩后的长度，可能和 Dur_gt 累加有一两帧的舍入误差
        max_len = x_noisy.shape[2]
        if c_text.shape[1] > max_len:
            c_text = c_text[:, :max_len, :]
        elif c_text.shape[1] < max_len:
            pad_len = max_len - c_text.shape[1]
            c_text = F.pad(c_text, (0, 0, 0, pad_len))
            
        # 6. 送入 Backbone 预测速度场和 F0
        u_pred, f0_uv_loss, f0_pred = self.model(
            x_noisy, t, c_text, y_spk, f0_gt=f0_gt, infer=infer
        )
        
        if not infer:
            return u_pred, f0_uv_loss, f0_pred, pred_dur_log
        else:
            return u_pred, f0_uv_loss, f0_pred, dur

    def p_losses(self, x_start, cond, t, noise=None):
        """计算三大总 Loss 并返回字典用于日志记录"""
        noise = torch.randn_like(x_start) if noise is None else noise
        
        # 构建 CFM (Flow Matching) 的插值与目标速度场 ut
        t_unsqueeze = t.unsqueeze(1).unsqueeze(2).float() / self.num_timesteps
        x_noisy = t_unsqueeze * x_start + (1. - (1 - self.sigma_min) * t_unsqueeze) * noise
        ut = x_start - (1 - self.sigma_min) * noise 
        
        # 跑一整遍前向传播
        u_pred, f0_uv_loss, f0_pred, pred_dur_log = self.apply_model(x_noisy, t, cond)
        
        # Loss 1: Flow 速度场回归
        loss_cfm = F.mse_loss(u_pred, ut, reduction='mean')
        
        # Loss 2: 时长预测 Loss (忽略 Padding 区域，且 padding ID = 59)
        padding_mask = (cond['ph'] == 59)
        dur_gt_log = torch.log1p(cond['dur_gt'].float())
        loss_dur = F.mse_loss(
            pred_dur_log.masked_select(~padding_mask), 
            dur_gt_log.masked_select(~padding_mask),
            reduction='mean'
        )
        
        # 聚合三大 Loss (权重可以后续根据 TensorBoard 的曲线量级动态调整)
        total_loss = loss_cfm + 1.0 * f0_uv_loss + 1.0 * loss_dur
        
        loss_dict = {
            'train/loss_cfm': loss_cfm.detach(),
            'train/loss_f0_uv': f0_uv_loss.detach(),
            'train/loss_dur': loss_dur.detach(),
            'train/total_loss': total_loss.detach()
        }
        
        return total_loss, loss_dict