import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from ldm.dataset.diffsvs_dataset import PH_PAD_ID
# 导入你原本依赖的时长预测与编码组件
from utils.commons.rel_transformer import RelTransformerEncoder
from utils.commons.duration import DurationPredictor, NoteEncoder

class DurPredModel(pl.LightningModule):

    def __init__(self, 
                 ddconfig, 
                 learning_rate=1e-4, 
                 monitor="val/dur_loss",
                 **kwargs):
        """
        独立的时长预测模型 (Duration Predictor)
        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.monitor = monitor

        # 1. 音素编码器 (处理歌词文本音素), phn_size [0, 60)
        self.ph_encoder = RelTransformerEncoder(
            60, ddconfig['hidden_size'], ddconfig['hidden_size'],
            ddconfig['hidden_size']*4, ddconfig['num_heads'], ddconfig['enc_layers'],
            ddconfig['enc_ffn_kernel_size'], ddconfig['dropout'], 
            prenet=ddconfig['enc_prenet'], pre_ln=ddconfig['enc_pre_ln']
        )
        
        # 2. 音符编码器 (处理乐谱中的音高、音符时长、音符类型), pitch 范围 [0, 128),
        # notedur 是连续值, notetype 范围 [0, 4]
        self.note_encoder = NoteEncoder(
            n_vocab=129, 
            hidden_channels=ddconfig['hidden_size']
        )

        # 3. 时长预测器 (基于融合后的特征预测)
        self.dur_predictor = DurationPredictor(
            ddconfig['hidden_size'],
            n_chans=ddconfig['hidden_size'],
            n_layers=ddconfig['dur_predictor_layers'],
            dropout_rate=ddconfig['predictor_dropout'],
            kernel_size=ddconfig['dur_predictor_kernel']
        )

    def forward(self, ph, notedurs, pitches, notetypes, padding_mask=None):
        """
        前向传播：融合文本与音符特征，预测对应的时长分布
        """
        # 获取音素嵌入
        ph_emb = self.ph_encoder(ph) 
        
        # 获取音符特征
        note_features = self.note_encoder(pitches, notedurs, notetypes)
        
        # 特征相加融合
        combined = ph_emb + note_features
        
        # 预测 duration (B, T)
        pred_durs = self.dur_predictor(combined, x_padding=padding_mask)

        return pred_durs

    def training_step(self, batch, batch_idx):
        # 从 batch 中获取数据 (此时无需 audio/mel 等特征)
        pitches = batch['pitches'].long()
        notedurs = batch['notedurs'].float()
        notetypes = batch['notetypes'].long()
        ph = batch['ph'].long()
        ph_durs = batch['ph_durs'].float()
        
        # 构建 padding mask（与 dataset 一致：PAD ID=0）
        padding_mask = (ph == PH_PAD_ID)
        
        # 预测时长
        pred_durs = self(ph, notedurs, pitches, notetypes, padding_mask)   # (B, T)
        
        # 计算 Loss (使用 log1p 缩放真实时长，并且忽略 padding 区域)
        dur_loss = F.mse_loss(
            pred_durs.masked_select(~padding_mask),        
            torch.log1p(ph_durs).masked_select(~padding_mask),
            reduction='mean'
        )

        self.log("train/dur_loss", dur_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return dur_loss

    def validation_step(self, batch, batch_idx):
        pitches = batch['pitches'].long()
        notedurs = batch['notedurs'].float()
        notetypes = batch['notetypes'].long()
        ph = batch['ph'].long()
        ph_durs = batch['ph_durs'].float()
        
        padding_mask = (ph == PH_PAD_ID)
        pred_durs = self(ph, notedurs, pitches, notetypes, padding_mask)

        dur_loss = F.mse_loss(
            pred_durs.masked_select(~padding_mask),        
            torch.log1p(ph_durs).masked_select(~padding_mask),
            reduction='mean'                               
        )

        # 记录验证集 loss 用于 checkpoint 保存
        self.log("val/dur_loss", dur_loss, prog_bar=True, logger=True, on_epoch=True)
        return dur_loss

    def configure_optimizers(self):
        # 移除了双优化器 (因为没有 Discriminator 了)，现在只优化当前网络的参数
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate, 
            betas=(0.5, 0.9)
        )
        return optimizer

    @torch.no_grad()
    def infer_duration(self, ph, notedurs, pitches, notetypes):
        """
        提供给后续主模型生成（推理）时调用的便捷方法
        将对数尺度的时间转换回真实的帧数序列
        """
        padding_mask = (ph == PH_PAD_ID)
        log_durs = self(ph, notedurs, pitches, notetypes, padding_mask)
        # 逆运算: exp(x) - 1，并确保不能为负数，四舍五入为整数帧
        durs = torch.clamp(torch.round(torch.exp(log_durs) - 1), min=0.0).long()
        return durs