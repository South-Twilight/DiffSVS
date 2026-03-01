import torch
import torch.nn as nn
import torchaudio
import logging
import os
from huggingface_hub import hf_hub_download

# --- 日志配置 ---
logger = logging.getLogger("")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# ==========================================
# 官方工具函数 (集成自 ASLP-LAB 原版)
# ==========================================
class PadCrop(torch.nn.Module):
    """音频对齐工具：裁剪或填充音频到指定采样点数"""
    def __init__(self, n_samples, randomize=True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal):
        n, s = signal.shape
        start = 0 if (not self.randomize) else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, :min(s, self.n_samples)] = signal[:, start:end]
        return output

def set_audio_channels(audio, target_channels):
    """强制转换音频声道数"""
    if target_channels == 1:
        audio = audio.mean(1, keepdim=True)
    elif target_channels == 2:
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]
    return audio

# ==========================================
# 核心 VAE 封装类 (已完美适配 YAML Config)
# ==========================================
class DiffRhythmVAE(nn.Module): # 🌟 关键修改 1：继承 nn.Module 以防外层框架报错

    # 🌟 关键修改 2：增加 ckpt_path 参数，并用 **kwargs 吸收 YAML 中可能多余的参数
    def __init__(self, ckpt_path=None, device="cuda", repo_id="ASLP-lab/DiffRhythm-vae", **kwargs):
        super().__init__() # 必须初始化父类
        self.device = device
        self.sampling_rate = 44100
        self.downsampling_ratio = 2048
        self.latent_dim = 128      # 官方固定的总通道数 (包含了 mean 和 scale)
        self.io_channels = 2       # 官方 VAE 模型原生输入/输出的声道数
        self.target_channels = 1   # 你期望最终拿到的单声道结果
        
        try:
            # 🌟 关键修改 3：优先使用 YAML 传入的本地路径，没有才去下 HF
            if ckpt_path and os.path.exists(ckpt_path):
                logger.info(f"正在从配置路径加载 VAE 模型: {ckpt_path}")
                load_path = ckpt_path
            else:
                logger.info(f"未找到/未指定有效的 ckpt_path，尝试从 HuggingFace 本地缓存加载: {repo_id}")
                load_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="vae_model.pt",
                    local_files_only=True 
                )
            
            # 使用 torch.jit.load 加载 .pt 模型
            self.model = torch.jit.load(load_path, map_location="cpu").to(self.device).eval()
            logger.info("✅ VAE 模型加载成功。")
        except Exception as e:
            logger.error(f"❌ 加载模型失败: {e}")
            raise

    def preprocess_audio(self, audio, in_sr, target_length=None):
        audio = audio.to(self.device)

        if in_sr != self.sampling_rate:
            resample_tf = torchaudio.functional.Resample(in_sr, self.sampling_rate).to(self.device)
            audio = resample_tf(audio)
            
        if target_length is None:
            target_length = audio.shape[-1]
            
        audio = PadCrop(target_length, randomize=False)(audio)

        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)

        audio = set_audio_channels(audio, self.io_channels)
        return audio

    def vae_sample(self, mean, scale):
        stdev = torch.nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        latents = torch.randn_like(mean) * stdev + mean

        kl = (mean * mean + var - logvar - 1).sum(1).mean()
        return latents, kl

    @torch.no_grad()
    def encode(self, audio, chunked=True, overlap=32, chunk_size=128):
        samples_per_latent = self.downsampling_ratio
        
        remainder = audio.shape[2] % samples_per_latent
        if remainder != 0:
            pad_len = samples_per_latent - remainder
            audio = torch.nn.functional.pad(audio, (0, pad_len))
            
        total_size = audio.shape[2] 
        batch_size = audio.shape[0]
        chunk_size_samples = chunk_size * samples_per_latent 
        
        if not chunked or total_size <= chunk_size_samples:
            return self.model.encode_export(audio)
            
        overlap_samples = overlap * samples_per_latent 
        hop_size = chunk_size_samples - overlap_samples
        
        chunks = []
        i = 0  # 兜底变量
        
        for i in range(0, total_size - chunk_size_samples + 1, hop_size):
            chunk = audio[:, :, i:i+chunk_size_samples]
            chunks.append(chunk)
            
        if i + chunk_size_samples != total_size:
            chunk = audio[:, :, -chunk_size_samples:]
            chunks.append(chunk)
            
        chunks = torch.stack(chunks)
        num_chunks = chunks.shape[0]
        
        y_size = total_size // samples_per_latent
        y_final = torch.zeros((batch_size, self.latent_dim, y_size)).to(audio.device)
        
        for i in range(num_chunks):
            x_chunk = chunks[i, :]
            y_chunk = self.model.encode_export(x_chunk)
            
            if i == num_chunks - 1:
                t_end = y_size
                t_start = t_end - y_chunk.shape[2]
            else:
                t_start = i * hop_size // samples_per_latent
                t_end = t_start + chunk_size_samples // samples_per_latent
                
            ol = overlap_samples // samples_per_latent // 2
            chunk_start = 0
            chunk_end = y_chunk.shape[2]
            
            if i > 0:
                t_start += ol
                chunk_start += ol
            if i < num_chunks - 1:
                t_end -= ol
                chunk_end -= ol
                
            y_final[:, :, t_start:t_end] = y_chunk[:, :, chunk_start:chunk_end]
            
        return y_final

    @torch.no_grad()
    def decode(self, latents, chunked=True, overlap=32, chunk_size=128):
        total_size = latents.shape[2]
        batch_size = latents.shape[0]
        
        if not chunked or total_size <= chunk_size:
            y_final = self.model.decode_export(latents)
            return set_audio_channels(y_final, self.target_channels)
            
        hop_size = chunk_size - overlap
        
        chunks = []
        i = 0  # 兜底变量
        
        for i in range(0, total_size - chunk_size + 1, hop_size):
            chunk = latents[:, :, i : i + chunk_size]
            chunks.append(chunk)
            
        if i + chunk_size != total_size:
            chunk = latents[:, :, -chunk_size:]
            chunks.append(chunk)
            
        chunks = torch.stack(chunks)
        num_chunks = chunks.shape[0]
        samples_per_latent = self.downsampling_ratio
        
        y_size = total_size * samples_per_latent
        y_final = torch.zeros((batch_size, self.io_channels, y_size)).to(latents.device)
        
        for i in range(num_chunks):
            x_chunk = chunks[i, :]
            y_chunk = self.model.decode_export(x_chunk)
            
            if i == num_chunks - 1:
                t_end = y_size
                t_start = t_end - y_chunk.shape[2]
            else:
                t_start = i * hop_size * samples_per_latent
                t_end = t_start + chunk_size * samples_per_latent
                
            ol = (overlap // 2) * samples_per_latent
            chunk_start = 0
            chunk_end = y_chunk.shape[2]
            
            if i > 0:
                t_start += ol
                chunk_start += ol
            if i < num_chunks - 1:
                t_end -= ol
                chunk_end -= ol
                
            y_final[:, :, t_start:t_end] = y_chunk[:, :, chunk_start:chunk_end]

        y_final = set_audio_channels(y_final, self.target_channels)
        return y_final

##### 测试代码 #####

def normalize_audio(y, target_dbfs=-6.0):
    """音频分贝归一化"""
    max_amplitude = torch.max(torch.abs(y))
    if max_amplitude == 0:
        return y
    target_amplitude = 10.0 ** (target_dbfs / 20.0)
    scale_factor = target_amplitude / max_amplitude
    normalized_audio = y * scale_factor
    return normalized_audio

def run_vae_reconstruction_test(input_wav_path, output_wav_path="reconstructed_mono.wav"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🚀 使用设备: {device}")

    try:
        vae = DiffRhythmVAE(device=device)

        if not os.path.exists(input_wav_path):
            logger.error(f"❌ 找不到输入文件: {input_wav_path}")
            return

        logger.info(f"📂 正在读取音频: {input_wav_path}")
        waveform, sr = torchaudio.load(input_wav_path)

        # 1. 预处理 (官方逻辑)
        processed_audio = vae.preprocess_audio(waveform, sr)
        # 官方可选：归一化音量到 -6dBFS (模拟 get_reference_latent 的行为)
        processed_audio = normalize_audio(processed_audio, target_dbfs=-6.0)
        logger.info(f"📊 预处理完成。形状: {processed_audio.shape}, 采样率: 44100Hz")

        # 2. 编码 (拿到 [Batch, 128, Time] 的完整分布矩阵)
        logger.info("⚡ 正在编码至潜在空间 (Encoding)...")
        latents_raw = vae.encode(processed_audio, chunked=True)
        logger.info(f"✨ 提取完整特征分布。形状: {latents_raw.shape} (前64通道为mean, 后64为scale)")
        
        # 3. 采样 (DiT 模型实际读取时会执行这一步)
        mean, scale = latents_raw.chunk(2, dim=1)
        latents, kl_loss = vae.vae_sample(mean, scale)
        logger.info(f"🎲 重参数化采样完成。具体 Latent 形状: {latents.shape}, KL: {kl_loss.item():.4f}")

        # 4. 解码 (重构音频，并根据 target_channels 转回单声道)
        logger.info("🔊 正在从潜在空间解码回波形 (Decoding)...")
        reconstructed_audio = vae.decode(latents, chunked=True)
        logger.info(f"🎵 解码完成。最终音频形状: {reconstructed_audio.shape}")

        output_waveform = reconstructed_audio.cpu().squeeze(0)
        torchaudio.save(output_wav_path, output_waveform, 44100)
        
        logger.info(f"✅ 测试成功！重构音频已保存至: {output_wav_path}")

    except Exception as e:
        logger.exception(f"💥 测试过程中发生错误: {e}")

if __name__ == "__main__":
    TEST_PATH = "/data7/tyx/dataset/opencpop/segments/wavs/2001000001.wav"
    run_vae_reconstruction_test(TEST_PATH)
