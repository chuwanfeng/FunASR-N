import torch
import torchaudio
import librosa
import numpy as np
from config import SAMPLE_RATE, N_MELS, WIN_LENGTH, HOP_LENGTH

class AudioProcessor:
    """音频预处理：加载 -> 重采样 -> 梅尔特征"""
    def __init__(self):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

    def load_audio(self, audio_path):
        """加载音频"""
        wav, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        return wav

    def get_mel_feature(self, wav):
        """提取梅尔频谱特征"""
        wav = torch.FloatTensor(wav)
        mel = self.mel_transform(wav)
        mel = torch.log(mel + 1e-8)  # 对数压缩
        mel = mel.transpose(0, 1)  # [T, 80]
        return mel