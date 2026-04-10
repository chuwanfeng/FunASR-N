import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from config import DEVICE

class FunASRNanoModel(nn.Module):
    """FunASR-Nano 完整模型"""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        """前向传播"""
        feat = self.encoder(x)
        logits = self.decoder(feat)
        # 转 CTC 要求格式 [T, B, C]
        logits = logits.transpose(0, 1)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs