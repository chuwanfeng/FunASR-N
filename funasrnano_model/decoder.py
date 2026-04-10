import torch
import torch.nn as nn
from config import DECODER_DIM, NUM_CLASSES, DROPOUT

class Decoder(nn.Module):
    """FunASR-Nano 解码器"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(DECODER_DIM, DECODER_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(DECODER_DIM, NUM_CLASSES)
        )

    def forward(self, x):
        # x: [B, T, ENCODER_DIM]
        x = self.linear(x)  # [B, T, NUM_CLASSES]
        return x