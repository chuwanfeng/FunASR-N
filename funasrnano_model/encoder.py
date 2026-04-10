import torch
import torch.nn as nn
from config import ENCODER_DIM, DROPOUT

class Encoder(nn.Module):
    """FunASR-Nano 编码器"""
    def __init__(self, input_dim=80):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=64 * (input_dim // 4),
            hidden_size=ENCODER_DIM // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=DROPOUT
        )

    def forward(self, x):
        # x: [B, T, 80]
        x = x.unsqueeze(1)  # [B, 1, T, 80]
        x = self.conv(x)  # [B, 64, T//4, 20]
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, -1)  # [B, T, C*F]
        x, _ = self.lstm(x)  # [B, T, ENCODER_DIM]
        return x