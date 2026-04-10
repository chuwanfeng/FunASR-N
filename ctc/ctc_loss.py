import torch
import torch.nn as nn
from config import DEVICE

class CTCLoss(nn.Module):
    """CTC 损失函数"""
    def __init__(self, blank=0):
        super().__init__()
        self.blank = blank
        self.loss_func = nn.CTCLoss(blank=blank, zero_infinity=True)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        log_probs: [T, B, C]  模型输出
        targets: [B, T]  文本标签
        input_lengths: [B]  序列长度
        target_lengths: [B]  标签长度
        """
        return self.loss_func(
            log_probs, targets, input_lengths, target_lengths
        ).to(DEVICE)