import torch
import numpy as np

class CTCDecoder:
    """CTC 贪婪解码 + 去重去空白符"""
    def __init__(self, blank=0):
        self.blank = blank

    def greedy_decode(self, log_probs):
        """贪婪解码"""
        # [T, B, C] -> [B, T]
        preds = torch.argmax(log_probs, dim=-1).transpose(0, 1).cpu().numpy()
        results = []
        for pred in preds:
            # 去重 + 去空白符
            uniq_pred = [p for i, p in enumerate(pred) if i == 0 or p != pred[i-1]]
            uniq_pred = [p for p in uniq_pred if p != self.blank]
            results.append(uniq_pred)
        return results