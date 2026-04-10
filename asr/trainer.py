import torch
from tqdm import tqdm
from tools.utils import get_logger, to_device
from ctc.ctc_loss import CTCLoss
from config import LR, EPOCHS

logger = get_logger()

class Trainer:
    """ASR 训练器"""
    def __init__(self, model, train_dataloader):
        self.model = to_device(model)
        self.train_dataloader = train_dataloader
        self.loss_fn = CTCLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for feats, targets, feat_lens, target_lens in tqdm(self.train_dataloader):
            feats, targets = to_device(feats), to_device(targets)
            log_probs = self.model(feats)
            loss = self.loss_fn(log_probs, targets, feat_lens, target_lens)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss

    def train(self):
        logger.info("开始训练 FunASR-Nano 模型...")
        for epoch in range(EPOCHS):
            loss = self.train_epoch()
            logger.info(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}")
        logger.info("训练完成！")