import torch
from tools.utils import to_device, get_logger
from ctc.ctc_decoder import CTCDecoder

logger = get_logger()

class Inferencer:
    def __init__(self, model, text_processor):
        self.model = to_device(model)
        self.model.eval()
        self.text_processor = text_processor
        self.decoder = CTCDecoder()

    @torch.no_grad()
    def infer(self, mel_feat):
        feat = to_device(mel_feat.unsqueeze(0))
        log_probs = self.model(feat)
        pred_ids = self.decoder.greedy_decode(log_probs)[0]
        text = self.text_processor.decode(pred_ids)
        text = text.replace("<space>", " ").strip()
        return text