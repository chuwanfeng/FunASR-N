# ============================================================
# FunASR-N 配置文件
# 支持 Paraformer / SenseVoice / Qwen3-ASR 三大引擎
# ============================================================

import torch
import multiprocessing
import os
from pathlib import Path

# ==================== 🎯 引擎与模型选择 ====================
# 【必选】选择要使用的 ASR 引擎
# 可选值:
#   "paraformer-zh"      → Paraformer 中文模型 (轻量、快速)
#   "paraformer-zh-small" → Paraformer 缩小版 (更轻量)
#   "sensevoice"          → SenseVoice 多语种模型 (支持中英日韩等)
#   "qwen"               → Qwen3-ASR-0.6B (最高准确率，需要先转换)
ASR_ENGINE = "paraformer-zh"

# 根据引擎自动推断设备类型（通常不需要修改）
DEVICE = "openvino"  # CPU / OpenVINO
OPENVINO_DEVICE = "GPU"  # OpenVINO 设备: CPU / GPU / AUTO

# ==================== 🔧 引擎详细配置 ====================

# ---------- Paraformer / SenseVoice 配置 ----------
# 模型源: "ms" = ModelScope (推荐，国内速度快), "hf" = HuggingFace
MODEL_HUB = "ms"

# 【仅 Paraformer】选择具体模型
#   "paraformer-zh"         → 工业级中文识别，CER ~4-5%，推荐
#   "paraformer-zh-small"  → 轻量版，适合低配机器
PARAFORMER_MODEL = "paraformer-zh"
PARAFORMER_REVISION = "v2.0.4"

# 【仅 SenseVoice】模型名称
SENSEVOICE_MODEL = "iic/SenseVoiceSmall"
SENSEVOICE_REVISION = "v2.0.4"

# 标点恢复模型 (Paraformer / SenseVoice)
PUNCTUATION_MODEL = "ct-punc"

# ---------- Qwen3-ASR 配置 ----------
# 【仅 Qwen】魔搭模型 ID (下载用)
QWEN_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
# 【仅 Qwen】OpenVINO 转换后的模型目录 (转换后自动生成)
QWEN_OV_MODEL_DIR = "Qwen3-ASR-0.6B-OV"
# Qwen 模型缓存目录
QWEN_MODEL_CACHE = r"D:\Scoop\qwen_models"

# ==================== ⚡ 性能配置 ====================

# CPU 线程数 (建议: 核心数 * 2/3，留余量给系统)
# i5-12400 (6核12线程) → 8
# i7-12700 (12核20线程) → 12
# i9 / Ryzen 9 → 16
CPU_COUNT = multiprocessing.cpu_count()
NUM_THREADS = max(4, int(CPU_COUNT * 0.6))
print(f"[Config] CPU: {CPU_COUNT}核, 分配线程数: {NUM_THREADS}")

# 批处理大小 (CPU 环境建议 1-4，显存充足可加大)
BATCH_SIZE = 2

# ==================== 🎵 音频处理 ====================
SAMPLE_RATE = 16000  # ASR 标准采样率
N_MELS = 80
WIN_LENGTH = 400
HOP_LENGTH = 160

# ==================== 🔇 VAD 静音检测 ====================
# 启用后可跳过静音段，提升长音频处理速度 (约 30-50%)
USE_VAD = False  # 开启 VAD，在静音处切分，避免切在句子中间
VAD_MIN_SILENCE_DURATION = 0.3  # 静音超此时长则切分(秒)，0.3s 对中文句间停顿更敏感
VAD_SPEED_UP = 1.0              # VAD 加速倍数

# ==================== ✏️ 后处理 ====================
ENABLE_PUNCTUATION = True  # 启用标点恢复

# 字幕片段时长控制
SRT_MAX_SEGMENT_DURATION = 8.0  # 最大时长(秒)
SRT_MIN_SEGMENT_DURATION = 1.0  # 最小时长(秒)

# ASR 内部分段时长 (超过此长度自动切分)
ASR_SEGMENT_DURATION = 15.0  # 中文建议 15s，日韩语建议 30s

# ==================== 🌐 服务配置 ====================
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8889

# ==================== 🔥 热词配置 ====================
# 在此添加自定义热词，可显著提升特定词汇识别率
HOTWORDS = [
    
]  # 例如: ["人工智能", "深度学习", "机器学习"]

# ==================== 📁 缓存目录 ====================
MODEL_CACHE_DIR = Path(os.environ.get("HOME", r"D:\Scoop")) / "funasr_models"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 兼容旧版 ====================
ENCODER_DIM = 512
DECODER_DIM = 512
NUM_CLASSES = 2000
DROPOUT = 0.1
EPOCHS = 20
LR = 1e-4


# ============================================================
# 🔍 引擎自动检测与信息汇总（无需修改）
# ============================================================

def _get_engine_display_name():
    """获取引擎显示名称"""
    names = {
        "paraformer-zh": "Paraformer-ZH (中文)",
        "paraformer-zh-small": "Paraformer-ZH-Small (轻量)",
        "sensevoice": "SenseVoice (多语种)",
        "qwen": "Qwen3-ASR-0.6B + OpenVINO",
    }
    return names.get(ASR_ENGINE, ASR_ENGINE)


def _check_qwen_available():
    """检测 Qwen 模型是否已转换"""
    if ASR_ENGINE != "qwen":
        return False
    ov_path = Path(QWEN_OV_MODEL_DIR) / "thinker"
    return ov_path.exists() and list(ov_path.glob("*.xml"))


ENGINE_DISPLAY_NAME = _get_engine_display_name()
QWEN_AVAILABLE = _check_qwen_available()
