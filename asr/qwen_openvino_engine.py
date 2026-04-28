"""
Qwen3-ASR-0.6B 推理引擎 (基于官方 qwen_3_asr_helper + OpenVINO)
使用 OpenVINO 加速推理，支持 Intel 核显/CPU 优化

安装依赖:
    pip install openvino>=2025.4.0 qwen-asr

模型转换:
    python convert_qwen3_ov.py
"""

import os
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np

# 音频处理
import librosa

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False

from config import (
    DEVICE,
    OPENVINO_DEVICE,
    SAMPLE_RATE,
    USE_VAD,
    SILERO_VAD_THRESHOLD,
    SILERO_VAD_MIN_SPEECH_DURATION,
    SILERO_VAD_MIN_SILENCE_DURATION,
    SILERO_VAD_SPEECH_PAD,
    ASR_SEGMENT_DURATION,
    HOTWORDS,
)

from tools.utils import get_logger

logger = get_logger()


@dataclass
class RecognitionSegment:
    """识别片段"""
    text: str
    start: float
    end: float
    text_with_punc: Optional[str] = None
    confidence: Optional[float] = None
    language: Optional[str] = None


class QwenASREngine:
    """Qwen3-ASR-0.6B 推理引擎 (基于官方 qwen_3_asr_helper + OpenVINO)"""

    def __init__(self):
        self.model = None
        self._init_lock = threading.Lock()
        self._initialized = False
        self._ov_model_dir = None
        self.vad_model = None

        # 设备选择
        if DEVICE == "openvino":
            self.device = OPENVINO_DEVICE if OPENVINO_DEVICE in ["CPU", "GPU", "AUTO"] else "AUTO"
        else:
            self.device = "CPU"

        logger.info(f"🎯 Qwen3-ASR 引擎配置: device={self.device}")
    
    def initialize(self):
        """初始化引擎，加载 OpenVINO 模型"""
        if self._initialized:
            return

        with self._init_lock:
            if self._initialized:
                return

            logger.info("=" * 60)
            logger.info("🚀 初始化 Qwen3-ASR 引擎 (官方 qwen_3_asr_helper + OpenVINO)")
            logger.info("=" * 60)

            start_time = time.time()

            try:
                # 添加 asr 目录到路径
                import sys
                asr_dir = Path(__file__).parent
                if str(asr_dir) not in sys.path:
                    sys.path.insert(0, str(asr_dir))

                # 查找 OpenVINO 模型目录
                self._ov_model_dir = self._get_ov_model_dir()

                if self._ov_model_dir is None:
                    logger.warning("未找到 OpenVINO 模型，请先运行 convert_qwen3_ov.py 转换模型")
                else:
                    # 使用官方 OVQwen3ASRModel
                    from qwen_3_asr_helper import OVQwen3ASRModel

                    logger.info(f"正在加载 OpenVINO 模型: {self._ov_model_dir}")
                    logger.info(f"设备: {self.device}")

                    self.model = OVQwen3ASRModel.from_pretrained(
                        model_dir=str(self._ov_model_dir),
                        device=self.device,
                        max_inference_batch_size=32
                    )

                    logger.info(f"✅ OpenVINO 模型加载成功 (device={self.device})")
                    self._use_pytorch = False

            except ImportError as e:
                logger.error(f"无法导入 qwen_3_asr_helper: {e}")
                logger.info("请确保 qwen_3_asr_helper.py 在 asr/ 目录下")

            except Exception as e:
                logger.error(f"OpenVINO 模型加载失败: {e}")

             # 加载 SileroVAD 模型
            if USE_VAD:
                try:
                    from silero_vad import load_silero_vad
                    self.vad_model = load_silero_vad()
                    self.vad_model.eval()
                    logger.info("SileroVAD 模型加载成功")
                except Exception as e:
                    logger.warning(f"SileroVAD 加载失败: {e}，将使用简单分段")
                    self.vad_model = None


            init_time = time.time() - start_time
            logger.info(f"✅ 引擎初始化完成，耗时 {init_time:.2f} 秒")

            self._initialized = True

    def _get_ov_model_dir(self) -> Optional[Path]:
        """获取 OpenVINO 模型目录"""
        possible_paths = [
            Path("Qwen3-ASR-0.6B-OV"),
            Path("Qwen3-ASR-0.6B-OV").absolute(),
            Path(r"D:\vps\python\FunASR-N\Qwen3-ASR-0.6B-OV"),
            Path(r"D:\Scoop\qwen_models\Qwen3-ASR-0.6B-OV"),
        ]

        for path in possible_paths:
            if path.exists() and path.is_dir():
                # 检查 thinker 子目录中是否有 OpenVINO 模型文件
                thinker_path = path / "thinker"
                if thinker_path.exists():
                    ov_files = list(thinker_path.glob("*.xml"))
                    if ov_files:
                        logger.info(f"找到 OpenVINO 模型目录: {path} (包含 thinker/)")
                        return path
                # 也检查根目录
                ov_files = list(path.glob("*.xml"))
                if ov_files:
                    logger.info(f"找到 OpenVINO 模型: {path}")
                    return path

        return None

    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """加载音频文件，返回 16kHz mono 音频"""
        if FFMPEG_AVAILABLE:
            try:
                import soundfile as sf
                import io
                out, _ = (
                    ffmpeg.input(audio_path)
                    .output("pipe:", format="wav", acodec="pcm_s16le", ac=1, ar=SAMPLE_RATE)
                    .run(capture_stdout=True, capture_stderr=True, quiet=True)
                )
                audio, sr = sf.read(io.BytesIO(out))
                return audio.astype(np.float32), sr
            except Exception as e:
                logger.warning(f"FFmpeg 加载失败: {e}")

        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        return audio.astype(np.float32), sr

    def _apply_vad(self, audio_np: np.ndarray, sample_rate: int = 16000):
        """
            使用 VAD 模型对音频进行语音活动检测，返回有效语音片段的时间戳（秒）
        Args:
            audio_np: 音频 numpy 数组，形状为 (N,)，单声道、float32 / int16 均可
            sample_rate: 采样率，fsmn-vad 固定要求 16000
        Returns:
            speech_segments: list[list[float, float]]，如 [[0.5, 2.3], [3.1, 5.0]]
            无语音时返回空列表
        """
        try:
            audio_np = np.asarray(audio_np, dtype=np.float32)
            
            # SileroVAD 处理
            import torch
            from silero_vad import get_speech_timestamps
            
            audio_tensor = torch.from_numpy(audio_np)
            timestamps = get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                threshold=SILERO_VAD_THRESHOLD,
                sampling_rate=sample_rate,
                min_speech_duration_ms=int(SILERO_VAD_MIN_SPEECH_DURATION * 1000),
                min_silence_duration_ms=int(SILERO_VAD_MIN_SILENCE_DURATION * 1000),
                speech_pad_ms=int(SILERO_VAD_SPEECH_PAD * 1000),
            )
            
            speech_segments = []
            for ts in timestamps:
                start = round(ts['start'] / sample_rate, 3)
                end = round(ts['end'] / sample_rate, 3)
                if end > start and (end - start) >= SILERO_VAD_MIN_SPEECH_DURATION:
                    speech_segments.append((start, end))
            
            logger.info(f"SileroVAD 检测到 {len(speech_segments)} 段有效语音")
            return speech_segments

        except Exception as e:
            # 降级：每 ASR_SEGMENT_DURATION 秒一段
            segment_duration = ASR_SEGMENT_DURATION
            total_duration = len(audio_np) / sample_rate
            segments = []
            start = 0.0
            while start < total_duration:
                end = min(start + segment_duration, total_duration)
                segments.append((start, end))
                start = end
            logger.warning(
                f"VAD 处理失败: {e}，使用 {segment_duration} 秒分段，共 {len(segments)} 段"
            )
            return segments

    def transcribe_file(
        self,
        audio_path: str,
        return_timestamps: bool = True,
        hotwords: Optional[List[str]] = None,
    ) -> Union[str, List[RecognitionSegment]]:
        """转录音频文件"""
        self.initialize()

        audio, sr = self._load_audio(audio_path)
        total_duration = len(audio) / sr

        # VAD 分段
        if self.vad_model is not None:
            segments = self._apply_vad(audio, sr)
        else:
            segments = []
            start = 0.0
            while start < total_duration:
                end = min(start + ASR_SEGMENT_DURATION, total_duration)
                segments.append((start, end))
                start = end
            logger.info(f"使用 {ASR_SEGMENT_DURATION} 秒分段，共 {len(segments)} 段")
        #logger.info(f"分段结果: {segments}")

        results = []

        for i, (seg_start, seg_end) in enumerate(segments):
            start_sample = int(seg_start * sr)
            end_sample = int(seg_end * sr)
            seg_audio = audio[start_sample:end_sample]

            if len(seg_audio) < sr * 0.3:
                continue

            try:
                # OpenVINO 推理 - 需要保存临时文件
                import tempfile
                import soundfile as sf

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name
                    sf.write(temp_path, seg_audio, sr)

                try:
                    result = self.model.transcribe(
                        audio=temp_path,
                        language=None
                    )
                    text = result[0].text.strip()
                    language = getattr(result[0], 'language', 'auto')
                finally:
                    os.unlink(temp_path)

                #logger.info(f"识别结果 [{seg_start:.2f}-{seg_end:.2f}]: {result} (语言: {language})")

                if text:
                    import re
                    text = re.sub(r"<\|[^|]*\|>", "", text).strip()

                    if text:
                        results.append(RecognitionSegment(
                            text=text,
                            start=seg_start,
                            end=seg_end,
                            confidence=1.0,
                            language=language
                        ))

                if (i + 1) % 5 == 0:
                    logger.info(f"已处理 {i + 1}/{len(segments)} 段")

            except Exception as e:
                logger.error(f"片段识别失败 [{seg_start:.2f}-{seg_end:.2f}]: {e}")

        if return_timestamps:
            return results
        else:
            return " ".join([r.text for r in results])

    def transcribe_with_timeline(
        self, audio_path: str, hotwords: Optional[List[str]] = None
    ) -> List[RecognitionSegment]:
        """生成带时间轴的字幕片段"""
        return self.transcribe_file(audio_path, return_timestamps=True, hotwords=hotwords)

    def generate_srt(
        self,
        audio_path: str,
        output_path: str = None,
        hotwords: Optional[List[str]] = None,
        return_segments: bool = False,
    ):
        """生成 SRT 字幕文件"""
        segments = self.transcribe_with_timeline(audio_path, hotwords=hotwords)

        if not segments:
            if return_segments:
                return "", []
            return ""

        def format_time(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        lines = []
        segment_list = []
        for idx, seg in enumerate(segments, 1):
            lines.append(str(idx))
            lines.append(f"{format_time(seg.start)} --> {format_time(seg.end)}")
            lines.append(seg.text.strip())
            lines.append("")
            segment_list.append({
                "id": idx,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            })

        content = "\n".join(lines)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

        if return_segments:
            return content, segment_list
        return content

    def is_available(self) -> bool:
        """检查引擎是否可用"""
        return True


# 全局单例
_qwen_engine_instance = None


def get_qwen_engine() -> QwenASREngine:
    """获取全局 Qwen ASR 引擎实例"""
    global _qwen_engine_instance
    if _qwen_engine_instance is None:
        _qwen_engine_instance = QwenASREngine()
    return _qwen_engine_instance
