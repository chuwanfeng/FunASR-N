"""
Paraformer ASR 推理引擎
基于 FunASR 官方实现，支持 CPU 多线程加速、VAD、时间戳对齐
"""

import os
import re
import gc
import time
import threading
import tempfile
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np

# FunASR 相关导入
try:
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    FUNASR_AVAILABLE = True
except ImportError:
    FUNASR_AVAILABLE = False
    print("警告: funasr 未安装，请运行: pip install funasr")

# 音频处理
import librosa
import soundfile as sf
from scipy.signal import resample

try:
    import ffmpeg

    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False

from config import (
    DEVICE,
    SAMPLE_RATE,
    NUM_THREADS,
    USE_VAD,
    VAD_MIN_SILENCE_DURATION,
    VAD_SPEED_UP,
    ENABLE_PUNCTUATION,
    PUNCTUATION_MODEL,
    ASR_ENGINE,
    PARAFORMER_MODEL,
    PARAFORMER_REVISION,
    SENSEVOICE_MODEL,
    SENSEVOICE_REVISION,
    MODEL_HUB,
    MODEL_CACHE_DIR,
    BATCH_SIZE,
    ASR_SEGMENT_DURATION,
)
from tools.utils import get_logger

logger = get_logger()


@dataclass
class RecognitionSegment:
    """识别片段
    
    字段说明:
        text: 原始识别文本（可能无标点，用于内部处理）
        start: 开始时间(秒)
        end: 结束时间(秒)
        text_with_punc: 标点恢复后的文本（用于显示给用户）
        confidence: 置信度
        timestamps: 原始词级时间戳 [[start_ms, end_ms], ...]
    """

    text: str
    start: float
    end: float
    text_with_punc: Optional[str] = None
    confidence: Optional[float] = None
    timestamps: Optional[List[List[int]]] = None


@dataclass
class WordTimestamp:
    """词级别时间戳"""
    word: str
    start_ms: int  # 毫秒
    end_ms: int    # 毫秒
    
    @property
    def start_sec(self) -> float:
        return self.start_ms / 1000.0
    
    @property
    def end_sec(self) -> float:
        return self.end_ms / 1000.0


def format_time(seconds: float) -> str:
    """将秒数格式化为 SRT 时间格式 HH:MM:SS,mmm"""
    total_millis = int(round(seconds * 1000))
    hours = total_millis // 3600000
    minutes = (total_millis % 3600000) // 60000
    secs = (total_millis % 60000) // 1000
    millis = total_millis % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class ParaformerEngine:
    """Paraformer ASR 推理引擎"""

    def __init__(self):
        self.model = None
        self.vad_model = None
        self.punc_model = None
        self._init_lock = threading.Lock()
        self._initialized = False

    def initialize(self):
        """初始化模型（懒加载）"""
        if self._initialized:
            return

        with self._init_lock:
            if self._initialized:
                return

            # 同步 PyTorch OpenMP 线程数，防止 CPU 占满
            torch.set_num_threads(NUM_THREADS)

            if not FUNASR_AVAILABLE:
                raise RuntimeError(
                    "FunASR 未安装。请运行: pip install funasr modelscope torchaudio"
                )

            # 根据配置选择模型
            if ASR_ENGINE == "sensevoice":
                _model_name = SENSEVOICE_MODEL
                _model_rev = SENSEVOICE_REVISION
                logger.info(f"正在加载 SenseVoice 模型: {_model_name}")
            else:
                _model_name = PARAFORMER_MODEL
                _model_rev = PARAFORMER_REVISION
                logger.info(f"正在加载 Paraformer 模型: {_model_name}")

            start_time = time.time()

            # 设置设备 - 自动适配 NVIDIA / Apple Silicon / CPU
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"  # Mac 专用
            else:
                device = "cpu"

            logger.info(f"使用设备: {device}")

            # 加载主模型
            self.model = AutoModel(
                model=_model_name,
                model_revision=_model_rev,
                hub=MODEL_HUB,
                device=device,
                trust_remote_code=False,
                disable_update=True,
                disable_log=True,
                batch_size=BATCH_SIZE,
                ncpu=NUM_THREADS,
            )

            # 可选：加载 VAD 模型
            if USE_VAD:
                try:
                    from funasr import AutoModel as VadModel

                    self.vad_model = VadModel(
                        model="fsmn-vad",
                        model_revision="v2.0.4",
                        device=device,
                        disable_log=True,
                    )
                    logger.info("VAD 模型加载成功")
                except Exception as e:
                    logger.warning(f"VAD 模型加载失败: {e}，将使用简单分段")
                    self.vad_model = None

            # 可选：加载标点恢复模型
            if ENABLE_PUNCTUATION:
                try:
                    from funasr import AutoModel as PuncModel

                    self.punc_model = PuncModel(
                        model=PUNCTUATION_MODEL,
                        model_revision="v2.0.4",
                        device=device,
                        disable_log=True,
                    )
                    logger.info("标点恢复模型加载成功")
                except Exception as e:
                    logger.warning(f"标点恢复模型加载失败: {e}")
                    self.punc_model = None

            load_time = time.time() - start_time
            logger.info(f"模型加载完成，耗时 {load_time:.2f}s")
            self._initialized = True

    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        加载音频文件，统一转为 16kHz 单声道
        支持各种格式（通过 ffmpeg 或 librosa）
        """
        # 尝试使用 ffmpeg（支持更多格式）
        if FFMPEG_AVAILABLE:
            try:
                out, _ = (
                    ffmpeg.input(audio_path)
                    .output(
                        "pipe:", format="wav", acodec="pcm_s16le", ac=1, ar=SAMPLE_RATE
                    )
                    .run(capture_stdout=True, capture_stderr=True, quiet=True)
                )
                import io

                audio, sr = sf.read(io.BytesIO(out))
                return audio.astype(np.float32), sr
            except Exception as e:
                logger.warning(f"ffmpeg 加载失败: {e}，尝试 librosa")

        # 回退到 librosa
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
            result = self.vad_model.generate(
                input=audio_np,
                chunk_size=int(
                    10000 / VAD_SPEED_UP
                ),  # 在这里用加速倍数调整 VAD 的 chunk_size，默认 10s，VAD_SPEED_UP=10 时变为 1s
                cache={},
                data_type="sound",
                sampling_rate=sample_rate,
            )

            if not result:
                logger.warning("VAD 未检测到有效语音")
                return []

            vad_segments = result[0].get("value", [])
            speech_segments = []
            for seg in vad_segments:
                if len(seg) >= 2:
                    start = round(float(seg[0]) / 1000.0, 3)  # 👈 除以1000！毫秒→秒
                    end = round(float(seg[1]) / 1000.0, 3)  # 👈 除以1000！
                    
                    # 过滤：时长至少 VAD_MIN_SILENCE_DURATION 秒（默认0.5s）
                    if end > start and (end - start) >= VAD_MIN_SILENCE_DURATION:
                        speech_segments.append((start, end))

            logger.info(f"VAD 检测到 {len(speech_segments)} 段有效语音")
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

    def _apply_punctuation(self, text: str) -> str:
        """应用标点恢复"""
        if self.punc_model is None or not text.strip():
            return text

        try:
            result = self.punc_model.generate(input=text)
            if result and len(result) > 0:
                text = result[0].get("text", text)
        except Exception as e:
            logger.warning(f"标点恢复失败: {e}")

        return text

    def transcribe_file(
        self,
        audio_path: str,
        return_timestamps: bool = True,
        hotwords: Optional[List[str]] = None,
    ) -> Union[str, List[RecognitionSegment]]:
        """
        转录音频文件

        Args:
            audio_path: 音频文件路径
            return_timestamps: 是否返回时间戳分段
            hotwords: 热词列表，提高特定词汇识别率

        Returns:
            if return_timestamps: List[RecognitionSegment]
            else: str (完整文本)
        """
        self.initialize()

        # 加载音频
        audio, sr = self._load_audio(audio_path)

        # 降噪已禁用（经测试降噪会降低识别准确率）
        # 如需启用，取消下方注释
        # try:
        #     import noisereduce as nr
        #     audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8)
        #     logger.info("音频降噪完成")
        # except Exception as e:
        #     logger.warning(f"降噪失败: {e}")

        total_duration = len(audio) / sr

        # 音量归一化（提升弱音部分）
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95

        # 获取 VAD 分段（优先在静音处切分，避免切在句子中间）
        if self.vad_model is not None and USE_VAD:
            vad_segments = self._apply_vad(audio, sr)
            logger.info(f"VAD 检测到 {len(vad_segments)} 个有效语音段")
            if vad_segments:
                segments = vad_segments
            else:
                # 无有效语音时，使用固定时长分段
                segment_duration = ASR_SEGMENT_DURATION
                segments = []
                start = 0.0
                while start < total_duration:
                    end = min(start + segment_duration, total_duration)
                    segments.append((start, end))
                    start = end
                logger.info(
                    f"未检测到有效语音，使用 {segment_duration} 秒分段，共 {len(segments)} 段"
                )
        else:
            # VAD 未启用或模型未加载，使用固定时长分段
            segment_duration = ASR_SEGMENT_DURATION
            segments = []
            start = 0.0
            while start < total_duration:
                end = min(start + segment_duration, total_duration)
                segments.append((start, end))
                start = end
            logger.info(f"使用 {segment_duration} 秒分段，共 {len(segments)} 段")

        # 逐段识别
        results = []
        processed_count = 0
        for seg_start, seg_end in segments:
            start_sample = int(seg_start * sr)
            end_sample = int(seg_end * sr)
            seg_audio = audio[start_sample:end_sample]

            if len(seg_audio) < sr * 0.3:  # 忽略短于 0.3s 的片段
                continue

            # 临时保存片段
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                sf.write(temp_path, seg_audio, sr)

            try:
                # 识别（所有引擎统一参数，差异在 config.py 的模型选择）
                res = self.model.generate(
                    input=temp_path,
                    batch_size=1,
                    hotwords=hotwords,
                    output_timestamp=True,
                )
                # logger.info(f"识别结果: {res}")

                if res and len(res) > 0:
                    original_text = res[0].get("text", "")
                    timestamp = res[0].get("timestamp", None)  # 提取时间戳

                    if original_text:
                        # 过滤 SenseVoice 特殊标记 <|xxx|>
                        original_text = re.sub(r"<\|[^|]*\|>", "", original_text)
                        original_text = original_text.strip()

                        if original_text:  # 过滤后还有内容才添加
                            # 如果有精确时间戳，按原始文本分词
                            if timestamp and len(timestamp) > 0:

                                final_text = self._apply_punctuation(original_text)
                                
                                results.append(
                                        RecognitionSegment(
                                            text=original_text,  # ← 原始文本
                                            start=seg_start,
                                            end=seg_end,
                                            text_with_punc=final_text,  # 标点恢复后的文本
                                            confidence=res[0].get("confidence", None),
                                            timestamps=timestamp,  # 原始时间戳
                                        )
                                )       
                            else:
                                # 没有时间戳，使用整个片段的时间
                                final_text = self._apply_punctuation(original_text)
                                results.append(
                                    RecognitionSegment(
                                        text=original_text,  # ← 原始文本
                                        start=seg_start,
                                        end=seg_end,
                                        text_with_punc=final_text,  # 标点恢复后的文本
                                        confidence=res[0].get("confidence", None),
                                        timestamps=None,
                                    )
                                )
            except Exception as e:
                logger.error(f"片段识别失败 [{seg_start:.2f}-{seg_end:.2f}]: {e}")
            finally:
                # 清理临时文件
                try:
                    os.unlink(temp_path)
                except OSError as e:
                    logger.warning(f"清理临时文件失败: {e}")
                processed_count += 1
                # 每处理50个片段主动触发一次GC，防止内存膨胀
                if processed_count % 50 == 0:
                    gc.collect()

        if return_timestamps:
            return results
        else:
            return " ".join([r.text for r in results])

    def transcribe_with_timeline(
        self, audio_path: str, hotwords: Optional[List[str]] = None
    ) -> List[RecognitionSegment]:
        """
        生成带时间轴的字幕片段
        优先使用模型返回的精确时间戳按句子分拆，如果没有则使用按标点智能拆分
        """
        # 获取识别结果（包含精确时间戳）
        segments = self.transcribe_file(
            audio_path, return_timestamps=True, hotwords=hotwords
        )

        if not segments:
            return []

        final_segments = []

        for seg in segments:
            # 如果有精确时间戳（词级别），基于时间戳按句子分拆
            if seg.timestamps is not None and len(seg.timestamps) > 0:
                # 将文本分词：优先按空格，如果没有空格则按字符（中文场景）
                timestamps = seg.timestamps

                # 检查文本中是否有空格
                if ' ' in seg.text:
                    words = seg.text.split()
                else:
                    # 中文场景：按字符拆分
                    words = list(seg.text)

                # 确保词数和时间戳数量一致
                if len(words) == len(timestamps):
                    # 使用 text_with_punc 中的标点符号来拆分
                    text_with_punc = getattr(seg, 'text_with_punc', None)

                    if text_with_punc is None or not text_with_punc:
                        # 没有标点恢复文本，输出整个文本作为一条字幕
                        sentence_text = seg.text  # 或 seg.text_with_punc
                        start_ms = timestamps[0][0]
                        end_ms = timestamps[-1][1]
                        final_segments.append(
                            RecognitionSegment(
                                text=sentence_text,
                                start=seg.start + start_ms / 1000.0,
                                end=seg.start + end_ms / 1000.0,
                                confidence=seg.confidence,
                                timestamps=timestamps,
                            )
                        )
                    else:
                        # 循环处理：每次找第一个标点符号，取出符号前的文本和时间戳
                        remaining_text = text_with_punc
                        word_idx = 0

                        while remaining_text:
                            # 查找第一个标点符号的位置
                            match = re.search(r'[。！？；，、,.]', remaining_text)

                            if not match:
                                # 没有标点符号了，剩余所有词作为一个片段
                                if word_idx < len(words):
                                    remaining_words = words[word_idx:]
                                    remaining_timestamps = timestamps[word_idx:]
                                    sentence_text = ''.join(remaining_words)
                                    start_ms = remaining_timestamps[0][0]
                                    end_ms = remaining_timestamps[-1][1]
                                    final_segments.append(
                                        RecognitionSegment(
                                            text=sentence_text,
                                            start=seg.start + start_ms / 1000.0,
                                            end=seg.start + end_ms / 1000.0,
                                            confidence=seg.confidence,
                                            timestamps=remaining_timestamps,
                                        )
                                    )
                                break

                            # 获取标点符号前的内容
                            punc_pos = match.start()
                            segment_text = remaining_text[:punc_pos]  # 标点前的文本

                            # 计算这个片段包含多少个词
                            segment_words = []
                            segment_timestamps = []
                            char_count = 0

                            while word_idx < len(words) and char_count < len(segment_text):
                                word = words[word_idx]
                                segment_words.append(word)
                                segment_timestamps.append(timestamps[word_idx])
                                char_count += len(word)
                                word_idx += 1

                            # 添加这个片段
                            if segment_words:
                                start_ms = segment_timestamps[0][0]
                                end_ms = segment_timestamps[-1][1]
                                final_segments.append(
                                    RecognitionSegment(
                                        text="".join(segment_words),
                                        start=seg.start + start_ms / 1000.0,
                                        end=seg.start + end_ms / 1000.0,
                                        confidence=seg.confidence,
                                        timestamps=segment_timestamps,
                                    )
                                )

                            # 移除已处理的部分（包括标点符号）
                            remaining_text = remaining_text[punc_pos + 1 :]
                else:
                    # 词数不匹配，使用整个片段
                    final_segments.append(seg)
                continue

            # 没有精确时间戳，使用按标点符号智能拆分（降级方案）
            text = seg.text
            start_time = seg.start
            end_time = seg.end
            duration = end_time - start_time

            # 按句号、感叹号、问号、分号、逗号、顿号拆分
            sentences = re.split(r"(?<=[。！？；，、])\s*", text)
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) > 1:
                # 有多个句子，按时间比例分配（保证不越界）
                total_chars = len(text)
                current_start = start_time
                remaining_duration = duration
                for i, sent in enumerate(sentences):
                    sent_chars = len(sent)
                    # 最后一句用完剩余时间，避免累加误差
                    if i == len(sentences) - 1:
                        sent_end = end_time
                    else:
                        sent_duration = max(1.5, (sent_chars / total_chars) * duration)
                        sent_duration = min(sent_duration, remaining_duration - 1.5 * (len(sentences) - i - 1))
                        sent_end = min(current_start + sent_duration, end_time)
                    final_segments.append(
                        RecognitionSegment(
                            text=sent,
                            start=current_start,
                            end=sent_end,
                            confidence=seg.confidence,
                            timestamps=None,
                        )
                    )
                    remaining_duration -= (sent_end - current_start)
                    current_start = sent_end
            else:
                # 没有标点符号，但如果时间太长（超过12秒），按固定时长切分
                if duration > 12.0:
                    num_parts = int(duration / 8.0) + 1
                    part_duration = duration / num_parts
                    chars_per_part = len(text) / num_parts
                    for i in range(num_parts):
                        part_start = start_time + i * part_duration
                        part_end = min(part_start + part_duration, end_time)
                        char_start = int(i * chars_per_part)
                        char_end = int(min((i + 1) * chars_per_part, len(text)))
                        part_text = text[char_start:char_end].strip()
                        if part_text:
                            final_segments.append(
                                RecognitionSegment(
                                    text=part_text,
                                    start=part_start,
                                    end=part_end,
                                    confidence=seg.confidence,
                                    timestamps=None,
                                )
                            )
                else:
                    final_segments.append(seg)

        return final_segments

    def generate_srt(
        self,
        audio_path: str,
        output_path: str = None,
        hotwords: Optional[List[str]] = None,
        return_segments: bool = False,
    ):
        """
        生成 SRT 字幕文件

        Args:
            audio_path: 音频文件路径
            output_path: 输出文件路径（可选）
            hotwords: 热词列表
            return_segments: 是否返回分段信息

        Returns:
            if return_segments: (srt_content, segments)
            else: srt_content
        """
        segments = self.transcribe_with_timeline(audio_path, hotwords=hotwords)

        if not segments:
            if return_segments:
                return "", []
            return ""

        lines = []
        segment_list = []
        for idx, seg in enumerate(segments, 1):
            lines.append(str(idx))
            lines.append(f"{format_time(seg.start)} --> {format_time(seg.end)}")
            lines.append(seg.text.strip())
            lines.append("")
            segment_list.append(
                {
                    "id": idx,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                    "text_with_punc": getattr(seg, 'text_with_punc', seg.text.strip()),
                }
            )

        content = "\n".join(lines)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

        if return_segments:
            return content, segment_list
        return content

    def is_available(self) -> bool:
        """检查引擎是否可用"""
        return FUNASR_AVAILABLE

    def get_engine_info(self) -> dict:
        """返回当前引擎信息"""
        return {
            "engine": ASR_ENGINE,
            "model": SENSEVOICE_MODEL if ASR_ENGINE == "sensevoice" else PARAFORMER_MODEL,
            "vad_enabled": USE_VAD,
            "punctuation_enabled": ENABLE_PUNCTUATION,
        }


# 全局单例（线程安全）
_engine_instance = None
_engine_lock = threading.Lock()


def get_engine() -> ParaformerEngine:
    """获取全局 Paraformer 引擎实例（线程安全）"""
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = ParaformerEngine()
    return _engine_instance
