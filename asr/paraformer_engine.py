"""
Paraformer ASR 推理引擎
基于 FunASR 官方实现，支持 CPU 多线程加速、VAD、时间戳对齐
"""
import os
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
    DEVICE, SAMPLE_RATE, NUM_THREADS, 
    USE_VAD, VAD_MIN_SILENCE_DURATION, VAD_SPEED_UP,
    ENABLE_PUNCTUATION, PUNCTUATION_MODEL,
    PARAFORMER_MODEL, MODEL_HUB, PARAFORMER_REVISION, BATCH_SIZE,
    ASR_SEGMENT_DURATION
)
from tools.utils import get_logger

logger = get_logger()


@dataclass
class RecognitionSegment:
    """识别片段"""
    text: str
    start: float  # 开始时间(秒)
    end: float    # 结束时间(秒)
    confidence: Optional[float] = None


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
                
            if not FUNASR_AVAILABLE:
                raise RuntimeError(
                    "FunASR 未安装。请运行: pip install funasr modelscope torchaudio"
                )
            
            logger.info(f"正在加载 Paraformer 模型: {PARAFORMER_MODEL}")
            start_time = time.time()
            
            # 设置设备
            device = "cpu"  # 强制使用 CPU（用户无独显）
            
            # 加载主模型
            self.model = AutoModel(
                model=PARAFORMER_MODEL,
                #model_revision=PARAFORMER_REVISION,
                hub=MODEL_HUB,
                device=device,
                #trust_remote_code=False,
                #disable_update=True,
                disable_log=True,  # 减少日志输出
                batch_size=BATCH_SIZE,
                ncpu=NUM_THREADS,  # 多线程
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
                    .output("pipe:", format="wav", acodec="pcm_s16le", ac=1, ar=SAMPLE_RATE)
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
    
    def _apply_vad(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """
        基于能量检测的简单 VAD（兼容 Windows）
        返回有效语音段的时间区间
        """
        try:
            # 计算能量（短时能量）
            frame_length = int(0.025 * sr)  # 25ms 帧长
            hop_length = int(0.010 * sr)    # 10ms 帧移
            
            # 分帧并计算每帧能量
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
            energy = np.sum(frames ** 2, axis=0)
            
            # 动态阈值（能量均值的 1/5）
            threshold = np.mean(energy) * 0.25
            
            # 标记语音帧（能量超过阈值）
            speech_frames = energy > threshold
            
            # 合并连续的语音帧为时间段
            segments = []
            in_speech = False
            start_frame = 0

            # 最小语音段长度（1.5秒）
            min_speech_frames = int(1.0 * sr / hop_length)
            # 最大静音间隔（1.0秒）内合并
            max_silence_frames = int(0.5 * sr / hop_length)
            
            silence_count = 0
            
            for i, is_speech in enumerate(speech_frames):
                if is_speech and not in_speech:
                    # 开始新的语音段
                    in_speech = True
                    start_frame = i
                    silence_count = 0
                elif not is_speech and in_speech:
                    silence_count += 1
                    # 如果静音超过阈值，结束当前语音段
                    if silence_count > max_silence_frames:
                        end_frame = i - silence_count
                        duration = (end_frame - start_frame) * hop_length / sr
                        if duration >= min_speech_frames * hop_length / sr:
                            start_time = start_frame * hop_length / sr
                            end_time = end_frame * hop_length / sr
                            segments.append((start_time, end_time))
                        in_speech = False
            
            # 处理最后一个未结束的语音段
            if in_speech:
                end_frame = len(speech_frames) - silence_count
                start_time = start_frame * hop_length / sr
                end_time = end_frame * hop_length / sr
                if end_time - start_time >= 0.3:
                    segments.append((start_time, end_time))
            
            # 如果没有检测到语音段，返回整个音频作为一个片段
            if not segments:
                segments = [(0.0, len(audio) / sr)]
            
            logger.info(f"VAD 检测到 {len(segments)} 个语音段")
            return segments
            
        except Exception as e:
            logger.warning(f"VAD 处理失败: {e}，使用简单分段")
            # 降级：每 30 秒一段
            segment_duration = ASR_SEGMENT_DURATION
            total_duration = len(audio) / sr
            segments = []
            start = 0.0
            while start < total_duration:
                end = min(start + segment_duration, total_duration)
                segments.append((start, end))
                start = end
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
        hotwords: Optional[List[str]] = None
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
        
        # 获取 VAD 分段
        #segments = self._apply_vad(audio, sr)
        #logger.info(f"检测到 {len(segments)} 个有效语音段")

        # 使用配置的分段时长（避免内存爆炸）
        from config import ASR_SEGMENT_DURATION
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
        for seg_start, seg_end in segments:
            # 提取片段音频
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
                # 识别
                res = self.model.generate(
                    input=temp_path,
                    language="auto",  # 自动识别中/英/日/韩
                    batch_size=1,
                    hotwords=hotwords,
                    output_timestamp=True,  # 启用时间戳
                )
                
                if res and len(res) > 0:
                    text = res[0].get("text", "")
                    if text:
                        # 应用标点恢复
                        text = self._apply_punctuation(text)
                        results.append(RecognitionSegment(
                            text=text.strip(),
                            start=seg_start,
                            end=seg_end,
                            confidence=res[0].get("confidence", None)
                        ))
            except Exception as e:
                logger.error(f"片段识别失败 [{seg_start:.2f}-{seg_end:.2f}]: {e}")
            finally:
                # 清理临时文件
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        if return_timestamps:
            return results
        else:
            return " ".join([r.text for r in results])
    
    def transcribe_with_timeline(
        self, 
        audio_path: str,
        hotwords: Optional[List[str]] = None
    ) -> List[RecognitionSegment]:
        """
        生成带时间轴的字幕片段
        使用30秒分段 + 按标点智能拆分
        """
        # 获取识别结果（60秒一段）
        segments = self.transcribe_file(audio_path, return_timestamps=True, hotwords=hotwords)
        
        if not segments:
            return []
        
        final_segments = []
        
        for seg in segments:
            text = seg.text
            start_time = seg.start
            end_time = seg.end
            duration = end_time - start_time
            
            # 按标点符号拆分句子
            import re
            # 按句号、感叹号、问号、分号、逗号、顿号拆分
            sentences = re.split(r'(?<=[。！？；，、])\s*', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) > 1:
                # 有多个句子，按时间比例分配
                total_chars = len(text)
                current_start = start_time
                for sent in sentences:
                    sent_chars = len(sent)
                    sent_duration = max(1.5, (sent_chars / total_chars) * duration)
                    sent_end = min(current_start + sent_duration, end_time)
                    final_segments.append(RecognitionSegment(
                        text=sent,
                        start=current_start,
                        end=sent_end,
                        confidence=seg.confidence
                    ))
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
                        char_end = int(min((i+1) * chars_per_part, len(text)))
                        part_text = text[char_start:char_end].strip()
                        if part_text:
                            final_segments.append(RecognitionSegment(
                                text=part_text,
                                start=part_start,
                                end=part_end,
                                confidence=seg.confidence
                            ))
                else:
                    final_segments.append(seg)
        
        return final_segments

    
    def generate_srt(self, audio_path: str, output_path: str = None, hotwords: Optional[List[str]] = None, return_segments: bool = False):
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
                "text": seg.text.strip()
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
        return FUNASR_AVAILABLE


# 全局单例
_engine_instance = None

def get_engine() -> ParaformerEngine:
    """获取全局 Paraformer 引擎实例"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ParaformerEngine()
    return _engine_instance
