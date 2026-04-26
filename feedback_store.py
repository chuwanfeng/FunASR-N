"""
用户反馈存储与热词提取

Phase 1: 反馈 API（后端）
- 接收前端提交的修正反馈
- 存储到 feedback.jsonl（简单文件存储，无需数据库）
- 提供查询接口供热词提取使用

Phase 2: 热词自动提取
- 定时扫描 feedback.jsonl
- 提取高频修正词对
- 自动更新 HOTWORDS 配置
"""

import json
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import threading

# 反馈存储路径
FEEDBACK_FILE = Path(__file__).parent / "feedback.jsonl"
FEEDBACK_LOCK = threading.Lock()


@dataclass
class FeedbackEntry:
    """单条反馈记录"""
    audio_hash: str           # 音频文件 SHA256
    segment_id: int           # 分段 ID
    original_text: str        # 原始识别文本
    corrected_text: str       # 用户修正文本
    start: float              # 开始时间（秒）
    end: float                # 结束时间（秒）
    error_type: str           # 错误类型: recognition|timestamp|omission|insertion
    timestamp: str            # 提交时间 ISO 格式
    confidence: Optional[float] = None  # 原始置信度（如有）


def _ensure_feedback_file():
    """确保反馈文件存在"""
    if not FEEDBACK_FILE.exists():
        FEEDBACK_FILE.touch()


def save_feedback(entry: FeedbackEntry) -> bool:
    """
    保存单条反馈到文件
    
    Args:
        entry: 反馈记录
        
    Returns:
        是否保存成功
    """
    try:
        _ensure_feedback_file()
        with FEEDBACK_LOCK:
            with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"[Feedback] 保存失败: {e}")
        return False


def load_all_feedback() -> List[FeedbackEntry]:
    """
    加载所有反馈记录
    
    Returns:
        反馈记录列表（按时间倒序）
    """
    entries = []
    if not FEEDBACK_FILE.exists():
        return entries
    
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entries.append(FeedbackEntry(**data))
                except (json.JSONDecodeError, TypeError):
                    continue
    except Exception as e:
        print(f"[Feedback] 加载失败: {e}")
    
    # 按时间倒序
    entries.sort(key=lambda x: x.timestamp, reverse=True)
    return entries


def extract_hotwords(
    min_frequency: int = 2,
    max_hotwords: int = 50
) -> List[Tuple[str, int]]:
    """
    从反馈中提取高频修正词对作为热词
    
    策略：
    1. 找出 original_text != corrected_text 的记录
    2. 提取 corrected_text 中的新增词汇（用户纠正后的正确词）
    3. 按出现频率排序，取前 N 个
    
    Args:
        min_frequency: 最小出现次数
        max_hotwords: 最大热词数量
        
    Returns:
        [(热词, 频率), ...]
    """
    entries = load_all_feedback()
    
    # 只取 recognition 类型的反馈
    recognition_entries = [
        e for e in entries 
        if e.error_type == "recognition" and e.original_text != e.corrected_text
    ]
    
    # 统计 corrected_text 中的词汇频率（简单分词：按非中文字符分割）
    from collections import Counter
    import re
    
    word_counter = Counter()
    
    for entry in recognition_entries:
        # 提取 corrected_text 中的中文字词
        words = re.findall(r'[\u4e00-\u9fff]+', entry.corrected_text)
        for word in words:
            if len(word) >= 2:  # 至少2个字符
                word_counter[word] += 1
    
    # 过滤低频词，取前 N
    hotwords = [
        (word, count) 
        for word, count in word_counter.most_common(max_hotwords)
        if count >= min_frequency
    ]
    
    return hotwords


def update_config_hotwords(hotwords: List[str]) -> bool:
    """
    将提取的热词更新到 config.py 的 HOTWORDS 配置
    
    Args:
        hotwords: 热词列表
        
    Returns:
        是否更新成功
    """
    try:
        config_path = Path(__file__).parent / "config.py"
        
        # 读取现有配置
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 构建新的 HOTWORDS 行
        hotwords_str = ", ".join([f'"{w}"' for w in hotwords])
        new_line = f"HOTWORDS = [{hotwords_str}]\n"
        
        # 替换或追加
        if "HOTWORDS = " in content:
            import re
            content = re.sub(r"HOTWORDS = \[.*?\]\n", new_line, content)
        else:
            content += f"\n# 自动提取的热词（来自用户反馈）\n{new_line}"
        
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        return True
    except Exception as e:
        print(f"[Feedback] 更新配置失败: {e}")
        return False


def get_audio_hash(audio_path: str) -> str:
    """计算音频文件 SHA256"""
    sha256 = hashlib.sha256()
    with open(audio_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]  # 取前16位足够唯一


# 便捷函数：从 RecognitionSegment 生成反馈
def create_feedback_from_segment(
    audio_path: str,
    segment_id: int,
    original_text: str,
    corrected_text: str,
    start: float,
    end: float,
    error_type: str = "recognition",
    confidence: Optional[float] = None
) -> FeedbackEntry:
    """从识别分段创建反馈记录"""
    return FeedbackEntry(
        audio_hash=get_audio_hash(audio_path),
        segment_id=segment_id,
        original_text=original_text,
        corrected_text=corrected_text,
        start=start,
        end=end,
        error_type=error_type,
        timestamp=datetime.now().isoformat(),
        confidence=confidence
    )
