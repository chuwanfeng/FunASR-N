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
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import threading

# 反馈存储路径
FEEDBACK_FILE = Path(__file__).parent / "feedback.jsonl"

# config.py 路径（从 feedback/ 目录向上两级到项目根目录）
CONFIG_PATH = Path(__file__).parent.parent / "config.py"
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
    2. 提取用户纠正后的正确词（corrected_text 中新增/修正的词）
    3. 按出现频率排序，取前 N 个
    
    改进：使用 difflib 精确找出 original→corrected 的变更词
    
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
    
    from collections import Counter
    import difflib
    
    word_counter = Counter()
    
    for entry in recognition_entries:
        # 使用 SequenceMatcher 精确找出新增/修改的文本块
        try:
            sm = difflib.SequenceMatcher(None, entry.original_text, entry.corrected_text)
            
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag in ('replace', 'insert'):
                    # 这部分是 corrected 中新增/替换的文本
                    new_text = entry.corrected_text[j1:j2]
                    # 清理标点，提取有效词
                    clean = re.sub(r'[^\u4e00-\u9fff\w]', '', new_text)
                    if len(clean) >= 2:
                        word_counter[clean] += 1
        except Exception:
            pass
    
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
    
    同时更新内存中的热词缓存，无需重启服务
    
    Args:
        hotwords: 热词列表
        
    Returns:
        是否更新成功
    """
    try:
        config_path = CONFIG_PATH
        
        # 读取现有配置
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 去重并过滤空值
        hotwords = [w.strip() for w in hotwords if w and w.strip()]
        if not hotwords:
            print("[Feedback] 无有效热词需要更新")
            return False
        
        # 构建新的 HOTWORDS 行（带注释说明更新时间）
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        hotwords_str = ", ".join([f'"{w}"' for w in hotwords])
        new_lines = f"# 自动提取的热词（更新时间: {timestamp}）\nHOTWORDS = [{hotwords_str}]\n"
        
        # 替换或追加 - 使用简单字符串替换确保可靠
        if "HOTWORDS" in content:
            # 找到 HOTWORDS 所在行，整行替换
            lines = content.split('\n')
            new_lines_list = []
            replaced = False
            skip_next = False
            for i, line in enumerate(lines):
                if skip_next:
                    skip_next = False
                    continue
                if 'HOTWORDS' in line and '=' in line:
                    # 替换当前行，如果前一行是注释也替换掉
                    if new_lines_list and '自动提取的热词' in new_lines_list[-1]:
                        new_lines_list.pop()
                    new_lines_list.append(new_lines.rstrip('\n'))
                    replaced = True
                else:
                    new_lines_list.append(line)
            
            if not replaced:
                # 没找到，追加到末尾
                new_lines_list.append(new_lines.rstrip('\n'))
            
            content = '\n'.join(new_lines_list)
        else:
            content += f"\n{new_lines}"
        
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        # 更新内存中的热词（无需重启）
        import importlib
        import config
        importlib.reload(config)
        
        # 验证更新是否生效
        updated_hotwords = getattr(config, 'HOTWORDS', None)
        if updated_hotwords == hotwords:
            print(f"[Feedback] 热词已更新并生效: {updated_hotwords}")
            return True
        else:
            print(f"[Feedback] 热词写入但内存未同步: 文件={hotwords}, 内存={updated_hotwords}")
            # 强制设置
            config.HOTWORDS = hotwords
            return True
            
    except Exception as e:
        print(f"[Feedback] 更新配置失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_audio_hash(audio_path: str) -> str:
    """计算音频文件 SHA256"""
    sha256 = hashlib.sha256()
    with open(audio_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]  # 取前16位足够唯一


def get_current_hotwords() -> List[str]:
    """获取当前内存中的热词列表"""
    import importlib
    import config
    importlib.reload(config)
    return config.HOTWORDS if config.HOTWORDS else []


def get_feedback_stats() -> dict:
    """
    获取反馈统计信息
    
    Returns:
        {
            'total_feedback': 总反馈数,
            'unique_audio': 不同音频数,
            'top_errors': 常见错误类型统计,
            'recent_feedback': 最近反馈列表
        }
    """
    entries = load_all_feedback()
    
    from collections import Counter
    
    error_types = Counter(e.error_type for e in entries)
    
    return {
        'total_feedback': len(entries),
        'unique_audio': len(set(e.audio_hash for e in entries)),
        'top_errors': dict(error_types.most_common(5)),
        'recent_feedback': [
            {
                'audio_hash': e.audio_hash[:8] + '...',
                'original': e.original_text[:30] + '...' if len(e.original_text) > 30 else e.original_text,
                'corrected': e.corrected_text[:30] + '...' if len(e.corrected_text) > 30 else e.corrected_text,
                'timestamp': e.timestamp
            }
            for e in entries[:10]
        ]
    }


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
