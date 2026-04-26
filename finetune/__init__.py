"""
FunASR 模型微调模块

支持基于用户反馈数据的增量微调，提升特定领域识别准确率。
使用 FunASR 的 AutoModel.finetune() API。

流程：
1. 从 feedback.jsonl 加载修正数据
2. 转换为 FunASR 训练格式（音频路径 + 文本）
3. 调用 AutoModel.finetune() 进行微调
4. 保存微调后的模型
5. 热更新引擎使用新模型
"""

import json
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass

import torch

from feedback import load_all_feedback, FeedbackEntry
from tools.utils import get_logger

logger = get_logger()

# 微调配置
FINETUNE_DIR = Path(__file__).parent
FINETUNE_DIR.mkdir(exist_ok=True)

FINETUNE_DATA_DIR = FINETUNE_DIR / "data"
FINETUNE_DATA_DIR.mkdir(exist_ok=True)

FINETUNE_OUTPUT_DIR = FINETUNE_DIR / "output"
FINETUNE_OUTPUT_DIR.mkdir(exist_ok=True)

# 默认微调参数
DEFAULT_FINETUNE_CONFIG = {
    "batch_size": 4,
    "epochs": 3,
    "lr": 5e-5,
    "warmup_steps": 100,
    "save_steps": 100,
    "eval_steps": 50,
    "max_steps": 500,
    "gradient_accumulation_steps": 2,
    "fp16": True,
}


@dataclass
class FinetuneTask:
    """微调任务状态"""
    id: str
    status: str  # pending | running | completed | failed
    start_time: str
    end_time: Optional[str] = None
    progress: float = 0.0
    message: str = ""
    model_path: Optional[str] = None
    error: Optional[str] = None


class FinetuneManager:
    """微调任务管理器"""
    
    def __init__(self):
        self.tasks: Dict[str, FinetuneTask] = {}
        self.current_task: Optional[str] = None
        self._lock = threading.Lock()
        self._callback: Optional[Callable] = None
    
    def create_task(self) -> str:
        """创建新微调任务"""
        task_id = f"ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task = FinetuneTask(
            id=task_id,
            status="pending",
            start_time=datetime.now().isoformat(),
        )
        with self._lock:
            self.tasks[task_id] = task
            self.current_task = task_id
        return task_id
    
    def update_task(self, task_id: str, **kwargs):
        """更新任务状态"""
        with self._lock:
            if task_id in self.tasks:
                for key, value in kwargs.items():
                    setattr(self.tasks[task_id], key, value)
                if self._callback:
                    self._callback(self.tasks[task_id])
    
    def get_task(self, task_id: str) -> Optional[FinetuneTask]:
        """获取任务状态"""
        return self.tasks.get(task_id)
    
    def list_tasks(self, limit: int = 10) -> List[FinetuneTask]:
        """列出最近任务"""
        sorted_tasks = sorted(
            self.tasks.values(),
            key=lambda t: t.start_time,
            reverse=True
        )
        return sorted_tasks[:limit]
    
    def set_callback(self, callback: Callable):
        """设置状态更新回调"""
        self._callback = callback


# 全局管理器
finetune_manager = FinetuneManager()


def prepare_finetune_data(
    min_feedback_count: int = 10,
    max_samples: int = 1000
) -> Optional[Path]:
    """
    从反馈数据准备微调数据集
    
    需要音频文件路径，因此要求反馈时保留原始音频或音频哈希对应关系。
    当前实现：基于文本反馈生成伪训练数据（仅热词更新）。
    完整实现需要：保存用户上传的音频文件用于微调。
    
    Args:
        min_feedback_count: 最小反馈数量才触发微调
        max_samples: 最大样本数
        
    Returns:
        训练数据目录路径
    """
    entries = load_all_feedback()
    
    # 过滤有效反馈
    valid_entries = [
        e for e in entries
        if e.error_type == "recognition" 
        and e.original_text != e.corrected_text
    ]
    
    if len(valid_entries) < min_feedback_count:
        logger.info(f"反馈数量不足: {len(valid_entries)}/{min_feedback_count}")
        return None
    
    # 限制样本数
    valid_entries = valid_entries[:max_samples]
    
    # 创建训练数据目录
    data_dir = FINETUNE_DATA_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir.mkdir(exist_ok=True)
    
    # 生成训练清单
    # 注意：这里需要原始音频文件路径
    # 当前简化：只记录文本对，音频需要额外保存
    train_list = []
    for entry in valid_entries:
        # 查找对应的音频文件（需要预先保存）
        audio_path = find_audio_by_hash(entry.audio_hash)
        if audio_path and Path(audio_path).exists():
            train_list.append({
                "key": audio_path,
                "text": entry.corrected_text,
            })
    
    if len(train_list) < min_feedback_count // 2:
        logger.warning(f"有效音频样本不足: {len(train_list)}")
        return None
    
    # 保存训练清单
    train_file = data_dir / "train.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    logger.info(f"微调数据准备完成: {len(train_list)} 条样本 -> {data_dir}")
    return data_dir


def find_audio_by_hash(audio_hash: str) -> Optional[str]:
    """
    通过音频哈希查找原始音频文件路径
    
    需要在处理时保存 audio_hash -> file_path 的映射
    """
    mapping_file = FINETUNE_DIR / "audio_mapping.json"
    if not mapping_file.exists():
        return None
    
    try:
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        return mapping.get(audio_hash)
    except Exception:
        return None


def save_audio_mapping(audio_hash: str, audio_path: str):
    """保存音频哈希到路径的映射"""
    mapping_file = FINETUNE_DIR / "audio_mapping.json"
    mapping = {}
    if mapping_file.exists():
        try:
            with open(mapping_file, "r", encoding="utf-8") as f:
                mapping = json.load(f)
        except Exception:
            pass
    
    mapping[audio_hash] = str(audio_path)
    
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def run_finetune(
    data_dir: Path,
    base_model: str = "paraformer-zh",
    output_dir: Optional[Path] = None,
    config: Optional[Dict] = None
) -> Optional[str]:
    """
    执行模型微调
    
    Args:
        data_dir: 训练数据目录
        base_model: 基础模型名称
        output_dir: 输出目录
        config: 微调配置
        
    Returns:
        微调后模型路径
    """
    task_id = finetune_manager.create_task()
    
    try:
        finetune_manager.update_task(
            task_id,
            status="running",
            message="正在加载基础模型..."
        )
        
        # 导入 FunASR
        from funasr import AutoModel
        
        # 加载基础模型
        model = AutoModel(
            model=base_model,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        finetune_manager.update_task(
            task_id,
            progress=10,
            message="开始微调训练..."
        )
        
        # 设置输出目录
        if output_dir is None:
            output_dir = FINETUNE_OUTPUT_DIR / task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 合并配置
        train_config = DEFAULT_FINETUNE_CONFIG.copy()
        if config:
            train_config.update(config)
        
        # 执行微调
        # FunASR finetune API 示例（具体参数可能因版本而异）
        model.finetune(
            data=str(data_dir / "train.jsonl"),
            output_dir=str(output_dir),
            **train_config
        )
        
        finetune_manager.update_task(
            task_id,
            status="completed",
            progress=100,
            message="微调完成",
            model_path=str(output_dir),
            end_time=datetime.now().isoformat()
        )
        
        logger.info(f"微调完成: {output_dir}")
        return str(output_dir)
        
    except Exception as e:
        logger.error(f"微调失败: {e}", exc_info=True)
        finetune_manager.update_task(
            task_id,
            status="failed",
            message="微调失败",
            error=str(e),
            end_time=datetime.now().isoformat()
        )
        return None


def start_finetune_task(
    min_feedback_count: int = 10,
    base_model: str = "paraformer-zh",
    config: Optional[Dict] = None
) -> Optional[str]:
    """
    启动微调任务（异步）
    
    Returns:
        任务ID
    """
    # 准备数据
    data_dir = prepare_finetune_data(min_feedback_count)
    if data_dir is None:
        return None
    
    # 启动后台线程
    task_id = finetune_manager.create_task()
    
    def _run():
        run_finetune(data_dir, base_model, config=config)
    
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    
    return task_id


def get_finetune_status(task_id: str) -> Optional[Dict]:
    """获取微调任务状态"""
    task = finetune_manager.get_task(task_id)
    if task is None:
        return None
    return {
        "id": task.id,
        "status": task.status,
        "progress": task.progress,
        "message": task.message,
        "model_path": task.model_path,
        "error": task.error,
        "start_time": task.start_time,
        "end_time": task.end_time,
    }


def switch_to_finetuned_model(model_path: str) -> bool:
    """
    切换到微调后的模型
    
    修改 config.py 的模型路径并 reload
    """
    try:
        import importlib
        import config
        
        # 更新配置
        config_path = Path(__file__).parent.parent / "config.py"
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 添加或更新微调模型路径
        finetune_line = f'\n# 微调模型路径（自动更新）\nFINETUNED_MODEL_PATH = r"{model_path}"\n'
        
        if "FINETUNED_MODEL_PATH" in content:
            content = re.sub(
                r'# 微调模型路径.*?\nFINETUNED_MODEL_PATH = .*?\n',
                finetune_line,
                content,
                flags=re.DOTALL
            )
        else:
            content += finetune_line
        
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        # 刷新内存
        importlib.reload(config)
        
        logger.info(f"已切换到微调模型: {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"切换模型失败: {e}")
        return False


# 便捷函数：检查是否需要微调
def should_finetune(min_feedback_count: int = 50) -> bool:
    """检查反馈数量是否达到微调阈值"""
    entries = load_all_feedback()
    valid_count = len([
        e for e in entries
        if e.error_type == "recognition" and e.original_text != e.corrected_text
    ])
    return valid_count >= min_feedback_count


# 便捷函数：获取反馈统计
def get_finetune_readiness() -> Dict:
    """获取微调准备状态"""
    entries = load_all_feedback()
    valid_entries = [
        e for e in entries
        if e.error_type == "recognition" and e.original_text != e.corrected_text
    ]
    
    # 检查有多少条有音频文件
    audio_count = 0
    for entry in valid_entries:
        if find_audio_by_hash(entry.audio_hash):
            audio_count += 1
    
    return {
        "total_feedback": len(entries),
        "valid_feedback": len(valid_entries),
        "with_audio": audio_count,
        "ready_for_finetune": len(valid_entries) >= 10,
        "recommended": len(valid_entries) >= 50,
    }