"""
通用工具函数
"""
import logging
import sys
from datetime import datetime
from pathlib import Path


def get_logger(name: str = "FunASR") -> logging.Logger:
    """
    获取配置好的 logger
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 格式化
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        # 防止向上传递
        logger.propagate = False
    
    return logger


def ensure_dir(path: str) -> Path:
    """
    确保目录存在
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def timestamp() -> str:
    """
    返回当前时间戳字符串
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_filename(filename: str) -> str:
    """
    清理文件名，移除不安全字符
    """
    import re
    # 移除或替换不安全字符
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    # 限制长度
    if len(safe) > 200:
        name, ext = safe.rsplit('.', 1) if '.' in safe else (safe, '')
        safe = name[:190] + ('.' + ext if ext else '')
    return safe
