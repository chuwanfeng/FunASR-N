"""
Qwen3-ASR-0.6B 模型转换脚本 (使用官方 qwen_3_asr_helper)
将模型转换为 OpenVINO IR 格式，支持 Intel 硬件加速

使用方法:
    python convert_qwen3_ov.py

前置要求:
    pip install qwen-asr openvino>=2025.4.0 modelscope

转换后的模型将保存在: Qwen3-ASR-0.6B-OV/
"""

import os
import sys
import subprocess
from pathlib import Path

# ==================== 配置 ====================
# 模型缓存目录
MODEL_CACHE_DIR = r"D:\Scoop\qwen_models"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# 魔搭模型 ID
MODELSCOPE_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"

# 输出目录
OUTPUT_DIR = Path("Qwen3-ASR-0.6B-OV")


def check_dependencies():
    """检查并安装依赖"""
    print("=" * 60)
    print("📦 检查依赖")
    print("=" * 60)
    
    # 检查 qwen-asr
    try:
        import qwen_asr
        print(f"✅ qwen-asr 已安装 (版本: {qwen_asr.__version__ if hasattr(qwen_asr, '__version__') else 'unknown'})")
    except ImportError:
        print("❌ qwen-asr 未安装，正在安装...")
        subprocess.run([sys.executable, "-m", "pip", "install", "qwen-asr"], check=False)
        
    # 检查 openvino
    try:
        import openvino as ov
        print(f"✅ OpenVINO 版本: {ov.__version__}")
    except ImportError:
        print("❌ OpenVINO 未安装，正在安装...")
        subprocess.run([sys.executable, "-m", "pip", "install", "openvino>=2025.4.0"], check=False)
    
    # 检查 modelscope
    try:
        import modelscope
        print(f"✅ ModelScope 已安装")
    except ImportError:
        print("❌ ModelScope 未安装，正在安装...")
        subprocess.run([sys.executable, "-m", "pip", "install", "modelscope"], check=False)


def download_from_modelscope():
    """从魔搭下载模型"""
    print("=" * 60)
    print("📥 从魔搭 (ModelScope) 下载模型")
    print("=" * 60)
    
    print(f"\n📦 模型: {MODELSCOPE_MODEL_ID}")
    print(f"📁 缓存目录: {MODEL_CACHE_DIR}")
    
    try:
        from modelscope import snapshot_download
        
        print("\n⏳ 开始下载... (约 1.8GB，请耐心等待)")
        
        model_dir = snapshot_download(
            MODELSCOPE_MODEL_ID,
            cache_dir=MODEL_CACHE_DIR,
            revision="master"
        )
        
        print(f"\n✅ 下载完成!")
        print(f"📁 模型位置: {model_dir}")
        return model_dir
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return None


def find_model_dir():
    """查找已下载的模型目录"""
    possible_paths = [
        Path(MODEL_CACHE_DIR) / "Qwen" / "Qwen3-ASR-0.6B",
        Path(MODEL_CACHE_DIR) / "Qwen" / "Qwen3-ASR-0___6B",
        Path(MODEL_CACHE_DIR) / "models" / "Qwen" / "Qwen3-ASR-0.6B",
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_dir():
            safetensors = list(path.glob("*.safetensors"))
            if safetensors:
                print(f"✅ 找到已下载的模型: {path}")
                return str(path)
    
    return None


def convert_model():
    """使用 qwen_3_asr_helper 转换模型"""
    print("=" * 60)
    print("🚀 Qwen3-ASR 模型转换 (OpenVINO IR 格式)")
    print("=" * 60)
    
    # 检查依赖
    check_dependencies()
    
    # 查找或下载模型
    model_dir = find_model_dir()
    
    if model_dir is None:
        print("\n未找到本地模型，将从魔搭下载...")
        model_dir = download_from_modelscope()
        if model_dir is None:
            print("\n❌ 下载失败，请检查网络后重试")
            return False
    else:
        print(f"\n✅ 使用已存在的模型: {model_dir}")
    
    # 输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"📁 输出目录: {OUTPUT_DIR.absolute()}")
    
    # 尝试导入 qwen_3_asr_helper
    # 添加项目根目录到路径，使 asr 成为可导入的包
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    try:
        from asr.qwen_3_asr_helper import convert_qwen3_asr_model
        print("✅ 成功导入 asr.qwen_3_asr_helper")
    except ImportError as e:
        print(f"❌ 无法导入 asr.qwen_3_asr_helper: {e}")
        print("\n请确保:")
        print("  1. asr/__init__.py 存在 (已创建)")
        print("  2. qwen_3_asr_helper.py 在 asr/ 目录下")
        print("  3. 已安装 qwen-asr: pip install qwen-asr")
        return False
    
    print("\n" + "=" * 60)
    print("⚙️  开始转换... (可能需要 10-20 分钟)")
    print("=" * 60)
    
    try:
        convert_qwen3_asr_model(
            model_id=model_dir,
            output_dir=OUTPUT_DIR,
            quantization_config=None,
            use_local_dir=True,
        )
        
        print("\n" + "=" * 60)
        print("✅ 转换完成!")
        print("=" * 60)
        print(f"\n📁 模型位置: {OUTPUT_DIR.absolute()}")
        
        print("\n包含文件:")
        for f in OUTPUT_DIR.iterdir():
            if f.is_file():
                size = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name} ({size:.2f} MB)")
        
        print("\n💡 现在可以启动服务:")
        print("   python main.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    convert_model()
