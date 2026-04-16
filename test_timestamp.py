"""
测试时间戳功能的脚本
"""
import sys
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from asr.paraformer_engine import ParaformerEngine, RecognitionSegment

def test_with_audio(audio_path):
    """测试音频文件"""
    print(f"测试音频: {audio_path}")
    
    # 初始化配置
    config = Config(
        model_type="paraformer-zh",
        device="auto",
        use_vad=True,
        sample_rate=16000
    )
    
    # 初始化引擎
    print("加载模型中...")
    engine = ParaformerEngine(config)
    
    # 执行识别
    print("开始识别...")
    results = engine.transcribe_file(audio_path, return_timestamps=True)
    
    print(f"\n识别到 {len(results)} 个片段:")
    for i, seg in enumerate(results, 1):
        print(f"  [{i}] {seg.start:.2f}s -> {seg.end:.2f}s: {seg.text}")
        if seg.timestamps:
            print(f"      时间戳数量: {len(seg.timestamps)}")
    
    # 生成SRT字幕
    print("\n生成SRT字幕...")
    srt_content = engine.generate_srt(audio_path, return_segments=False)
    print(srt_content[:500] + "..." if len(srt_content) > 500 else srt_content)
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python test_timestamp.py <音频文件路径>")
        print("示例: python test_timestamp.py test.wav")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    if not Path(audio_path).exists():
        print(f"文件不存在: {audio_path}")
        sys.exit(1)
    
    test_with_audio(audio_path)
