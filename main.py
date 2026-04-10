"""
语音转字幕服务 - 基于 FunASR Paraformer
支持视频/音频文件上传，生成 SRT 字幕
"""
import os

#Hugging Face国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"  # 超时10分钟

os.environ["MODELSCOPE_CACHE"] = r"D:\Scoop\funasr_models"
os.environ["FUNASR_CACHE_DIR"] = r"D:\Scoop\funasr_models"
os.environ["TRANSFORMERS_CACHE"] = r"D:\Scoop\funasr_models"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\Scoop\funasr_models"
os.environ["XDG_CACHE_HOME"] = r"D:\Scoop\funasr_models"
os.environ["HOME"] = r"D:\Scoop"

for p in [r"D:\Scoop\funasr_models"]:
    os.makedirs(p, exist_ok=True)

import tempfile
import shutil
from typing import List, Optional
from pathlib import Path

from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from config import SERVER_HOST, SERVER_PORT, HOTWORDS, PARAFORMER_MODEL
from asr.paraformer_engine import get_engine, RecognitionSegment
from tools.utils import get_logger

logger = get_logger()

# 输出目录配置（持久化存储）
OUTPUT_DIR = Path(__file__).parent / "output" / "subtitles"

# 初始化 FastAPI
app = FastAPI(
    title="语音转字幕服务",
    description="基于 FunASR Paraformer 的高准确率语音识别，支持视频/音频转字幕",
    version="2.0.0"
)

# 初始化 ASR 引擎
asr_engine = get_engine()

# 确保临时目录存在
TEMP_DIR = Path(tempfile.gettempdir()) / "asr_subtitle"
TEMP_DIR.mkdir(exist_ok=True)

# 支持的文件扩展名
ALLOWED_AUDIO_EXT = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}
ALLOWED_VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
ALLOWED_EXT = ALLOWED_AUDIO_EXT | ALLOWED_VIDEO_EXT


def is_audio_file(filename: str) -> bool:
    """判断是否为音频文件"""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_AUDIO_EXT


def is_video_file(filename: str) -> bool:
    """判断是否为视频文件"""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_VIDEO_EXT


def extract_audio_from_video(video_path: str, output_audio_path: str) -> bool:
    """
    从视频中提取音频
    使用 ffmpeg 或 librosa
    """
    try:
        import ffmpeg
        (
            ffmpeg.input(video_path)
            .output(output_audio_path, acodec='pcm_s16le', ac=1, ar=16000)
            .run(overwrite_output=True, quiet=True)
        )
        return True
    except ImportError:
        # 如果没有 ffmpeg-python，尝试使用 subprocess
        import subprocess
        try:
            subprocess.run(
                ['ffmpeg', '-i', video_path, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000',
                 '-y', output_audio_path],
                capture_output=True, check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("ffmpeg 未找到，请安装 ffmpeg 并添加到 PATH")
            return False
    except Exception as e:
        logger.error(f"音频提取失败: {e}")
        return False


def process_file(file_path: str, original_filename: str, hotwords: Optional[List[str]] = None) -> dict:
    """
    处理音频/视频文件，返回识别结果
    """
    result = {
        "success": False,
        "text": "",
        "segments": [],
        "srt_content": "",
        "error": None
    }
    
    try:
        # 确定需要处理的音频文件
        audio_path = file_path
        
        if is_video_file(original_filename):
            # 视频文件：提取音频
            audio_path = str(TEMP_DIR / f"{Path(file_path).stem}_audio.wav")
            if not extract_audio_from_video(file_path, audio_path):
                result["error"] = "视频音频提取失败，请确保已安装 ffmpeg"
                return result
        
        # 检查引擎是否可用
        if not asr_engine.is_available():
            result["error"] = "ASR 引擎不可用，请安装依赖: pip install funasr modelscope"
            return result
        
        # 初始化引擎
        asr_engine.initialize()
        
        # 生成字幕（同时获取分段信息，避免重复识别）
        srt_content, segment_list = asr_engine.generate_srt(
            audio_path, None, hotwords=hotwords, return_segments=True
        )
        
        # 持久化保存字幕文件
        if srt_content:
            original_name = Path(original_filename).stem
            safe_name = "".join(c for c in original_name if c.isalnum() or c in '._- ')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            srt_filename = f"{safe_name}_{timestamp}.srt"
            srt_filepath = OUTPUT_DIR / srt_filename
            with open(srt_filepath, "w", encoding="utf-8") as f:
                f.write(srt_content)
            result["srt_file"] = str(srt_filepath)
            logger.info(f"字幕已保存: {srt_filepath}")
        
        # 使用已经获取的分段结果
        from asr.paraformer_engine import RecognitionSegment
        segments = [RecognitionSegment(
            text=seg["text"],
            start=seg["start"],
            end=seg["end"]
        ) for seg in segment_list]
        
        # 获取完整文本
        full_text = " ".join([seg.text for seg in segments]) if segments else ""
        
        result["success"] = True
        result["text"] = full_text
        result["srt_content"] = srt_content
        result["segments"] = [
            {
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
                "confidence": seg.confidence
            }
            for seg in segments
        ]
        
        # 清理临时音频文件
        if audio_path != file_path and os.path.exists(audio_path):
            os.unlink(audio_path)
            
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        result["error"] = str(e)
    
    return result


@app.on_event("startup")
async def startup_event():
    """服务启动时预热模型"""
    logger.info("正在预热 ASR 引擎...")
    try:
        asr_engine.initialize()
        logger.info("ASR 引擎预热完成")
    except Exception as e:
        logger.warning(f"ASR 引擎预热失败: {e}")


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面"""
    html_path = Path(__file__).parent / "templates" / "index.html"
    if not html_path.exists():
        return HTMLResponse(content="""
        <html>
        <head><title>语音转字幕</title></head>
        <body>
            <h1>语音转字幕服务</h1>
            <p>模板文件不存在，请创建 templates/index.html</p>
            <hr>
            <h3>API 端点:</h3>
            <ul>
                <li>POST /recognize - 单文件识别</li>
                <li>POST /batch-recognize - 批量识别</li>
                <li>GET /health - 健康检查</li>
            </ul>
        </body>
        </html>
        """, status_code=200)
    
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return JSONResponse({
        "status": "ok",
        "engine_available": asr_engine.is_available(),
        "device": "cpu",
        "model": "paraformer-zh"
    })


@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
    return_srt: bool = Form(False),
    hotwords: Optional[str] = Form(None)
):
    """
    单文件识别接口
    
    Args:
        file: 上传的音频/视频文件
        return_srt: 是否返回 SRT 字幕内容
        hotwords: 热词（逗号分隔）
    
    Returns:
        JSON 格式的识别结果
    """
    # 验证文件扩展名
    if not any(file.filename.lower().endswith(ext) for ext in ALLOWED_EXT):
        return JSONResponse({
            "code": 1,
            "error": f"不支持的文件格式。支持: {', '.join(ALLOWED_EXT)}"
        })
    
    # 保存临时文件
    temp_path = TEMP_DIR / f"input_{os.urandom(8).hex()}_{file.filename}"
    # 输出目录配置（持久化存储）
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # 处理热词
        hotword_list = None
        if hotwords:
            hotword_list = [h.strip() for h in hotwords.split(",") if h.strip()]
        else:
            hotword_list = HOTWORDS if HOTWORDS else None
        
        # 识别
        result = process_file(str(temp_path), file.filename, hotword_list)
        
        if not result["success"]:
            return JSONResponse({
                "code": 1,
                "error": result["error"]
            })
        
        response_data = {
            "code": 0,
            "text": result["text"],
            "segments": result["segments"],
            "filename": file.filename
        }
        
        if return_srt:
            response_data["srt"] = result["srt_content"]
        
        return JSONResponse(response_data)
        
    except Exception as e:
        logger.error(f"识别异常: {e}")
        return JSONResponse({
            "code": 1,
            "error": str(e)
        })
    finally:
        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()


@app.post("/batch-recognize")
async def batch_recognize(
    files: List[UploadFile] = File(...),
    return_srt: bool = Form(False)
):
    """
    批量文件识别接口
    """
    results = []
    
    for file in files:
        temp_path = TEMP_DIR / f"batch_{os.urandom(4).hex()}_{file.filename}"
        try:
            content = await file.read()
            with open(temp_path, "wb") as f:
                f.write(content)
            
            result = process_file(str(temp_path), file.filename, HOTWORDS if HOTWORDS else None)
            
            results.append({
                "filename": file.filename,
                "success": result["success"],
                "text": result.get("text", ""),
                "error": result.get("error")
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    return JSONResponse({
        "code": 0,
        "results": results
    })


@app.post("/generate-srt")
async def generate_srt_endpoint(
    file: UploadFile = File(...),
    hotwords: Optional[str] = Form(None)
):
    """
    生成 SRT 字幕文件并返回下载链接
    """
    # 验证文件
    if not any(file.filename.lower().endswith(ext) for ext in ALLOWED_EXT):
        return JSONResponse({
            "code": 1,
            "error": f"不支持的文件格式。支持: {', '.join(ALLOWED_EXT)}"
        })
    
    temp_path = TEMP_DIR / f"srt_{os.urandom(8).hex()}_{file.filename}"
    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # 处理热词
        hotword_list = None
        if hotwords:
            hotword_list = [h.strip() for h in hotwords.split(",") if h.strip()]
        
        # 生成 SRT
        asr_engine.initialize()
        
        # 确定音频路径
        audio_path = str(temp_path)
        if is_video_file(file.filename):
            audio_path = str(TEMP_DIR / f"{temp_path.stem}_audio.wav")
            if not extract_audio_from_video(str(temp_path), audio_path):
                return JSONResponse({"code": 1, "error": "视频音频提取失败"})
        
        srt_filename = f"{Path(file.filename).stem}_subtitle.srt"
        srt_path = TEMP_DIR / srt_filename
        
        asr_engine.generate_srt(audio_path, str(srt_path), hotwords=hotword_list)
        
        if not srt_path.exists():
            return JSONResponse({"code": 1, "error": "字幕生成失败"})
        
        return FileResponse(
            path=str(srt_path),
            media_type="text/plain",
            filename=srt_filename
        )
        
    except Exception as e:
        logger.error(f"SRT 生成失败: {e}")
        return JSONResponse({"code": 1, "error": str(e)})
    finally:
        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()

@app.get("/history")
async def list_subtitles():
    """列出已保存的字幕文件"""
    files = []
    for f in OUTPUT_DIR.glob("*.srt"):
        files.append({
            "name": f.name,
            "path": str(f),
            "size": f.stat().st_size,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        })
    # 按修改时间倒序
    files.sort(key=lambda x: x["modified"], reverse=True)
    return JSONResponse({"code": 0, "files": files})


if __name__ == "__main__":
    logger.info(f"🚀 语音转字幕服务启动: http://{SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"📁 临时目录: {TEMP_DIR}")
    logger.info(f"🎯 ASR 引擎: " + PARAFORMER_MODEL)
    
    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info"
    )
