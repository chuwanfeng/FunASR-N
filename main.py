"""
语音转字幕服务 - 基于 FunASR Paraformer
支持视频/音频文件上传，生成 SRT 字幕
"""

import os

# Hugging Face国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"  # 超时10分钟

os.environ["MODELSCOPE_CACHE"] = r"D:\Scoop\funasr_models"
os.environ["FUNASR_CACHE_DIR"] = r"D:\Scoop\funasr_models"
os.environ["HF_HOME"] = r"D:\Scoop\funasr_models"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\Scoop\funasr_models"
os.environ["XDG_CACHE_HOME"] = r"D:\Scoop\funasr_models"
os.environ["HOME"] = r"D:\Scoop"

for p in [r"D:\Scoop\funasr_models"]:
    os.makedirs(p, exist_ok=True)

import tempfile
import shutil
from typing import List, Optional, Callable
from pathlib import Path

from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from config import (
    SERVER_HOST, SERVER_PORT, HOTWORDS,
    ASR_ENGINE, DEVICE, OPENVINO_DEVICE,
    PARAFORMER_MODEL, PARAFORMER_REVISION,
    SENSEVOICE_MODEL, SENSEVOICE_REVISION,
    QWEN_MODEL_ID, QWEN_OV_MODEL_DIR, QWEN_AVAILABLE,
    MODEL_HUB, MODEL_CACHE_DIR,
    ENABLE_PUNCTUATION, PUNCTUATION_MODEL,
    USE_VAD, VAD_MIN_SILENCE_DURATION, VAD_SPEED_UP,
    NUM_THREADS, BATCH_SIZE,
    SAMPLE_RATE, ASR_SEGMENT_DURATION,
    SRT_MAX_SEGMENT_DURATION, SRT_MIN_SEGMENT_DURATION,
    ENGINE_DISPLAY_NAME,
)
from tools.utils import get_logger
from feedback_store import (
    save_feedback, 
    load_all_feedback, 
    extract_hotwords,
    update_config_hotwords,
    FeedbackEntry,
    get_audio_hash,
)

logger = get_logger()

# 输出目录配置（持久化存储）
OUTPUT_DIR = Path(__file__).parent / "output" / "subtitles"

# 根据配置选择 ASR 引擎
_asr_engine_map = {
    "paraformer-zh": ("paraformer_engine", "get_engine", "Paraformer-ZH (中文)"),
    "paraformer-zh-small": ("paraformer_engine", "get_engine", "Paraformer-ZH-Small (轻量)"),
    "sensevoice": ("paraformer_engine", "get_engine", "SenseVoice (多语种)"),
    "qwen": ("qwen_openvino_engine", "get_qwen_engine", "Qwen3-ASR-0.6B + OpenVINO"),
}

_engine_key = ASR_ENGINE
if _engine_key not in _asr_engine_map:
    raise ValueError(f"[Config] 未知的 ASR_ENGINE: '{_engine_key}'，可选值: {list(_asr_engine_map.keys())}")

_module, _func_name, _engine_display = _asr_engine_map[_engine_key]
_engine_mod = __import__(f"asr.{_module}", fromlist=[_func_name])
asr_engine = getattr(_engine_mod, _func_name)()
logger.info(f"[引擎] 已加载: {_engine_display} | 设备: {DEVICE} | OpenVINO: {OPENVINO_DEVICE}")

# 初始化 FastAPI
app = FastAPI(
    title="语音转字幕服务",
    description=f"基于 {_engine_display} 的高准确率语音识别，支持视频/音频转字幕<br>"
                f"<b>当前引擎:</b> {_engine_display}<br>"
                f"<b>加速模式:</b> {DEVICE.upper()}",
    version="4.0.0",
)

# 确保临时目录存在
TEMP_DIR = Path(tempfile.gettempdir()) / "asr_subtitle"
TEMP_DIR.mkdir(exist_ok=True)

# 支持的文件扩展名
ALLOWED_AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
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
            .output(output_audio_path, acodec="pcm_s16le", ac=1, ar=16000)
            .run(overwrite_output=True, quiet=True)
        )
        return True
    except ImportError:
        # 如果没有 ffmpeg-python，尝试使用 subprocess
        import subprocess

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    video_path,
                    "-acodec",
                    "pcm_s16le",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-y",
                    output_audio_path,
                ],
                capture_output=True,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("ffmpeg 未找到，请安装 ffmpeg 并添加到 PATH")
            return False
    except Exception as e:
        logger.error(f"音频提取失败: {e}")
        return False


def process_file(
    file_path: str, original_filename: str, hotwords: Optional[List[str]] = None,
    progress_callback: Optional[callable] = None
) -> dict:
    """
    处理音频/视频文件，返回识别结果
    
    Args:
        file_path: 文件路径
        original_filename: 原始文件名
        hotwords: 热词列表
        progress_callback: 进度回调函数，接收 (current, total, message) 参数
    """
    result = {
        "success": False,
        "text": "",
        "segments": [],
        "srt_content": "",
        "error": None,
    }

    temp_audio_path = None
    try:
        # 确定需要处理的音频文件
        audio_path = file_path

        if is_video_file(original_filename):
            # 视频文件：提取音频
            temp_audio_path = str(TEMP_DIR / f"{Path(file_path).stem}_audio.wav")
            if not extract_audio_from_video(file_path, temp_audio_path):
                result["error"] = "视频音频提取失败，请确保已安装 ffmpeg"
                return result
            audio_path = temp_audio_path

        # 检查引擎是否可用
        if not asr_engine.is_available():
            result["error"] = (
                "ASR 引擎不可用，请安装依赖: pip install funasr modelscope"
            )
            return result

        # 初始化引擎
        asr_engine.initialize()

        # 获取音频总时长
        import soundfile as sf
        audio_info = sf.info(audio_path)
        total_duration = audio_info.duration
        logger.info(f"音频总时长: {total_duration:.2f} 秒")

        # 生成字幕（同时获取分段信息，避免重复识别）
        if progress_callback:
            progress_callback(0, 100, "正在识别...")
        
        srt_content, segment_list = asr_engine.generate_srt(
            audio_path, None, hotwords=hotwords, return_segments=True
        )
        
        if progress_callback:
            progress_callback(80, 100, "正在生成字幕...")

        # 持久化保存字幕文件
        if srt_content:
            original_name = Path(original_filename).stem
            safe_name = "".join(c for c in original_name if c.isalnum() or c in "._- ")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            srt_filename = f"{safe_name}_{timestamp}.srt"
            srt_filepath = OUTPUT_DIR / srt_filename
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(srt_filepath, "w", encoding="utf-8") as f:
                f.write(srt_content)
            result["srt_file"] = str(srt_filepath)
            logger.info(f"字幕已保存: {srt_filepath}")

        # 使用已经获取的分段结果
        # 根据引擎类型创建 RecognitionSegment 对象
        if ASR_ENGINE == "qwen":
            # Qwen 引擎返回的 segment_list 格式: list of dict with text, start, end
            from asr.qwen_openvino_engine import RecognitionSegment as QwenSegment
            segments = [
                QwenSegment(text=seg["text"], start=seg["start"], end=seg["end"])
                for seg in segment_list
            ]
        else:
            # Paraformer / SenseVoice 共用引擎
            from asr.paraformer_engine import RecognitionSegment as ParaformerSegment
            segments = [
                ParaformerSegment(text=seg["text"], start=seg["start"], end=seg["end"])
                for seg in segment_list
            ]

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
                "confidence": seg.confidence,
            }
            for seg in segments
        ]
        # 保存音频哈希，用于后续反馈关联
        result["audio_hash"] = get_audio_hash(audio_path)

        if progress_callback:
            progress_callback(100, 100, "处理完成")

    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        result["error"] = str(e)

    finally:
        # 清理临时音频文件
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except:
                pass

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
        return HTMLResponse(
            content="""
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
        """,
            status_code=200,
        )

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return JSONResponse(
        {
            "status": "ok",
            "engine": ASR_ENGINE,
            "engine_display": _engine_display,
            "engine_available": asr_engine.is_available(),
            "device": DEVICE,
            "openvino_device": OPENVINO_DEVICE,
            "model": PARAFORMER_MODEL if ASR_ENGINE.startswith("paraformer") else (SENSEVOICE_MODEL if ASR_ENGINE == "sensevoice" else QWEN_MODEL_ID),
            "qwen_ready": QWEN_AVAILABLE,
            "vad_enabled": USE_VAD,
            "punctuation_enabled": ENABLE_PUNCTUATION,
        }
    )


@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
    return_srt: bool = Form(False),
    hotwords: Optional[str] = Form(None),
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
        return JSONResponse(
            {"code": 1, "error": f"不支持的文件格式。支持: {', '.join(ALLOWED_EXT)}"}
        )

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
            return JSONResponse({"code": 1, "error": result["error"]})

        response_data = {
            "code": 0,
            "text": result["text"],
            "segments": result["segments"],
            "filename": file.filename,
            "audio_hash": result.get("audio_hash", ""),
        }

        if return_srt:
            response_data["srt"] = result["srt_content"]

        return JSONResponse(response_data)

    except Exception as e:
        logger.error(f"识别异常: {e}")
        return JSONResponse({"code": 1, "error": str(e)})
    finally:
        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()


@app.post("/batch-recognize")
async def batch_recognize(
    files: List[UploadFile] = File(...), return_srt: bool = Form(False)
):
    """
    批量文件识别接口 (带实时进度)
    使用 Server-Sent Events 推送进度
    """
    results = []
    total = len(files)

    async def progress_stream():
        """生成进度 SSE 流"""
        import json
        for i, file in enumerate(files):
            # 推送当前文件进度
            yield f"event: file_progress\ndata: {json.dumps({'current': i, 'total': total, 'filename': file.filename, 'status': 'processing'})}\n\n"
            
            temp_path = TEMP_DIR / f"batch_{os.urandom(4).hex()}_{file.filename}"
            try:
                content = await file.read()
                with open(temp_path, "wb") as f:
                    f.write(content)

                result = process_file(
                    str(temp_path), file.filename, HOTWORDS if HOTWORDS else None
                )

                results.append(
                    {
                        "filename": file.filename,
                        "success": result["success"],
                        "text": result.get("text", ""),
                        "srt": result.get("srt_content"),
                        "segments": result.get("segments", []),
                        "error": result.get("error"),
                    }
                )

            except Exception as e:
                results.append(
                    {"filename": file.filename, "success": False, "error": str(e)}
                )
            finally:
                if temp_path.exists():
                    temp_path.unlink()

            # 推送完成状态
            yield f"event: file_done\ndata: {json.dumps({'current': i + 1, 'total': total, 'filename': file.filename, 'status': 'done'})}\n\n"

        # 推送最终结果
        yield f"event: complete\ndata: {json.dumps({'code': 0, 'results': results})}\n\n"

    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        progress_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/generate-srt")
async def generate_srt_endpoint(
    file: UploadFile = File(...), hotwords: Optional[str] = Form(None)
):
    """
    生成 SRT 字幕文件并返回下载链接
    """
    # 验证文件
    if not any(file.filename.lower().endswith(ext) for ext in ALLOWED_EXT):
        return JSONResponse(
            {"code": 1, "error": f"不支持的文件格式。支持: {', '.join(ALLOWED_EXT)}"}
        )

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
            path=str(srt_path), media_type="text/plain", filename=srt_filename
        )

    except Exception as e:
        logger.error(f"SRT 生成失败: {e}")
        return JSONResponse({"code": 1, "error": str(e)})
    finally:
        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()


@app.post("/feedback")
async def submit_feedback(
    audio_hash: str = Form(...),
    segment_id: int = Form(...),
    original_text: str = Form(...),
    corrected_text: str = Form(...),
    start: float = Form(...),
    end: float = Form(...),
    error_type: str = Form("recognition"),
    confidence: Optional[float] = Form(None),
):
    """
    提交用户反馈（字幕修正）
    
    Args:
        audio_hash: 音频文件哈希
        segment_id: 分段 ID
        original_text: 原始识别文本
        corrected_text: 用户修正文本
        start: 开始时间（秒）
        end: 结束时间（秒）
        error_type: 错误类型（recognition|timestamp|omission|insertion）
        confidence: 原始置信度（可选）
        
    Returns:
        提交结果
    """
    try:
        entry = FeedbackEntry(
            audio_hash=audio_hash,
            segment_id=segment_id,
            original_text=original_text,
            corrected_text=corrected_text,
            start=start,
            end=end,
            error_type=error_type,
            timestamp=datetime.now().isoformat(),
            confidence=confidence,
        )
        
        if save_feedback(entry):
            return JSONResponse({
                "code": 0,
                "message": "反馈已保存",
                "data": {
                    "audio_hash": audio_hash,
                    "segment_id": segment_id,
                    "error_type": error_type,
                }
            })
        else:
            return JSONResponse({"code": 1, "error": "反馈保存失败"})
    except Exception as e:
        logger.error(f"反馈提交失败: {e}")
        return JSONResponse({"code": 1, "error": str(e)})


@app.get("/feedback")
async def list_feedback(
    audio_hash: Optional[str] = None,
    error_type: Optional[str] = None,
    limit: int = 100,
):
    """
    查询反馈记录
    
    Args:
        audio_hash: 按音频哈希筛选（可选）
        error_type: 按错误类型筛选（可选）
        limit: 最大返回条数
        
    Returns:
        反馈记录列表
    """
    try:
        entries = load_all_feedback()
        
        # 筛选
        if audio_hash:
            entries = [e for e in entries if e.audio_hash == audio_hash]
        if error_type:
            entries = [e for e in entries if e.error_type == error_type]
        
        # 限制数量
        entries = entries[:limit]
        
        return JSONResponse({
            "code": 0,
            "count": len(entries),
            "data": [
                {
                    "audio_hash": e.audio_hash,
                    "segment_id": e.segment_id,
                    "original_text": e.original_text,
                    "corrected_text": e.corrected_text,
                    "start": e.start,
                    "end": e.end,
                    "error_type": e.error_type,
                    "timestamp": e.timestamp,
                    "confidence": e.confidence,
                }
                for e in entries
            ]
        })
    except Exception as e:
        logger.error(f"反馈查询失败: {e}")
        return JSONResponse({"code": 1, "error": str(e)})


@app.post("/feedback/extract-hotwords")
async def extract_and_update_hotwords(
    min_frequency: int = Form(2),
    max_hotwords: int = 50,
    dry_run: bool = Form(True),
):
    """
    从反馈中提取热词并更新配置
    
    Args:
        min_frequency: 最小出现次数
        max_hotwords: 最大热词数量
        dry_run: 仅预览不更新配置
        
    Returns:
        提取的热词列表
    """
    try:
        hotwords = extract_hotwords(min_frequency, max_hotwords)
        
        result = {
            "code": 0,
            "count": len(hotwords),
            "hotwords": hotwords,
            "dry_run": dry_run,
        }
        
        if not dry_run:
            # 更新配置
            words_only = [w for w, _ in hotwords]
            if update_config_hotwords(words_only):
                result["updated"] = True
                result["message"] = "热词已更新到 config.py"
            else:
                result["updated"] = False
                result["message"] = "热词更新失败"
        
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"热词提取失败: {e}")
        return JSONResponse({"code": 1, "error": str(e)})


@app.get("/history")
async def list_subtitles():
    """列出已保存的字幕文件"""
    files = []
    for f in OUTPUT_DIR.glob("*.srt"):
        files.append(
            {
                "name": f.name,
                "path": str(f),
                "size": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
        )
    # 按修改时间倒序
    files.sort(key=lambda x: x["modified"], reverse=True)
    return JSONResponse({"code": 0, "files": files})


if __name__ == "__main__":
    logger.info(f"🚀 语音转字幕服务启动: http://{SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"📁 临时目录: {TEMP_DIR}")
    logger.info(f"🎯 ASR 引擎: " + PARAFORMER_MODEL)

    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")
