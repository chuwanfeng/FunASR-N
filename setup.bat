@echo off
chcp 65001 > nul
echo ========================================
echo   FunASR-Paraformer 字幕生成器安装脚本
echo ========================================
echo.

echo [1/3] 检查 Python 环境...
python --version > nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)
python --version
echo.

echo [2/3] 安装 Python 依赖...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo 警告: 清华源安装失败，尝试默认源...
    pip install -r requirements.txt
)
echo.

echo [3/3] 检查 FFmpeg...
ffmpeg -version > nul 2>&1
if errorlevel 1 (
    echo 警告: 未找到 FFmpeg，视频处理将受影响
    echo 请访问 https://ffmpeg.org/download.html 下载并添加到 PATH
) else (
    echo FFmpeg 已安装
)
echo.

echo ========================================
echo   安装完成！
echo   启动命令: python main.py
echo   访问地址: http://127.0.0.1:8888
echo ========================================
pause
