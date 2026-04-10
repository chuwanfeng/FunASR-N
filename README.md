# FunASR-Paraformer 智能字幕生成器

> 基于阿里达摩院 FunASR Paraformer 的高准确率语音识别系统，专为 Windows下仅核显环境优化，支持视频/音频转字幕。

## ✨ 特性

- 🎯 **高准确率**：使用 Paraformer-zh 工业级中文模型，CER 低至 4-5%
- ⚡ **CPU 优化**：针对 i5-12400 6核12线程优化，1小时视频约5-6分钟处理完成
- 🎬 **多格式支持**：MP4、AVI、MOV、MKV、MP3、WAV、FLAC、M4A 等
- 📝 **SRT 字幕**：自动生成带时间轴的标准 SRT 字幕文件
- 🔧 **VAD 静音检测**：自动跳过静音段，提升处理速度
- 🌐 **Web 界面**：简洁美观的拖拽上传界面，即开即用
- 🔥 **热词定制**：支持自定义热词，提高专业术语识别率

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- Windows 10/11（已测试 Intel i5-12400）
- 8GB+ RAM（推荐 16GB）

### 2. 安装

```bash
# 克隆或进入项目目录
cd D:\python\FunASR-N

# 运行安装脚本（推荐）
setup.bat

# 或手动安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 启动服务

```bash
python main.py
```

服务启动后访问：http://127.0.0.1:8888

### 4. 使用

- 拖拽或点击上传视频/音频文件
- 可选：输入热词（逗号分隔）提高特定词汇识别率
- 选择输出格式（纯文本 / SRT / 两者）
- 点击"开始识别"
- 下载生成的 SRT 字幕文件

## 📊 性能测试（i5-12400 + 16GB RAM）

| 视频时长 | 处理时间 | 实时率 |
|---------|---------|--------|
| 10分钟  | ~1分钟  | 0.1x   |
| 30分钟  | ~3分钟  | 0.1x   |
| 1小时   | ~6分钟  | 0.09x  |

> 实际速度受音频复杂度影响，VAD 静音检测可提升 30-50% 处理速度

## 🔧 配置说明

编辑 `config.py` 可调整：

```python
# 模型选择
PARAFORMER_MODEL = "paraformer-zh"  # 或 "paraformer-zh-small"（更快）

# CPU 线程数（针对 12400 优化）
NUM_THREADS = 8

# VAD 静音检测
USE_VAD = True
VAD_MIN_SILENCE_DURATION = 0.5  # 最小静音时长(秒)

# 字幕参数
SRT_MAX_SEGMENT_DURATION = 8.0   # 最大片段时长
SRT_MIN_SEGMENT_DURATION = 1.0   # 最小片段时长
```

## 📁 项目结构

```
FunASR-N/
├── main.py                 # FastAPI 主入口
├── config.py               # 配置文件
├── requirements.txt        # 依赖清单
├── setup.bat              # Windows 安装脚本
├── asr/
│   ├── paraformer_engine.py   # Paraformer 推理引擎
│   ├── trainer.py             # 训练器（保留）
│   ├── inferencer.py          # 旧版推理器（保留）
│   └── evaluator.py           # 评估器
├── preprocess/             # 音频预处理
├── tools/                  # 工具函数
├── funasrnano_model/       # 旧版模型（保留）
├── ctc/                    # CTC 模块
└── templates/
    └── index.html          # Web 界面
```

## 🎯 API 接口

### POST /recognize

单文件识别

**请求**：`multipart/form-data`
- `file`: 音频/视频文件
- `return_srt`: 是否返回 SRT（可选）
- `hotwords`: 热词（可选，逗号分隔）

**响应**：
```json
{
  "code": 0,
  "text": "识别文本",
  "srt": "SRT 内容",
  "segments": [
    {"text": "...", "start": 1.23, "end": 3.45}
  ]
}
```

### POST /batch-recognize

批量识别

### POST /generate-srt

直接生成 SRT 文件并下载

### GET /health

健康检查

## 🐛 常见问题

### Q: 提示 "No module named 'funasr'"
A: 运行 `pip install funasr modelscope`

### Q: 视频处理失败
A: 安装 FFmpeg 并添加到 PATH
- 下载：https://ffmpeg.org/download.html
- 解压后添加 bin 目录到系统 PATH

### Q: 模型下载慢
A: 首次运行会自动下载模型（约 200MB），请耐心等待，或使用镜像：
```bash
set MODELSCOPE_CACHE=D:\modelscope_cache
```

### Q: 内存占用过高
A: 在 config.py 中减小 BATCH_SIZE 为 1

## 📝 更新日志

### v2.0.0 (2026-04-09)
- 🎉 重构为 Paraformer 引擎
- ✨ 新增 VAD 静音检测
- ✨ 新增 SRT 字幕生成
- 🎨 全新 Web 界面
- ⚡ CPU 多线程优化

### v1.0.0
- 原始 FunASR-Nano 实现

## 📄 许可证

MIT License

## 🙏 致谢

- [FunASR](https://github.com/alibaba-damo-academy/FunASR) - 阿里达摩院语音识别框架
- [Modelscope](https://modelscope.cn) - 模型社区

---

**Made with 🦐 by 小虾 & 叶子**
