# FunASR-N 智能字幕生成器

> 基于阿里达摩院 FunASR 的高准确率语音识别系统，支持 Paraformer / SenseVoice / Qwen3-ASR 三大引擎，专为中文场景优化。

## ✨ 特性

- 🎯 **多引擎支持**：Paraformer-zh（中文）、SenseVoice（多语种）、Qwen3-ASR-0.6B（最高准确率）
- ⚡ **硬件加速**：支持 OpenVINO GPU 加速，CPU 多线程优化
- 🎬 **多格式支持**：MP4、AVI、MOV、MKV、MP3、WAV、FLAC、M4A 等
- 📝 **SRT 字幕**：自动生成带时间轴的标准 SRT 字幕文件
- 🔧 **SileroVAD 静音检测**：准确率更高，不易丢失短对话，提升处理速度
- 🌐 **Web 界面**：简洁美观的拖拽上传界面，支持字幕实时编辑
- 🔥 **热词定制**：支持自定义热词，从用户反馈自动提取热词
- 🧠 **模型微调**：基于用户反馈数据增量微调，越用越准确
- 📊 **反馈闭环**：用户修正 → 反馈存储 → 热词提取 → 模型微调

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- Windows 10/11 / Linux / macOS
- 8GB+ RAM（推荐 16GB）
- FFmpeg（视频处理需要）

### 2. 安装

```bash
# 进入项目目录
cd FunASR-N

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 启动服务

```bash
python main.py
```

服务启动后访问：http://127.0.0.1:8889

### 4. 使用

1. **语音识别**：拖拽或点击上传视频/音频文件，点击「开始识别」
2. **字幕编辑**：在「时间轴片段」标签页点击字幕进行编辑修正
3. **提交反馈**：修改后按 Ctrl+Enter 或点击「提交反馈」保存修正记录
4. **热词管理**：切换到「热词管理」页面，点击「提取并写入热词」
5. **模型微调**：积累足够反馈后，在「模型微调」页面启动训练

## 📁 项目结构

```
FunASR-N/
├── main.py                 # FastAPI 主入口
├── config.py               # 配置文件
├── requirements.txt        # 依赖清单
├── README.md              # 项目说明
│
├── asr/                    # ASR 引擎模块
│   ├── paraformer_engine.py   # Paraformer 推理引擎
│   ├── qwen_openvino_engine.py # Qwen3-ASR OpenVINO 引擎
│   ├── inferencer.py          # 推理工具
│   └── __init__.py
│
├── feedback/               # 用户反馈模块
│   ├── __init__.py           # 反馈存储、热词提取
│   └── feedback.jsonl        # 反馈数据文件
│
├── finetune/               # 模型微调模块
│   ├── __init__.py           # 微调任务管理
│   ├── audio_mapping.json    # 音频哈希映射
│   ├── data/                 # 训练数据
│   └── output/               # 微调输出
│
├── preprocess/             # 音频预处理
│   ├── audio_processor.py
│   └── text_processor.py
│
├── tools/                  # 工具函数
│   └── utils.py
│
├── templates/              # Web 前端页面
│   ├── index.html           # 语音识别页面
│   ├── hotwords.html        # 热词管理页面
│   └── finetune.html        # 模型微调页面
│
├── output/                 # 输出目录
│   ├── subtitles/           # 生成的字幕文件
│   └── audio_archive/       # 持久化音频（用于微调）
│
├── funasrnano_model/       # 旧版模型（保留）
├── ctc/                    # CTC 模块
└── docs/                   # 文档
```

## 🔧 配置说明

编辑 `config.py` 可调整：

```python
# 引擎选择
ASR_ENGINE = "paraformer-zh"  # paraformer-zh / paraformer-zh-small / sensevoice / qwen

# 设备配置
DEVICE = "openvino"           # cpu / openvino
OPENVINO_DEVICE = "GPU"       # CPU / GPU / AUTO

# 模型配置
PARAFORMER_MODEL = "paraformer-zh"
SENSEVOICE_MODEL = "iic/SenseVoiceSmall"
QWEN_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"

# 性能配置
NUM_THREADS = 8               # CPU 线程数
BATCH_SIZE = 2                # 批处理大小

# VAD 静音检测 (SileroVAD，准确率更高)
USE_VAD = True
VAD_MIN_SILENCE_DURATION = 0.5

# SileroVAD 配置
SILERO_VAD_THRESHOLD = 0.5
SILERO_VAD_MIN_SPEECH_DURATION = 0.25
SILERO_VAD_MIN_SILENCE_DURATION = 0.5
SILERO_VAD_SPEECH_PAD = 0.1

# 字幕参数
SRT_MAX_SEGMENT_DURATION = 8.0
SRT_MIN_SEGMENT_DURATION = 1.0

# 服务配置
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8889

# 热词配置
HOTWORDS = []                 # 自定义热词列表
```

## 🎯 API 接口

### 语音识别

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/recognize` | 单文件识别 |
| POST | `/batch-recognize` | 批量识别（SSE 进度流）|
| POST | `/generate-srt` | 生成 SRT 文件并下载 |

### 反馈管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/feedback` | 提交反馈 |
| GET | `/feedback` | 查询反馈记录 |
| GET | `/feedback/stats` | 反馈统计 |
| POST | `/feedback/extract-hotwords` | 提取热词（预览） |

### 热词管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/hotwords` | 获取当前热词 |
| POST | `/hotwords/extract` | 提取并写入热词 |

### 模型微调

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/finetune/status` | 微调准备状态 |
| POST | `/finetune/start` | 启动微调任务 |
| GET | `/finetune/task/{task_id}` | 查询任务状态 |
| GET | `/finetune/tasks` | 任务列表 |

### 其他

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/history` | 字幕文件列表 |

## 📊 性能参考

| 硬件 | 视频时长 | 处理时间 | 实时率 |
|------|---------|---------|--------|
| i5-12400 + 16GB | 10分钟 | ~1分钟 | 0.1x |
| i5-12400 + 16GB | 1小时 | ~6分钟 | 0.09x |

> 实际速度受音频复杂度影响，VAD 静音检测可提升 30-50% 处理速度

## 🐛 常见问题

### Q: 提示 "No module named 'funasr'"
A: 运行 `pip install funasr modelscope`

### Q: 提示 "No module named 'silero_vad'"
A: 运行 `pip install silero-vad`

### Q: 视频处理失败
A: 安装 FFmpeg 并添加到 PATH
- 下载：https://ffmpeg.org/download.html
- 解压后添加 bin 目录到系统 PATH

### Q: 模型下载慢
A: 首次运行会自动下载模型（约 200MB），使用国内镜像：
```bash
set MODELSCOPE_CACHE=D:\modelscope_cache
```

### Q: 内存占用过高
A: 在 config.py 中减小 BATCH_SIZE 为 1

### Q: 热词提取不生效
A: 确保有反馈数据（先提交一些修正），然后点击「提取并写入热词」

## 📝 更新日志

### v4.0.0 (2026-04-27)
- 🎉 新增三页面架构：语音识别 / 热词管理 / 模型微调
- ✨ 新增用户反馈闭环系统
- ✨ 新增热词自动提取与手动管理
- ✨ 新增模型微调任务管理
- 🎨 全新深色主题 Web 界面
- 🔧 代码结构重构，模块更清晰

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
- [ModelScope](https://modelscope.cn) - 模型社区
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) - 通义千问语音识别

---

**Made with 🦐 by 小虾 & 叶子**
