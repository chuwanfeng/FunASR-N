FunASR/                # 项目根目录（完整可运行包）
├── requirements.txt                # 项目依赖清单（一键安装）
├── main.py                        # 主入口文件（训练/推理一键启动）
├── config.py                      # 全局配置文件（参数统一管理）
├── README.md                      # 项目使用说明
# 核心整合模块（你要求的全部模块）
├── ctc/                           # CTC 损失、解码核心模块
│   ├── __init__.py
│   ├── ctc_loss.py                # CTC 损失函数
│   └── ctc_decoder.py             # CTC 贪婪/束搜索解码
├── tools/
│   ├── __init__.py
│   └── utils.py                   # 通用工具函数（日志/路径/张量处理）
├── funasrnano_model/              # FunASR-Nano 模型定义
│   ├── __init__.py
│   ├── encoder.py                 # 编码器
│   ├── decoder.py                 # 解码器
│   └── funasrnano.py              # 模型总入口
├── asr/                           # ASR 流程封装（训练/推理/评估）
│   ├── __init__.py
│   ├── trainer.py                 # 训练器
│   ├── inferencer.py              # 推理器
│   └── evaluator.py               # 评估器
└── preprocess/                    # 数据预处理模块
    ├── __init__.py
    ├── audio_processor.py         # 音频处理（加载/分帧/特征提取）
    └── text_processor.py          # 文本处理（编码/解码/字典）