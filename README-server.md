# 服务器环境

工作目录：`/hy-tmp/E`；项目解释器：`/hy-tmp/E/.venv/bin/python`。

2026-09-23 安装完成：Transformers 4.44.2、Captum 0.9.0、openSMILE 2.6.0、Accelerate 0.34.2，以及科学计算、绘图、MediaPipe 等配套依赖。均通过导入；GPU 矩阵运算、反向传播与优化器更新、随机小 BERT 前向、Captum IG、openSMILE `(96, 25)` 特征提取均通过。

服务器检查记录：`/hy-tmp/E/reports/python-environment-check.txt`，退出码 0；新增包锁定清单：`/hy-tmp/E/requirements-server.local.lock.txt`。该清单不包含继承的系统包，应与预装 CUDA 栈和本文件一起保留。

该环境使用 `--system-site-packages` 复用服务商预装的 PyTorch 2.4.0+cu121、torchvision 0.19.0+cu121、torchaudio 2.4.0+cu121，其余依赖安装到项目虚拟环境。它不是完全隔离环境，不能脱离服务商基础镜像直接复制到另一台机器。

```bash
cd /hy-tmp/E
source scripts/activate-server.sh
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

复查 Python 库、GPU、媒体命令和 MediaPipe 模型：

```bash
python scripts/check_environment.py --cuda-build cu121 --models
```

安装目标见 `requirements-server.in`，使用清华 PyPI 镜像。Transformers 固定为 4.44.2、Accelerate 为 0.34.2，以配合预装 PyTorch；不直接套用本地 Windows 的 CUDA 12.8 锁定清单。

初始化命令（相同服务商基础镜像）：

```bash
python -m venv --without-pip --system-site-packages /hy-tmp/E/.venv
/hy-tmp/E/.venv/bin/python -m pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple -r /hy-tmp/E/requirements-server.in
```

使用 `--without-pip` 是因为服务商镜像的 ensurepip 初始化失败；虚拟环境通过继承使用基础镜像的 pip 模块，包仍安装到项目环境。

## 额外工具与模型

| 工具/模型 | 服务器位置 |
|---|---|
| FFmpeg / ffprobe 4.4.2 | `/usr/bin/ffmpeg`、`/usr/bin/ffprobe` |
| OpenFace FeatureExtraction | `/hy-tmp/E/tools/OpenFace-linux/build/bin/FeatureExtraction` |
| OpenFace 专用 dlib 19.24 | `/hy-tmp/E/tools/dlib` |
| OpenFace 四个 CE-CLM 模型 | `/hy-tmp/E/models/openface-ceclm`，同时部署于 OpenFace 的源码和 build/bin 模型目录 |
| MFA 3.3.10、CPU Kaldi | `/hy-tmp/E/tools/mfa` |
| MFA 英文声学模型、词典 v3.0.0 | `/hy-tmp/E/models/mfa/english_us_arpa.zip`、`english_us_arpa.dict` |
| BERT base uncased | `/hy-tmp/E/models/bert-base-uncased` |
| Whisper base.en（第一问异常ASR诊断，不是正式文本标签） | `/hy-tmp/E/models/whisper-base.en`；修订号`911407f4214e0e1d82085af863093ec0b66f9cd6`，经`HF_ENDPOINT=https://hf-mirror.com`获取，10个文件SHA-256与官方源下载一致；权重不纳入GitHub/提交包 |
| MediaPipe 人脸模型 | `/hy-tmp/E/models/mediapipe/face_landmarker.task` |

`activate-server.sh` 定义 `FeatureExtraction`、`mfa` 函数，分别使用各自工具环境；运行 Python 时仍使用项目 `.venv`。调用 MFA 不会把主环境切换成 Conda。英文对齐命令示例：

```bash
mfa align /path/to/corpus \
  /hy-tmp/E/models/mfa/english_us_arpa.dict \
  /hy-tmp/E/models/mfa/english_us_arpa.zip \
  /path/to/new_output --num_jobs 4
```

语料需要同名音频与转写 `.lab`/`.txt`。工具安装成功不证明实际样本的对齐精度；须人工核查词边界、静音、缺失和失败样本。

OpenFace 命令示例（默认 CE-CLM 模型）：

```bash
FeatureExtraction -f /path/to/video.mp4 -out_dir /path/to/new_output -aus -pose -gaze -2Dfp
```

CLNF 和默认 CE-CLM 两种模式均在官方示例视频的 2 秒片段上通过测试：30/30 帧成功，470 列输出；只表示环境可用，不代表题目视频均能成功检测。MFA 合成语音测试也已通过，8 个词的内容、顺序和区间合法性检查通过。完整验收见 `reports/服务器工具验收.md`。

BERT 由用户授权选择 `google-bert/bert-base-uncased`，已固定 revision 并下载 safetensors、词表与配置；RTX 3090 Ti 上预训练模型前向输出 `(1, 14, 768)`，数值有限。用 `local_files_only=True, use_safetensors=True` 从上述目录加载即可。它与附件历史 tokenizer 的匹配仍待数据核验，不能当作已确认。资源来源与哈希见 `reports/模型来源.md`。

服务商提示：按量付费实例连续 24 小时未启动，`/hy-tmp` 可能清除。代码、依赖清单和训练结果需要备份。用户自行上传题目数据，本次工具安装未操作上传目录。

系统原始 `pip check` 有 `pygobject 3.42.1 requires pycairo` 提示；它属于服务商基础镜像已有缺项，不是本题模型依赖。FFmpeg/ffprobe 已补齐，MediaPipe 所需 EGL/GLES 运行库也已安装，模型加载和空白图推理通过。
