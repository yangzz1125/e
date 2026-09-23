# 环境安装与验证

使用 Windows x64、Python 3.11，当前项目目录为 `E:\MathModelWorkspaces\E`，项目虚拟环境为 `.venv`，不继承全局包。

## 基础依赖

在项目目录执行：

```powershell
uv pip install --python .\.venv\Scripts\python.exe --link-mode copy --default-index https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple -r requirements-base.lock.txt
.\.venv\Scripts\python.exe scripts\check_environment.py --base-only
.\.venv\Scripts\python.exe -m pip check
```

锁定清单包含分析、绘图、Excel 文件读取、openSMILE、MediaPipe、OpenCV、Transformers 等实际安装版本及传递依赖。不要同时安装多个提供 `cv2` 的包，本项目使用 MediaPipe 所需的 `opencv-contrib-python`。

## GPU 与解释工具

```powershell
uv pip install --python .\.venv\Scripts\python.exe --link-mode copy --index https://mirrors.nju.edu.cn/pytorch/whl/cu128 --default-index https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple -r requirements-gpu.txt
.\.venv\Scripts\python.exe scripts\check_environment.py
.\.venv\Scripts\python.exe -m pip check
```

本地 GPU 环境已安装并通过完整自检：CUDA 矩阵计算、随机小 BERT 前向、Captum IG、torchaudio 重采样、torchvision NMS、MediaPipe 模型加载及空白图推理。`pip check` 无冲突。完整版本见 `requirements-win-cu128.lock.txt`；accelerate/Captum 也已锁定版本。该锁定文件仅用于 Windows/Python 3.11/CUDA 12.8，不能直接套用于 Linux 服务器。

若大 wheel 流式下载反复中断，可执行 `python scripts/download_torch.py`。该脚本仅下载固定的 Windows CPython 3.11 / CUDA 12.8 / torch 2.11.0 安装包，按 8 MiB 分段、4 路并发保存，重跑可复用完整分段；合并后校验与官方索引一致的 SHA256。之后使用 uv 安装：

```powershell
.\.venv\Scripts\python.exe scripts\download_torch.py
uv pip install --python .\.venv\Scripts\python.exe --link-mode copy --index https://mirrors.nju.edu.cn/pytorch/whl/cu128 --default-index https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple .\.download-cache\nju-cu128\torch-2.11.0+cu128-cp311-cp311-win_amd64.whl -r requirements-gpu.txt
```

镜像来源：[南京大学 PyTorch 镜像](https://mirrors.nju.edu.cn/pytorch/whl/)、[清华 PyPI 镜像说明](https://mirror.tuna.tsinghua.edu.cn/help/pypi/)。国内镜像用于下载同一版本的软件；本次只在命令中指定源，没有修改系统全局配置。

## openSMILE 中文路径兼容

openSMILE 2.6.0 将配置选项路径编码为 ASCII。项目由 `E题` 改名为 `E` 后，直接初始化 `opensmile.Smile` 已通过测试。项目封装仍可使用，并在配置路径为英文时直接调用原生接口：

```python
from scripts.opensmile_compat import egemaps_lld
import soundfile as sf

audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
mono = audio.mean(axis=1)
with egemaps_lld() as extractor:
    features = extractor.process_signal(mono, sample_rate)
```

遇到中文配置路径时，封装将随包配置复制到临时英文路径，退出后自动清理；提取器必须在 `with` 内使用。若系统临时目录本身含中文，先将进程的 `TMP` 设置为已有的英文临时目录。

## 仍需准备

- MediaPipe FaceLandmarker 模型文件已下载并通过加载、空白图推理测试；真实视频效果仍待验证。来源与哈希见 `reports/模型来源.md`。运行 `python scripts/check_environment.py --models` 可包含模型加载自检。
- 本地 Windows 尚未配置预训练 BERT；服务器已选择并下载 BERT base uncased，tokenizer 与题目附件的匹配关系仍待核验。
- 本地 Windows 未部署强制对齐；服务器 MFA、OpenFace、FFmpeg 已安装并通过功能测试，详见 `README-server.md`。
- 尚未安装 OpenFace、librosa、SHAP 等备选工具；三模态 Shapley 可直接枚举联盟。

FFmpeg/ffprobe 使用系统已有程序；安装包缓存和 `.venv` 不纳入提交材料。原有 `reports/环境检查.md` 是全局环境的历史检查，不代表当前项目虚拟环境状态。
