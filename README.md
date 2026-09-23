# E题建模工作区

本仓库跟踪代码、规划与进度文档。**不包含**赛题原始数据、模型权重、虚拟环境和生成的逐样本结果；这些材料留在授权的本地或服务器工作目录。比赛提交前仍须核对匿名性、许可和≤50MB附件限制。

## 当前进度

第一问：100条全片媒体预处理完成；100条编辑列表候选区间已记录；MFA初次100条对齐后，对6条无TextGrid样本做隔离束宽重试，带溯源的结果为**83条自动检查通过、17条待复核**。自动通过不等于人工确认词边界。完整视频的三模态特征、统一时序、100条特征验收**尚未完成**。细节见 [`code/problem1/README.md`](code/problem1/README.md) 和 [`reports/problem1-alignment/逐词对齐记录.md`](reports/problem1-alignment/逐词对齐记录.md)。

## 服务器协作

- Git工作副本：`/hy-tmp/E-collab`。在这里修改并提交代码/文档；不要直接把数据或模型加入Git。
- 现有Python环境与开源工具：`/hy-tmp/E/.venv`、`/hy-tmp/E/tools`、`/hy-tmp/E/models`。完整赛题输入：`/hy-tmp/mathmodel/E/input/`（只读）。中间结果：`/hy-tmp/E/problem1/`。工具版本与环境检查见 [`README-server.md`](README-server.md)。
- 运行第一问脚本时从协作副本调用代码，例如：

```bash
cd /hy-tmp/E-collab
/hy-tmp/E/.venv/bin/python code/problem1/preprocess_media.py \
  --input /hy-tmp/mathmodel/E/input --output /hy-tmp/E/problem1/media
/hy-tmp/E/.venv/bin/python code/problem1/map_transcripts.py \
  --media-dir /hy-tmp/E/problem1/media
```

MFA对齐及隔离重试命令见`code/problem1/README.md`；从本目录执行时用`/hy-tmp/E/.venv/bin/python`，不可省略绝对解释器路径。`/hy-tmp/E-collab`中的Git只同步代码与文档，不自动同步服务器中间结果。

多人协作时先`git pull --ff-only`、在各自分支改动，合并前复核报告与代码一致。服务器当前直接访问GitHub可能超时；网络未通或未配置认证时，勿把`git push`失败解释为工作丢失。重要代码应及时备份；`/hy-tmp`的数据保留期受服务商规则影响。
