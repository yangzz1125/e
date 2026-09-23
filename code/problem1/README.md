# 问题一代码与运行状态

本目录只放准备作为**第一问正式入口**的代码。已实现媒体、编辑列表候选区间、逐词对齐批处理与受限合并；三模态特征与最终人工验收**尚未完成，不得把本目录当成第一问已完成的提交包**。

| 脚本 | 输入 | 输出 | 状态 |
|---|---|---|---|
| `preprocess_media.py` | 原始附件1视频与`label-100.xlsx`（只读） | 100条派生完整MP4、16 kHz单声道WAV、`manifest.csv` | 100/100条通过时长、帧数与PTS校验；中间媒体不入最终附件 |
| `map_transcripts.py` | `manifest.csv`及原始MP4 | `transcript-segments.json`：各轨道编辑列表的时间尺度、起点、时长及题面文本的**候选**片段 | 100条候选已生成；不是人工确认的逐词对齐 |
| `align_words.py` | 只读原片、原文、候选音频区间、完整WAV | MFA TextGrid、`alignment.json`、状态/波形核验 | 初次100条：79通过、15待复核、6条无TextGrid；六条独立加大束宽后4通过、2待复核 |
| `combine_alignments.py` | 初次100条JSON与六条独立重试JSON | `alignment-effective.json`（保留重试溯源） | 83条自动通过、17条待复核、0缺失；仍需人工验收 |

## 运行（服务器）

环境：`/hy-tmp/E/.venv/bin/python`（需`openpyxl`）、FFmpeg/ffprobe；输入目录`/hy-tmp/mathmodel/E/input/`。所有派生文件写到`/hy-tmp/E/problem1/media/`，不修改附件1。实际命令：

```bash
cd /hy-tmp/E-collab
/hy-tmp/E/.venv/bin/python code/problem1/preprocess_media.py \
  --input /hy-tmp/mathmodel/E/input \
  --output /hy-tmp/E/problem1/media
/hy-tmp/E/.venv/bin/python code/problem1/map_transcripts.py \
  --media-dir /hy-tmp/E/problem1/media
/hy-tmp/E/.venv/bin/python code/problem1/align_words.py \
  --media-dir /hy-tmp/E/problem1/media \
  --output /hy-tmp/E/problem1/alignment/all --jobs 4
```

若只试跑3条，加`--ids-file /hy-tmp/E/problem1/media/pilot-ids.txt`，并把`--output`设为独立的`/hy-tmp/E/problem1/alignment/pilot`。初次六条无TextGrid后，按`reports/problem1-alignment/missing-ids.txt`在**独立目录**重跑：

```bash
/hy-tmp/E/.venv/bin/python code/problem1/align_words.py \
  --ids-file /hy-tmp/E/problem1/media/missing-ids.txt \
  --output /hy-tmp/E/problem1/alignment/retry_beam100 \
  --jobs 1 --beam 100 --retry-beam 400
/hy-tmp/E/.venv/bin/python code/problem1/combine_alignments.py \
  --original /hy-tmp/E/problem1/alignment/all/alignment.json \
  --retry /hy-tmp/E/problem1/alignment/retry_beam100/alignment.json \
  --output /hy-tmp/E/problem1/alignment/alignment-effective.json
```

合并脚本只允许替换原本无TextGrid的6条，按源哈希、原文和候选区间校验；不根据重试结果随意替换已对齐样本。MFA需在`/hy-tmp/E/tools/mfa`及`/hy-tmp/E/models/mfa`中有已安装的可执行文件、词典和声学模型；详细版本见`README-server.md`。批处理返回非零时检查`alignment.json`，**不等于全部数据丢失**。

`manifest.csv`须有100个唯一ID且`state=ok`；`transcript-segments.json`须有100条，但`candidate_only_unverified_audio`只表示解析成功，不表示文本覆盖整个视频，也不表示词时间已验证。`-UuX1xuaiiE/3.mp4`编辑列表在完整音轨的起点约6.573152秒，完整音轨10.612秒，默认解码只呈现末段约4.038秒。已有3条样本在完整WAV与默认解码WAV间进行波形匹配，起点与编辑列表给定值的差绝对值≤0.00003秒，归一化相关性≥0.994；**这不检验MFA词边界的准确率**。

`preprocess_media.py`可断点续跑：只跳过源SHA-256一致且两个派生文件存在的`ok`记录；失败记录重试。源数据禁止修改。`manifest.csv`包含本机绝对路径与原始转写，仅供内部追溯；正式提交前要改成可移植相对路径并去掉环境私有信息。派生MP4/WAV合计约139MB，不作为50MB提交包的一部分。

## 下一阶段（未完成）

1. **先处理当前对齐例外：** 合并结果仍有17条待复核：15条来自初次对齐（主要为缩写/标记分词差异及两条相关性低），2条来自束宽重试后静音导致波形校验不可用。优先人工回听与核查转写/声学模型；不篡改原始文本标签，不用均匀分摊造词时间。候选词时间已按**音频轨道**编辑列表起点映回全片，但自动通过不等于词边界准确。完整视频中无题面转写区间继续保留。
2. 提取全片语音帧（openSMILE）、全片视觉帧（OpenFace）与已转写词的冻结BERT文本特征，逐帧保留原片PTS、成功状态和工具版本；对无法配对的帧给出错误而非任意按位置匹配。
3. 根据固定时间窗产生三模态对齐序列，分开保存观测与padding掩码、有效长度、词/帧位置、窗口映射及未转写标记；实现100条全量结果汇总、至少一个音频/视频/文本可回查示例与独立验收。
4. 提交前锁定依赖/权重修订与下载方式，生成不含原视频的大量特征、代码、配置和复现说明；实测≤50MB，不假定BERT权重可不随包提交。

历史的`/hy-tmp/E/scripts/pilot_problem1*.py`只处理MP4编辑列表末段，现已设置安全拦截，**不要用于正式输出**。更详细的结果与限制见`reports/problem1-media/媒体预处理记录.md`、`reports/problem1-pilot/试跑记录.md`与`reports/问题一数据预处理说明.md`。
