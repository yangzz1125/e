# 第一问：Whisper独立语音识别试验（不替代题面转写或MFA）

## 模型、来源和范围

模型：`openai/whisper-base.en`，Hugging Face修订号`911407f4214e0e1d82085af863093ec0b66f9cd6`，模型卡许可证标示为Apache-2.0。使用服务器现有PyTorch 2.4.0+cu121、Transformers 4.44.2，RTX 3090 Ti推理；没有训练或微调。服务器从`https://hf-mirror.com`以该修订号下载，本地从官方Hugging Face下载，同一10个模型文件逐个SHA-256一致。模型保存在服务器`/hy-tmp/E/models/whisper-base.en/`，不纳入GitHub或竞赛50MB提交包；能否以外部权重复现仍以比赛规则为准。

从附件1选中6条，其中5条为P0/P1待复核样本，另选`-UuX1xuaiiE__3`作为已知转写的对照。每条先对MP4编辑列表**选中段**的16 kHz单声道WAV运行Whisper；对照样本再对完整约10.612秒WAV额外运行一次，共7条推理。只使用本地音频，结果未替换赛题原始转写、标签或MFA时间。脚本：`code/problem1/whisper_asr_check.py`，逐条原文/识别结果/音频SHA在本地`reports/problem1-alignment/whisper-pilot-mask.json`（含赛题转写，不上传GitHub），服务器`/hy-tmp/E/problem1/alignment/whisper-pilot-mask.json`。

| 样本 | 输入范围 | Whisper听到的内容（原样摘要） | 对人工审核的提示 |
|---|---|---|---|
| `-HwX2H8Z4hY__2` | 选中段 | “mediocrity. Look at the record.” | 没识别出方括号内“President Ronald Reagan”；它**可能**是说话人提示而非发声，必须回听，不能据ASR直接删原文。 |
| `-MeTTeMJBNc__13` | 选中段 | “it is. Just needs to be a simple statement and then you present the gift.” | 与题面主句基本一致，前面多出短语；回听起点和波形低相关原因。 |
| `-aqamKhZ1Ec__0` | 选中段 | “In 2008, there were very serious tribal warfare in Kenya and even as the following disputed elections.” | 与题面部分近似但插入/误识别明显；回听，不把ASR当正确文本。 |
| `-mJ2ud6oKI8__1`、`__2` | 各自选中段 | 均仅识别出“you” | 两条音频自动波形检查均标为过静；这可能是误识别或幻觉，不能用“you”验证MFA词边界。 |
| `-UuX1xuaiiE__3` | 选中段 | “My goal is to become a human rights lawyer.” | 与题面末句一致；只是ASR内容检查，不是逐词时间检验。 |
| 同上 | 完整底层音轨 | “The work that I'm doing at the Human Rights Center is that I am annotating a memoir about the Armenian genocide. My goal is to become a human rights lawyer.” | 进一步验证完整视频前段有可听语音；但前段没有此条题面给定转写，不得借ASR内容去修改题面或构造金标准。 |

Whisper是语音识别（ASR），**不是本试验所需的逐词强制对齐器**；文本与题面不同可能来自模型误识别、音质问题或题面只标注末段。显式传入输入attention mask后7条识别文本与首次试验完全一致；Transformers仍给出pad/eos相关注意力警告，因此本次试验只作定性线索，不以识别文本计算准确率、更新对齐边界或降低17条的人工复核要求。

## 复现

```bash
cd /hy-tmp/E-collab
OMP_NUM_THREADS=2 /hy-tmp/E/.venv/bin/python code/problem1/whisper_asr_check.py \
  --alignment /hy-tmp/E/problem1/alignment/alignment-effective.json \
  --media-dir /hy-tmp/E/problem1/media \
  --model-dir /hy-tmp/E/models/whisper-base.en \
  --ids-file /hy-tmp/E/problem1/alignment/whisper-pilot-ids.txt \
  --full-audio-ids=-UuX1xuaiiE__3 \
  --output /hy-tmp/E/problem1/alignment/whisper-rerun.json
```

`--output`必须是不存在的新文件，以免覆盖人工复核证据；不要把Whisper生成的前段转写写回原始Excel。
