# Whisper对17条待复核样本的独立语音识别诊断

**结论：Whisper有助于筛查，不能直接解决全部17条的逐词对齐。** 对17条的MP4编辑列表选中音频运行冻结的`openai/whisper-base.en`（修订号`911407f4214e0e1d82085af863093ec0b66f9cd6`，无训练、无微调），另对其中2条同时运行完整底层音轨。逐条原文与Whisper输出见本地`whisper-17.json`、`whisper-discrepant-full.json`，服务器`/hy-tmp/E/problem1/alignment/`同名文件；这些含赛题转写的JSON不上传GitHub。

| 原有异常类型 | 样本数 | Whisper给出的线索 | 是否能据此结案 |
|---|---:|---|---|
| P2 分词口径差异 | 10/12 | Whisper识别文本包含相应缩写/数字后缀/标记所对应的字母，支持“原文分词口径与MFA不同”的解释；对`and/in`等短词，这只是弱证据 | **只可处理字符串映射口径**；词边界仍需抽听，不自动改MFA状态 |
| P2 但ASR未识别争议词 | 2/12 | `-UuX1xuaiiE__0`在选中音轨和完整音轨均只识别到“enthusiasm and their passion”，未识别题面开头`^it's amazing...`；`-hnBHBN8p5A__7`完整音轨识别到的是“A single person can get the majority of an entire nation to believe and trust his or her words. Isn't that”，与题面“Picture this, before we actually start hacking, we're in a room formal setting”明显不一致 | **优先回听**原片与选中段；Whisper也可能漏词/误识别，不能据此改题面、改标签或硬配词时间 |
| P0 方括号内容与MFA词序不同 | 1 | `-HwX2H8Z4hY__2`识别到“mediocrity. Look at the record”，未识别方括号里的“President Ronald Reagan” | 可能是非语音说话人标记，需人工听证；不可将MFA的`[bracketed]`硬拆成三个词 |
| P1 波形相关性低 | 2 | `-MeTTeMJBNc__13`识别出题面主句但前面多“it is”；`-aqamKhZ1Ec__0`部分相似但插入/改词明显 | 回听剪辑边界和音质；ASR文本不验证音频偏移或MFA词边界 |
| P1 波形过静 | 2 | `-mJ2ud6oKI8__1`和`__2`均只得到“you” | 低能量语音下有误识别/幻觉风险，无法自动解决；保留低可信状态 |

**审查优先级更新（不改变原始P0/P1/P2分类）：** 先核对原P0一条、两条ASR与题面差异显著的P2、原P1四条，共7条。其余10条可先按确定的字符规范化规则建立原文—MFA词位置对应，再按缩写、`^`、数字后缀类型抽听。最终17条仍全部`needs_review`，经人工审核前不能宣称词边界可靠。

Whisper是ASR，不是本试验的逐词强制对齐器；识别文字既非赛题标准答案，也不是词时间。试验在传入输入attention mask后仍出现Transformers关于pad/eos的attention-mask警告，因此结果只能作定性线索。仅凭词面相似不能宣布对齐成功；完整原片中无题面转写的前段继续保留音视频，不用Whisper自动转写去扩充标签。

复现代码：`code/problem1/whisper_asr_check.py`。17条ID清单：本地`whisper-17-ids.txt`、服务器`/hy-tmp/E/problem1/alignment/whisper-17-ids.txt`。两条完整音轨复测ID清单：本地`whisper-discrepant-ids.txt`、服务器同目录同名文件。模型权重通过国内镜像下载、与官方源10个文件SHA-256一致；模型不入GitHub或竞赛附件。
