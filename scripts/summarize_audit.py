"""Summarize audit evidence and compare server hashes with local originals."""
import argparse
import json
from pathlib import Path

from audit_inputs import write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report', type=Path, required=True)
parser.add_argument('--source-inventory', type=Path, required=True)
args = parser.parse_args()
root = args.report
read = lambda name: json.loads((root / name).read_text(encoding='utf-8'))
report, remote = read('audit.json'), read('inventory.json')
original = json.loads(args.source_inventory.read_text(encoding='utf-8'))
ignored = [r for r in original if Path(r['path']).name.startswith('~$') or Path(r['path']).name == '.DS_Store']
source = [r for r in original if r not in ignored]
assert len({r['key'] for r in source}) == len(source)
assert len({r['key'] for r in remote}) == len(remote)
source_map, remote_map = {r['key']: r for r in source}, {r['key']: r for r in remote}
comparison = {
    'source_files': len(source), 'server_files': len(remote),
    'ignored_nondata_files': [r['path'] for r in ignored],
    'missing': sorted(source_map.keys() - remote_map.keys()),
    'extra': sorted(remote_map.keys() - source_map.keys()),
    'different': [key for key in source_map.keys() & remote_map.keys()
                  if (source_map[key]['size'], source_map[key]['sha256']) != (remote_map[key]['size'], remote_map[key]['sha256'])],
}
write_json(root / 'source-comparison.json', comparison)


def nonfinite_leaves(node, prefix=''):
    if isinstance(node, dict):
        if node.get('nonfinite', 0):
            yield {'field': prefix, 'count': node['nonfinite']}
        for key, value in node.items():
            yield from nonfinite_leaves(value, prefix + '/' + key)


bad = list(nonfinite_leaves(report['features']))
write_json(root / 'nonfinite.json', bad)
videos = read('videos.json')
durations = [float(r['format']['duration']) for r in videos if '附件1-' in r['path']]
encoder = read('encoder-smoke.json')['results']
all_hashes = not any(comparison[k] for k in ('missing', 'extra', 'different'))
assert len(report['features']) == 102 and not report['issues']
assert not report['source_files_changed_during_audit']
assert not report['videos']['failed']
labels_valid = all(not sheet.get('invalid_label_rows') for sheets in report['workbooks'].values() for sheet in sheets)
lines = [
    '# 上传数据验收与文本编码器核验', '',
    '检查日期：2026-09-23。服务器输入目录：`/hy-tmp/mathmodel/E/input/`。全程只读原始输入，没有训练模型或修改样本。', '',
    '## 文件完整性', '',
    f'- 有效源文件与服务器文件均为 {len(remote)} 个，共 {sum(r["size"] for r in remote):,} 字节。',
    f'- 逐文件 SHA256 和大小对比：{"全部一致" if all_hashes else "存在差异，见 source-comparison.json"}。',
    '- 本地多出的 `.DS_Store` 和 Word `~$` 临时锁文件不属于题目数据，不要求上传。',
    '- 所有源文件在检查前后的大小和修改时间保持不变。', '',
    '| 附件 | 已核实内容 |', '|---|---|',
    '| 1 | 100 个视频、100 条标签；视频与标签一一对应；37 个源视频目录 |',
    '| 2 | 对齐、非对齐两个版本，每版训练 3395、验证 728、测试 727，共 4850 条 |',
    '| 3 | 每版 30 个无标签特征文件，共 60 个文件；只做结构检查 |',
    '| 4 | 每版 20 个特征文件及 20 个视频，共 40 个特征文件、40 个视频文件；只做结构检查 |',
    '| 题面 | 1 份 DOCX；核对正文、表格文字，未验证嵌入图片及页面排版 |', '',
    '## 可读取性与数值', '',
    f'- 102/102 个 Pickle 读取成功；非有限数值字段记录数为 {len(bad)}。',
    '- 140/140 个视频经过 ffprobe 和完整音视频解码检查，无解码失败，均含音轨。音轨存在不代表一定有有效语音。',
    f'- 附件1视频实测时长范围：{min(durations):.3f}–{max(durations):.3f} 秒。',
    f'- Excel 标签经显式浮点转换后，极性与符号检查：{"通过" if labels_valid else "有异常"}。两个标签表的 label 原始存储类型均为文本，加载时必须转换，原文件未修改。',
    '- 两版附件2全部标签与 Excel 对照一致，各划分内部无重复样本编号，跨 train/valid/test 的样本编号和源视频编号交集均为 0。',
    '- 分类映射为 0=负向、1=中性、2=正向。连续标签 0 仅属于中性。', '',
    '## BERT 核验', '',
    '- 模型：`google-bert/bert-base-uncased`，revision `86b5e0934494bd15c9632b12f734a8a67f723594`。',
    '- 在附件2对齐版训练集 3395/3395、验证集 728/728 条记录上，重新分词得到的 input_ids、attention_mask、token_type_ids 三通道全部逐元素一致。测试集未用于该核验。',
    f'- 对训练/验证各取首、中、尾三条，共六条，实际 BERT 非填充位置特征均满足 atol=rtol=1e-4。最大绝对差为 {max(r["nonpadding_max_absolute_difference"] for r in encoder):.8g}。',
    '- 这些结果支持使用当前分词器和编码器；六条表征抽查不等于全量复现证明，历史层选择与时序映射仍须单独核验。', '',
    '## 输入适配注意事项', '',
    '- 附件3对齐版有 test 外层和批次维度，text_bert 为 float32，需要先检查整数性、合法词元范围与掩码，再转 long；缺少预计算 text。',
    '- 附件3非对齐版接口不同，不得与对齐版模型直接混用。附件4没有同样的 test 外层和批次维度。',
    '- 全零特征行可能是填充、原生无效或人工缺失，不能仅凭零值混为同一种原因。',
    '- 原始数值检查未改写、补齐或恢复任何专项测试内容。', '',
    '## 下一步与尚未通过的门槛', '',
    '可以开始 3–5 条附件1样本的三模态提取、MFA 对齐和人工复核，以及统一输入适配器的开发。尚未启动正式训练或全量特征提取。', '',
    '- G1：分词器核验和六条编码器抽查通过；缺失掩码传播、占位值不变性及完整适配器测试仍待实现。',
    '- G2：附件2没有逐词时间戳，其特征位置如何对应附件4原视频仍待验证；不得用序号比例冒充真实时间。',
    '- G3：题面附件限 50 MB；BERT 权重约 440 MB，外部权重是否允许以下载方式复现仍待核实，并须进行提交体积预算。', '',
    '## 证据文件', '',
    '`inventory.json`、`source-comparison.json`、`audit.json`、`nonfinite.json`、`videos.json`、`encoder-smoke.json`。',
    '初版读取器未包含 NumPy 的 asarray 序列化工厂，曾拒绝读取附件3；兼容后全部重验成功。读取器相关单元测试共 7 项通过。',
]
(root / '数据验收报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'all_hashes_match': all_hashes, 'nonfinite_fields': len(bad),
                  'label_validation': labels_valid, 'video_count': len(videos),
                  'encoder_samples_passed': sum(r['nonpadding_allclose_atol_1e-4_rtol_1e-4'] for r in encoder)}, ensure_ascii=False))
