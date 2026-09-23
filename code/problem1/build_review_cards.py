"""Build local per-sample review cards for the 17 unverified word alignments.

Contains contest transcript data: keep generated JSON out of GitHub.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main(alignment, classification, video_root, output):
    items = json.loads(alignment.read_text(encoding='utf-8'))
    classes = json.loads(classification.read_text(encoding='utf-8'))
    mapping = {item['sample_id']: item for item in items}
    if (len(items) != 100 or len(classes) != 17 or
            len(mapping) != 100 or len({item['sample_id'] for item in classes}) != 17):
        raise ValueError('unexpected input counts')
    output.mkdir(parents=True, exist_ok=True)
    lines = ['# 问题一：17条人工审核索引', '',
             '原视频根目录（本机只读）：`' + str(video_root) + '`。',
             '每条JSON位于：`' + str(output) + '`，其`video_path`为本机绝对路径；',
             '词时间是**完整原片音频秒数**。不要用旧的末段试跑JSON代替这些审核卡。',
             '下表是自动异常分类，不是人工审核结论。审核后另存记录，不改赛题原文、原视频或原始MFA记录。', '',
             '| 优先级 | 样本 | 原视频（相对上述根目录） | 审核JSON（相对本清单目录） | 待核原因 |',
             '|---|---|---|---|---|']
    for item in sorted(classes, key=lambda x: (x['priority'], x['sample_id'])):
        sid = item['sample_id']
        row = mapping[sid]
        if row['status'] != 'needs_review':
            raise ValueError('review list contains non-review sample: ' + sid)
        video_id, clip_id = sid.rsplit('__', 1)
        video = video_root / video_id / (clip_id + '.mp4')
        if not video.is_file() or sha256(video) != row['source_sha256']:
            raise ValueError('video missing or differs from aligned source: ' + sid)
        card = output / (sid + '.json')
        if card.exists():
            raise FileExistsError('review card already exists; do not overwrite manual notes: ' + str(card))
        content = {'sample_id': sid, 'priority': item['priority'], 'reason': item['reason'],
                   'video_path': str(video), 'video_sha256': row['source_sha256'],
                   'source_audio_interval_sec': row['source_audio_interval_sec'],
                   'full_audio_duration_sec': row['full_audio_duration_sec'],
                   'transcript': row['raw_text'], 'words_full_time': row['words'],
                   'word_sequence_differences': item['differences'],
                   'waveform_check': row['waveform_check'],
                   'review_status': '待人工回听；不得自动改为通过'}
        card.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')
        lines.append(f"| {item['priority']} | `{sid}` | `{video_id}/{clip_id}.mp4` | `review-cards/{sid}.json` | {item['reason']} |")
    index = output.parent / '人工审核清单.md'
    index.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('BUILT',len(classes),'cards in',output,'INDEX',index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment', type=Path, required=True)
    parser.add_argument('--classification', type=Path, required=True)
    parser.add_argument('--video-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.alignment, args.classification, args.video_root, args.output)
