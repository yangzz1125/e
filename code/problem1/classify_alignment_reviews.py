"""Classify automatic review reasons; do not promote MFA results to validated."""
import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import re

TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+")


def compact(parts):
    return ''.join(c for part in parts for c in part.casefold() if c.isalnum())


def classify(row):
    raw = [m.group() for m in TOKEN.finditer(row['raw_text'])]
    aligned = [w['word'] for w in row['words']]
    changes = [{'raw': raw[i:j], 'mfa': aligned[k:l]}
               for op, i, j, k, l in SequenceMatcher(a=[s.casefold() for s in raw],
                                                      b=[s.casefold() for s in aligned],
                                                      autojunk=False).get_opcodes() if op != 'equal']
    wave = row['waveform_check']['status']
    if wave == 'unverifiable_silence':
        reason, priority = '波形过静，无法验证偏移', 'P1'
    elif wave != 'pass':
        reason, priority = '波形相关性低于预设阈值', 'P1'
    elif changes and all(compact(x['raw']) == compact(x['mfa']) for x in changes):
        reason, priority = '缩写、撇号、数字后缀或前缀标记导致分词口径不一致', 'P2'
    elif changes:
        reason, priority = '题面文字与MFA输出内容不等价', 'P0'
    else:
        reason, priority = '其他，需调查', 'P1'
    return {'sample_id': row['sample_id'], 'priority': priority, 'reason': reason,
            'waveform_status': wave, 'abs_correlation': row['waveform_check'].get('abs_correlation'),
            'raw_token_count': len(raw), 'mfa_token_count': len(aligned), 'differences': changes,
            'review_status': '待人工复核，不得自动改为通过'}


def main(source, destination):
    data = json.loads(source.read_text(encoding='utf-8'))
    if len(data) != 100 or len({r['sample_id'] for r in data}) != 100:
        raise ValueError('expected complete 100-sample merged alignment')
    review = [classify(r) for r in data if r['status'] == 'needs_review']
    if len(review) != 17:
        raise ValueError('unexpected number of reviews; re-check input status')
    destination.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding='utf-8')
    for key in ('P0', 'P1', 'P2'):
        print(key, sum(r['priority'] == key for r in review))
    print('OUTPUT', destination)


if __name__ == '__main__':
    assert compact(['don', 't']) == compact(["don't"])
    assert compact(['10', 'th']) == compact(['10th'])
    assert compact(['president', 'ronald', 'reagan']) != compact(['[bracketed]'])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.input, args.output)
