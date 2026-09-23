"""Merge a documented MFA retry into the original 100-sample alignment status.

Only replace rows that lacked TextGrid in the original run; never silently overwrite
an alignment that already existed or promote a review to pass.
"""
import argparse
import json
from pathlib import Path


def main(original, retry, output):
    before = json.loads(original.read_text(encoding='utf-8'))
    later = json.loads(retry.read_text(encoding='utf-8'))
    if len(before) != 100 or len({r['sample_id'] for r in before}) != 100:
        raise ValueError('expected original 100 unique sample IDs')
    old = {r['sample_id']: r for r in before}
    if len(later) != len({r['sample_id'] for r in later}):
        raise ValueError('duplicate retry IDs')
    replaced = {}
    for r in later:
        sid = r['sample_id']
        if sid not in old or old[sid]['status'] != 'error':
            raise ValueError('retry may replace only an original error: ' + sid)
        if (r['status'] not in ('pass', 'needs_review') or
                r['source_sha256'] != old[sid]['source_sha256'] or
                r['raw_text'] != old[sid]['raw_text'] or
                r['source_audio_interval_sec'] != old[sid]['source_audio_interval_sec']):
            raise ValueError('unsafe retry result: ' + sid)
        r['provenance'] = {'original_status': 'error', 'retry_source': str(retry),
                           'retry_settings': 'MFA beam=100, retry_beam=400, jobs=1'}
        replaced[sid] = r
    if set(replaced) != {r['sample_id'] for r in before if r['status'] == 'error'}:
        raise ValueError('not all original errors were accounted for')
    merged = [replaced.get(r['sample_id'], r) for r in before]
    output.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding='utf-8')
    print('MERGED',len(merged),'pass',sum(x['status']=='pass' for x in merged),
          'needs_review',sum(x['status']=='needs_review' for x in merged),
          'errors',sum(x['status']=='error' for x in merged))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--retry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.original, args.retry, args.output)
