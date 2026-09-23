"""Align provided transcripts to the MP4-edited audio and map words onto full audio time.

Usage: python code/problem1/align_words.py --media-dir ... --output ... [--ids-file ...]
Requires FFmpeg, MFA and praatio in MFA's separate environment. Inputs remain read-only.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import correlate

PROJECT = Path('/hy-tmp/E')
MEDIA = PROJECT / 'problem1/media'
MFA = PROJECT / 'tools/mfa/bin/mfa'
MFA_PYTHON = PROJECT / 'tools/mfa/bin/python'
DICTIONARY = PROJECT / 'models/mfa/english_us_arpa.dict'
ACOUSTIC = PROJECT / 'models/mfa/english_us_arpa.zip'
TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+")
GRID_READER = """import json,sys
from praatio import textgrid
g=textgrid.openTextgrid(sys.argv[1],includeEmptyIntervals=False)
t=next(g.getTier(n) for n in g.tierNames if 'word' in n.lower())
print(json.dumps([{'word':v.label,'start_local_sec':v.start,'end_local_sec':v.end}
                  for v in t.entries if v.label.strip()]))"""


def run(args, **kwargs):
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=kwargs.pop('timeout', 180), **kwargs)


def audio_match(cropped_path, full_path, expected):
    """Independent offset check from waveforms, choosing the loudest 0.5s window."""
    a, sample_rate = sf.read(cropped_path, dtype='float32')
    b, full_rate = sf.read(full_path, dtype='float32')
    if sample_rate != full_rate or sample_rate != 16000 or len(a) < sample_rate // 2:
        return {'status': 'unverifiable_audio_length_or_rate'}
    window = sample_rate // 2
    points = list(range(0, len(a) - window + 1, window))
    i = max(points, key=lambda start: float(np.mean(a[start:start + window] ** 2)))
    segment = a[i:i + window]
    if np.mean(segment ** 2) < 1e-7:
        return {'status': 'unverifiable_silence'}
    approximate = round((expected + i / sample_rate) * sample_rate)
    lo = max(0, approximate - round(.10 * sample_rate))
    hi = min(len(b), approximate + round(.10 * sample_rate) + window)
    if hi - lo < window:
        return {'status': 'unverifiable_outside_full_audio'}
    values = correlate(b[lo:hi], segment, mode='valid', method='fft')
    idx = int(np.argmax(np.abs(values)))
    offset = (lo + idx - i) / sample_rate
    denominator = np.linalg.norm(segment) * np.linalg.norm(b[lo + idx:lo + idx + window])
    similarity = float(abs(values[idx]) / denominator) if denominator else 0.0
    delta = float(offset - expected)
    return {'status': 'pass' if abs(delta) <= .02 and similarity >= .9 else 'needs_review',
            'expected_offset_sec': expected, 'waveform_offset_sec': offset,
            'offset_error_sec': delta, 'abs_correlation': similarity}


def main(args):
    with (args.media_dir / 'manifest.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    intervals = {r['sample_id']: r for r in json.loads((args.media_dir / 'transcript-segments.json').read_text(encoding='utf-8'))}
    if len(rows) != 100 or len(intervals) != 100 or any(r['state'] != 'ok' for r in rows):
        raise ValueError('media manifest or edit-list candidates incomplete')
    available = {r['sample_id']: r for r in rows}
    ids = sorted(available)
    if args.ids_file:
        ids = [s.strip() for s in args.ids_file.read_text(encoding='utf-8').splitlines() if s.strip()]
        if len(ids) != len(set(ids)) or not ids or not set(ids) <= set(available):
            raise ValueError('invalid IDs in subset file')
    args.output.mkdir(parents=True, exist_ok=True)
    corpus = args.output / 'corpus'
    aligned = args.output / 'textgrids'
    records = {}
    for index, sid in enumerate(ids):
        row = available[sid]
        segment = intervals[sid]
        if segment['state'] != 'candidate_only_unverified_audio':
            raise ValueError('edit-list segment not safe to align: ' + sid)
        if abs(float(row['audio_start_pts_sec'])) > .001:
            raise ValueError('audio full WAV starts at nonzero PTS: need explicit conversion ' + sid)
        speaker = f'speaker_{index:03d}'
        base = f'clip_{index:03d}'
        folder = corpus / speaker
        folder.mkdir(parents=True, exist_ok=True)
        wav = folder / (base + '.wav')
        text = folder / (base + '.lab')
        if not wav.exists():
            run(['ffmpeg', '-nostdin', '-v', 'error', '-i', row['source'], '-map', '0:a:0',
                 '-ar', '16000', '-ac', '1', str(wav)])  # default edit-list view intentionally
        if text.exists() and text.read_text(encoding='utf-8') != row['raw_text']:
            raise ValueError('existing lab differs from original transcript: ' + sid)
        text.write_text(row['raw_text'], encoding='utf-8')
        local_duration = sf.info(wav).duration
        candidate = segment['candidate_audio_interval_sec']
        if abs(local_duration - (candidate[1] - candidate[0])) > .08:
            raise ValueError('cropped audio and edit duration differ: ' + sid)
        records[sid] = {'sample_id': sid, 'source_sha256': row['source_sha256'],
                        'raw_text': row['raw_text'], 'source_audio_interval_sec': candidate,
                        'full_audio_duration_sec': float(row['raw_audio_sec']),
                        'local_wav_duration_sec': local_duration, 'speaker': speaker, 'base': base}
    env = os.environ.copy()
    env['PATH'] = str(MFA.parent) + os.pathsep + env.get('PATH', '')
    env['MFA_ROOT_DIR'] = str(PROJECT / 'models/mfa')
    env['OMP_NUM_THREADS'] = '2'
    command = [str(MFA), 'align', str(corpus), str(DICTIONARY), str(ACOUSTIC),
               str(aligned), '--num_jobs', str(args.jobs), '--clean']
    if args.beam is not None:
        command += ['--beam', str(args.beam), '--retry_beam', str(args.retry_beam)]
    try:
        result = run(command, env=env, timeout=args.timeout)
        (args.output / 'mfa.log').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        (args.output / 'mfa.log').write_text(str(exc) + '\n' + (getattr(exc, 'stdout', '') or '') + '\n' + (getattr(exc, 'stderr', '') or ''), encoding='utf-8')
        raise
    status = []
    for sid in ids:
        info = records[sid]
        path = aligned / info['speaker'] / (info['base'] + '.TextGrid')
        entry = dict(info, status='missing_textgrid', word_count=0)
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            words = json.loads(run([str(MFA_PYTHON), '-c', GRID_READER, str(path)]).stdout)
            offset, candidate_end = info['source_audio_interval_sec']
            full_end = info['full_audio_duration_sec']
            for word in words:
                start, end = word['start_local_sec'], word['end_local_sec']
                if not (0 <= start < end <= info['local_wav_duration_sec'] + .03):
                    raise ValueError('MFA word outside cropped WAV')
                word['start_full_sec'] = round(offset + start, 6)
                word['end_full_sec'] = round(offset + end, 6)
                if word['end_full_sec'] > min(candidate_end, full_end) + .05:
                    raise ValueError('MFA word outside full audio/edit interval')
            transcribed = [match.group().casefold() for match in TOKEN.finditer(info['raw_text'])]
            aligned_words = [word['word'].casefold() for word in words]
            match = aligned_words == transcribed
            wave = audio_match(corpus / info['speaker'] / (info['base'] + '.wav'),
                               args.media_dir / sid / 'full-16k-mono.wav', offset)
            entry.update(status='pass' if match and wave['status'] == 'pass' else 'needs_review',
                         words=words, word_count=len(words), transcript_word_count=len(transcribed),
                         word_sequence_match=match, waveform_check=wave)
        except Exception as exc:
            entry.update(status='error', error=str(exc)[:400])
        status.append(entry)
        print(sid, entry['status'], entry['word_count'], flush=True)
    (args.output / 'alignment.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SUMMARY', {name: sum(r['status'] == name for r in status) for name in ('pass', 'needs_review', 'error', 'missing_textgrid')})
    if any(r['status'] in ('error', 'missing_textgrid') for r in status):
        raise SystemExit('incomplete word alignment; inspect alignment.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media-dir', type=Path, default=MEDIA)
    parser.add_argument('--output', type=Path, default=PROJECT / 'problem1/alignment/all')
    parser.add_argument('--ids-file', type=Path)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--beam', type=int, help='optional MFA beam for explicitly isolated retry')
    parser.add_argument('--retry-beam', type=int, default=400)
    main(parser.parse_args())
