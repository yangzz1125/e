"""Stage full original audio/video for all Problem 1 samples without altering source files.

Run on the server: /hy-tmp/E/.venv/bin/python scripts/preprocess_problem1_media.py
The derived media and manifest are NOT the final three-modality features.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import openpyxl

INPUT = Path('/hy-tmp/mathmodel/E/input')
OUTPUT = Path('/hy-tmp/E/problem1/media')
FIELDS = ['sample_id', 'video_id', 'clip_id', 'source', 'source_sha256', 'raw_video', 'full_wav',
          'raw_video_sec', 'raw_audio_sec', 'edited_video_sec', 'edited_audio_sec',
          'video_start_pts_sec', 'audio_start_pts_sec', 'raw_frame_count', 'decoded_frame_count',
          'wav_sec', 'raw_text', 'label', 'annotation', 'state', 'error']


def command(args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=180).stdout


def probe(path, *, ignore_editlist=False):
    args = ['ffprobe', '-v', 'error']
    if ignore_editlist:
        args += ['-ignore_editlist', '1']
    args += ['-show_entries', 'format=duration:stream=codec_type,duration,start_time', '-of', 'json', str(path)]
    info = json.loads(command(args))
    streams = {s['codec_type']: s for s in info['streams'] if s['codec_type'] in ('video', 'audio')}
    if set(streams) != {'video', 'audio'} or any('duration' not in s for s in streams.values()):
        raise ValueError('expected video and audio track durations')
    return {key + '_sec': float(s['duration']) for key, s in streams.items()} | {
        key + '_start_pts_sec': float(s.get('start_time', 0)) for key, s in streams.items()}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def frames(path, *, ignore_editlist=False):
    args = ['ffprobe', '-v', 'error']
    if ignore_editlist:
        args += ['-ignore_editlist', '1']
    raw = command(args + ['-select_streams', 'v:0', '-show_frames',
                          '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(path)])
    pts = [float(f['best_effort_timestamp_time']) for f in json.loads(raw)['frames'] if 'best_effort_timestamp_time' in f]
    if not pts or any(left >= right for left, right in zip(pts, pts[1:])):
        raise ValueError('missing or non-increasing video frame PTS')
    return pts


def main(input_dir, output_dir):
    labels_path = next(input_dir.rglob('label-100.xlsx'))
    sheet = openpyxl.load_workbook(labels_path, read_only=True, data_only=True)['label']
    header, *values = sheet.values
    if tuple(header[:5]) != ('video_id', 'clip_id', 'text', 'label', 'annotation'):
        raise ValueError('unexpected label columns')
    labels = {(str(row[0]), str(row[1])): row for row in values}
    if len(labels) != 100 or len(values) != 100:
        raise ValueError('label count/duplicate error')
    video_roots = list(input_dir.rglob('MOSEI数据集部分原始视频-100条'))
    if len(video_roots) != 1:
        raise ValueError('unexpected video root')
    videos = {(p.parent.name, p.stem): p for p in video_roots[0].rglob('*.mp4')}
    if set(videos) != set(labels) or len(videos) != 100:
        raise ValueError('source video/label mismatch')
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / 'manifest.csv'
    # Continue completed rows; never overwrite another source's derived media.
    done = {}
    records = {}
    if manifest.exists():
        with manifest.open(encoding='utf-8', newline='') as f:
            old = list(csv.DictReader(f))
        if len(old) != len({r['sample_id'] for r in old}):
            raise ValueError('duplicate manifest ID')
        records = {r['sample_id']: r for r in old}
        done = {key: r for key, r in records.items() if r['state'] == 'ok'}
    for (video_id, clip_id), source in sorted(videos.items()):
        sample_id = f'{video_id}__{clip_id}'
        if sample_id in done:
            prev = done[sample_id]
            if (prev['source_sha256'] != sha256(source) or not Path(prev['raw_video']).is_file()
                    or not Path(prev['full_wav']).is_file()):
                raise ValueError('existing manifest/source or derived file mismatch: ' + sample_id)
            print('SKIP', sample_id, flush=True)
            continue
        if sample_id in records and records[sample_id]['source_sha256'] != sha256(source):
            raise ValueError('source changed since previous attempt: ' + sample_id)
        path = output_dir / sample_id
        path.mkdir(exist_ok=True)
        full_video = path / 'full.mp4'
        full_wav = path / 'full-16k-mono.wav'
        row = {'sample_id': sample_id, 'video_id': video_id, 'clip_id': clip_id,
               'source': str(source), 'source_sha256': sha256(source),
               'raw_video': str(full_video), 'full_wav': str(full_wav),
               'raw_text': str(labels[video_id, clip_id][2]),
               'label': str(labels[video_id, clip_id][3]),
               'annotation': str(labels[video_id, clip_id][4]), 'state': 'error', 'error': ''}
        try:
            raw = probe(source, ignore_editlist=True)
            edited = probe(source)
            row.update(raw_video_sec=raw['video_sec'], raw_audio_sec=raw['audio_sec'],
                       edited_video_sec=edited['video_sec'], edited_audio_sec=edited['audio_sec'],
                       video_start_pts_sec=raw['video_start_pts_sec'], audio_start_pts_sec=raw['audio_start_pts_sec'])
            command(['ffmpeg', '-nostdin', '-v', 'error', '-ignore_editlist', '1', '-i', str(source),
                     '-map', '0:v:0', '-map', '0:a:0', '-c', 'copy', '-y', str(full_video)])
            command(['ffmpeg', '-nostdin', '-v', 'error', '-ignore_editlist', '1', '-i', str(source),
                     '-map', '0:a:0', '-ar', '16000', '-ac', '1', '-y', str(full_wav)])
            staged = probe(full_video)
            wav = json.loads(command(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                       '-of', 'json', str(full_wav)]))
            row['wav_sec'] = float(wav['format']['duration'])
            if abs(staged['video_sec'] - raw['video_sec']) > 0.05 or abs(staged['audio_sec'] - raw['audio_sec']) > 0.05 or abs(row['wav_sec'] - raw['audio_sec']) > 0.08:
                raise ValueError('full-stream durations do not match source')
            original_pts = frames(source, ignore_editlist=True)
            staged_pts = frames(full_video)
            row['raw_frame_count'] = len(original_pts)
            row['decoded_frame_count'] = len(staged_pts)
            if len(original_pts) != len(staged_pts) or any(abs(a - b) > 0.04 for a, b in zip(original_pts, staged_pts)):
                raise ValueError('remux changed decoded frame count or PTS')
            row['state'] = 'ok'
        except Exception as exc:
            row['error'] = str(exc)[:500]
            print('FAILED', sample_id, row['error'], flush=True)
        records[sample_id] = row
        with manifest.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(records[key] for key in sorted(records))
        print(row['state'].upper(), sample_id, len(records), '/100', flush=True)
    print('SUMMARY', len(records), '/100', 'ok', sum(r['state']=='ok' for r in records.values()), flush=True)
    if len(records) != 100 or any(r['state'] != 'ok' for r in records.values()):
        raise SystemExit('Media stage incomplete; inspect manifest.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=INPUT, help='read-only contest E input directory')
    parser.add_argument('--output', type=Path, default=OUTPUT, help='derived media directory')
    args = parser.parse_args()
    main(args.input, args.output)
