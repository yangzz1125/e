"""Problem 1 real-data pilot: inspect extraction and alignment, not final features.

Run on the server with /hy-tmp/E/.venv/bin/python scripts/pilot_problem1.py.
Never changes input files; outputs under /hy-tmp/E/problem1/pilot/.
"""
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import openpyxl
import opensmile
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
INPUT = Path('/hy-tmp/mathmodel/E/input')
OUTPUT = ROOT / 'problem1' / 'pilot'
SAMPLES = [('-mJ2ud6oKI8', '6'), ('-UuX1xuaiiE', '3'), ('-yRb-Jum7EQ', '1')]
MFA = ROOT / 'tools/mfa/bin/mfa'
OPENFACE = ROOT / 'tools/OpenFace-linux/build/bin/FeatureExtraction'
MODEL = ROOT / 'models/bert-base-uncased'


def run(args, *, env=None, timeout=900):
    p = subprocess.run(args, env=env, capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f'{args[0]} returned {p.returncode}: {p.stderr[-2500:]} {p.stdout[-1000:]}')
    return p


def main():
    if not INPUT.is_dir():
        raise FileNotFoundError(INPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = next(INPUT.rglob('label-100.xlsx'))
    sheet = openpyxl.load_workbook(table, read_only=True, data_only=True)['label']
    labels = {(str(v), str(c)): text for v, c, text, *_ in list(sheet.values)[1:]}
    videos = list(INPUT.rglob('MOSEI数据集部分原始视频-100条'))
    if len(videos) != 1:
        raise RuntimeError(f'Expected one original-video directory; got {videos}')
    corpus = OUTPUT / 'corpus'
    records = []
    for video_id, clip_id in SAMPLES:
        video = videos[0] / video_id / f'{clip_id}.mp4'
        text = labels[video_id, clip_id]
        if not video.is_file():
            raise FileNotFoundError(video)
        target = corpus / f'speaker_{len(records):02d}'
        target.mkdir(parents=True, exist_ok=True)
        name = f'clip_{len(records):02d}'
        (target / f'{name}.lab').write_text(text, encoding='utf-8')
        wav = target / f'{name}.wav'
        run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(video), '-vn', '-ar', '16000', '-ac', '1', str(wav)])
        info = json.loads(run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=index,codec_type,start_time,duration', '-of', 'json', str(video)]).stdout)
        durations = [float(stream['duration']) for stream in info['streams'] if stream['codec_type'] in ('audio', 'video')]
        if len(durations) != 2:
            raise ValueError('Both audio and video track durations required: ' + str(video))
        raw = json.loads(run(['ffprobe', '-v', 'error', '-ignore_editlist', '1',
                              '-show_entries', 'stream=codec_type,duration', '-of', 'json', str(video)]).stdout)
        raw_durations = {s['codec_type']: float(s['duration']) for s in raw['streams'] if s['codec_type'] in ('audio', 'video')}
        if any(raw_durations[s['codec_type']] - float(s['duration']) > 0.05 for s in info['streams'] if s['codec_type'] in raw_durations):
            raise RuntimeError('MP4 edit list hides source footage; this cropped-track pilot is obsolete: ' + str(video))
        records.append({'video_id': video_id, 'clip_id': clip_id, 'text': text, 'source': str(video),
                        'duration': max(durations), 'container_duration_reported': float(info['format']['duration']),
                        'streams': info['streams'],
                        'wav': str(wav), 'mfa_name': name, 'speaker': target.name})
    # Keep alignment separate from the original input; never use labels to adjust boundaries.
    env = os.environ.copy()
    env.update({'MFA_ROOT_DIR': str(ROOT / 'models/mfa'), 'OMP_NUM_THREADS': '2',
                'PATH': str(ROOT / 'tools/mfa/bin') + os.pathsep + env.get('PATH', '')})
    cmd = [str(MFA), 'align', str(corpus), str(ROOT / 'models/mfa/english_us_arpa.dict'),
           str(ROOT / 'models/mfa/english_us_arpa.zip'), str(OUTPUT / 'aligned'),
           '--num_jobs', '3', '--clean']
    align = run(cmd, env=env, timeout=1800)
    (OUTPUT / 'mfa.log').write_text(align.stdout + '\n' + align.stderr, encoding='utf-8')
    # praatio belongs to the isolated MFA environment.
    reader = """import json,sys
from praatio import textgrid
g=textgrid.openTextgrid(sys.argv[1],includeEmptyIntervals=False)
t=next(g.getTier(n) for n in g.tierNames if 'word' in n.lower())
print(json.dumps([{'word':w.label,'start':w.start,'end':w.end} for w in t.entries if w.label.strip()]))"""
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(str(MODEL), local_files_only=True, use_safetensors=True).eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
    for record in records:
        stem = record['mfa_name']
        grid = OUTPUT / 'aligned' / record['speaker'] / f'{stem}.TextGrid'
        record['textgrid'] = str(grid)
        if not grid.is_file():
            record['alignment_error'] = 'TextGrid not produced'
            record['words'] = []
        else:
            words = json.loads(run([str(ROOT / 'tools/mfa/bin/python'), '-c', reader, str(grid)]).stdout)
            record['words'] = words
            record['words_valid'] = all(0 <= w['start'] < w['end'] <= record['duration'] + 0.2 for w in words)
            record['word_count_transcript'] = len(re.findall(r"\b[\w']+\b", record['text']))
            record['word_count_aligned'] = len(words)
        voice = smile.process_file(record['wav'])
        record['audio_frames'] = len(voice)
        record['audio_dims'] = voice.shape[1]
        record['audio_finite'] = bool(np.isfinite(voice.to_numpy()).all())
        record['audio_first_intervals'] = [[str(start), str(end)] for _, start, end in voice.index[:3]]
        ids = tokenizer(record['text'], return_offsets_mapping=True, return_tensors='pt', truncation=False)
        offsets = ids.pop('offset_mapping')[0].tolist()
        if len(offsets) > model.config.max_position_embeddings:
            record['bert_error'] = 'longer than model context; requires chunking'
        else:
            with torch.inference_mode():
                hidden = model(**{k: v.to(model.device) for k, v in ids.items()}).last_hidden_state
            record['bert_tokens'] = len(offsets)
            record['bert_dims'] = hidden.shape[-1]
            record['bert_finite'] = bool(torch.isfinite(hidden).all())
            record['bert_first_offsets'] = offsets[:6]
        face_dir = OUTPUT / 'openface' / record['speaker']
        face_dir.mkdir(parents=True, exist_ok=True)
        face = run([str(OPENFACE), '-f', record['source'], '-out_dir', str(face_dir), '-aus', '-pose', '-gaze', '-2Dfp', '-q'], timeout=1800)
        (face_dir / 'run.log').write_text(face.stdout + '\n' + face.stderr, encoding='utf-8')
        csv_path = face_dir / (Path(record['source']).stem + '.csv')
        if not csv_path.is_file():
            record['face_error'] = 'OpenFace CSV not produced'
        else:
            with csv_path.open(newline='', encoding='utf-8') as f:
                rows = list(csv.DictReader(f, skipinitialspace=True))
            record['face_frames'] = len(rows)
            record['face_success'] = sum(float(r['success']) > 0 for r in rows)
            record['face_confidence_range'] = [min(float(r['confidence']) for r in rows), max(float(r['confidence']) for r in rows)] if rows else []
            record['face_timestamp_range'] = [float(rows[0]['timestamp']), float(rows[-1]['timestamp'])] if rows else []
            record['face_csv'] = str(csv_path)
        (OUTPUT / 'pilot-summary.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k: record.get(k) for k in ['video_id', 'clip_id', 'duration', 'word_count_transcript', 'word_count_aligned', 'words_valid', 'audio_frames', 'audio_dims', 'bert_tokens', 'face_frames', 'face_success']}, ensure_ascii=False), flush=True)
    print('PILOT_SUMMARY', OUTPUT / 'pilot-summary.json')


if __name__ == '__main__':
    main()
