"""Aggregate real pilot frames to MFA word intervals; do not modify the source data."""
import csv
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import opensmile
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'problem1' / 'pilot'
MODEL = ROOT / 'models/bert-base-uncased'


def aggregate(words, starts, ends, values, good):
    """Overlap-weighted mean, zero output only with mask=0."""
    start = np.asarray(starts, dtype=np.float64)
    end = np.asarray(ends, dtype=np.float64)
    value = np.asarray(values, dtype=np.float32)
    good = np.asarray(good, dtype=bool)
    result = np.zeros((len(words), value.shape[1]), dtype=np.float32)
    mask = np.zeros(len(words), dtype=np.uint8)
    coverage = np.zeros(len(words), dtype=np.float32)
    for k, word in enumerate(words):
        overlap = np.maximum(0, np.minimum(word['end'], end) - np.maximum(word['start'], start))
        overlap *= good
        if overlap.sum() > 0:
            result[k] = np.average(value, axis=0, weights=overlap)
            mask[k] = 1
            coverage[k] = min(1.0, overlap.sum() / (word['end'] - word['start']))
    return result, mask, coverage


def frame_intervals(timestamps, track_end):
    """Assign sampled visual frames midpoint support, bounded by real stream end."""
    times = np.asarray(timestamps, dtype=np.float64)
    if not len(times) or np.any(np.diff(times) <= 0) or track_end <= times[-1]:
        raise ValueError('invalid frame time range')
    mid = (times[1:] + times[:-1]) / 2
    return np.r_[max(0, times[0]), mid], np.r_[mid, track_end]


def main():
    records = json.loads((PILOT / 'pilot-summary.json').read_text(encoding='utf-8'))
    for record in records:
        raw = subprocess.run(['ffprobe', '-v', 'error', '-ignore_editlist', '1',
                              '-show_entries', 'stream=codec_type,duration', '-of', 'json', record['source']],
                             capture_output=True, text=True, check=True)
        durations = {s['codec_type']: float(s['duration']) for s in json.loads(raw.stdout)['streams'] if s['codec_type'] in ('audio', 'video')}
        if any(durations[s['codec_type']] - float(s['duration']) > 0.05 for s in record['streams'] if s['codec_type'] in durations):
            raise RuntimeError('pilot features are cropped by MP4 edit list; do not reuse as full-video features')
    out = PILOT / 'features'
    out.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), use_fast=True, local_files_only=True)
    model = AutoModel.from_pretrained(str(MODEL), use_safetensors=True, local_files_only=True).eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
    report = []
    for r in records:
        words = r['words']
        transcript = list(re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", r['text']))
        if len(words) != len(transcript) or any(w['word'].casefold() != match.group().casefold()
                                                    for w, match in zip(words, transcript)):
            raise ValueError('MFA words do not match transcript: ' + r['video_id'])
        audio_end_time = float(next(s['duration'] for s in r['streams'] if s['codec_type'] == 'audio'))
        if not all(0 <= w['start'] < w['end'] <= audio_end_time + 0.03 for w in words):
            raise ValueError('word interval outside decoded audio track')
        tokens = tokenizer(r['text'], return_offsets_mapping=True, return_tensors='pt', truncation=False)
        offsets = tokens.pop('offset_mapping')[0].tolist()
        if len(offsets) > model.config.max_position_embeddings:
            raise ValueError('text exceeds model context: needs chunking')
        with torch.inference_mode():
            hidden = model(**{k: v.to(model.device) for k, v in tokens.items()}).last_hidden_state[0].cpu().numpy()
        text = np.zeros((len(words), hidden.shape[1]), dtype=np.float32)
        token_indices = []
        for i, match in enumerate(transcript):
            indices = [j for j, (a, b) in enumerate(offsets) if b > a and a < match.end() and b > match.start()]
            if not indices:
                raise ValueError('no BERT tokens for word ' + match.group())
            text[i] = hidden[indices].mean(axis=0)
            token_indices.append(indices)
        audio = smile.process_file(r['wav'])
        audio_start = np.asarray([start.total_seconds() for _, start, _ in audio.index])
        audio_end = np.asarray([end.total_seconds() for _, _, end in audio.index])
        acoustic, audio_mask, audio_coverage = aggregate(words, audio_start, audio_end, audio.to_numpy(),
                                                          np.isfinite(audio.to_numpy()).all(axis=1))
        with Path(r['face_csv']).open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            names = [n for n in reader.fieldnames if n.endswith('_r') and n.startswith('AU')]
            names += ['pose_Rx', 'pose_Ry', 'pose_Rz', 'gaze_angle_x', 'gaze_angle_y']
            rows = list(reader)
        track_end = float(next(s['duration'] for s in r['streams'] if s['codec_type'] == 'video'))
        pts = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
                              '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', r['source']],
                             check=True, capture_output=True, text=True)
        visual_t = [float(frame['best_effort_timestamp_time']) for frame in json.loads(pts.stdout)['frames']]
        if len(visual_t) != len(rows):
            raise ValueError('decoded video frame count differs from OpenFace rows: ' + r['video_id'])
        timestamp_delta = max(abs(a - float(b['timestamp'])) for a, b in zip(visual_t, rows))
        visual_start, visual_end = frame_intervals(visual_t, track_end)
        visual_values = np.asarray([[float(row[n]) for n in names] for row in rows], dtype=np.float32)
        visual_good = np.asarray([int(float(row['success'])) == 1 and float(row['confidence']) >= 0.5
                                  for row in rows]) & np.isfinite(visual_values).all(axis=1)
        vision, vision_mask, vision_coverage = aggregate(words, visual_start, visual_end, visual_values, visual_good)
        if not all(np.isfinite(x).all() for x in (text, acoustic, vision)):
            raise ValueError('nonfinite features')
        spans = np.asarray([[w['start'], w['end']] for w in words], dtype=np.float32)
        chars = np.asarray([[m.start(), m.end()] for m in transcript], dtype=np.int32)
        filename = f"{r['video_id']}__{r['clip_id']}"
        np.savez_compressed(out / (filename + '.npz'), text=text, audio=acoustic, vision=vision,
                            word_times=spans, char_spans=chars, text_mask=np.ones(len(words), dtype=np.uint8),
                            audio_mask=audio_mask, vision_mask=vision_mask,
                            audio_coverage=audio_coverage, vision_coverage=vision_coverage)
        mapping = [{'word': w['word'], 'start': w['start'], 'end': w['end'],
                    'chars': chars[i].tolist(), 'bert_token_positions': token_indices[i],
                    'audio_observed': bool(audio_mask[i]), 'vision_observed': bool(vision_mask[i]),
                    'audio_coverage': float(audio_coverage[i]), 'vision_coverage': float(vision_coverage[i])}
                   for i, w in enumerate(words)]
        (out / (filename + '.json')).write_text(json.dumps({'id': filename, 'source': r['source'],
           'container_duration_reported': r['container_duration_reported'],
           'audio_stream_duration': audio_end_time, 'video_stream_duration': track_end,
           'video_decoded_frames': len(visual_t), 'openface_timestamp_max_delta_sec': timestamp_delta,
           'video_frame_pts_first_last': [visual_t[0], visual_t[-1]], 'transcript': r['text'], 'audio_feature_names': list(audio.columns),
           'vision_feature_names': names, 'words': mapping}, ensure_ascii=False, indent=2), encoding='utf-8')
        rec = {'id': filename, 'words': len(words), 'dims': [text.shape[1], acoustic.shape[1], vision.shape[1]],
               'audio_observed': int(audio_mask.sum()), 'vision_observed': int(vision_mask.sum()),
               'audio_coverage_mean': float(audio_coverage.mean()), 'vision_coverage_mean': float(vision_coverage.mean()),
               'openface_timestamp_max_delta_sec': timestamp_delta,
               'npz_bytes': (out / (filename + '.npz')).stat().st_size}
        report.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    (out / 'feature-summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    # Small check: a frame outside the word must never count as observed.
    assert aggregate([{'start': 1., 'end': 2.}], [0.], [0.5], [[7.]], [True])[1].tolist() == [0]
    main()
