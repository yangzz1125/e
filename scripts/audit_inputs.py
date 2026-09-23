"""Read-only audit of supplied multimodal inputs; never trains or rewrites data."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import json
from pathlib import Path
import pickle
import subprocess
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import numpy as np
import openpyxl


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            'numpy': {'ndarray', 'dtype', 'asarray'},
            'numpy.core.multiarray': {'_reconstruct', 'scalar'},
            'numpy._core.multiarray': {'_reconstruct', 'scalar'},
            'numpy.core.numeric': {'_frombuffer'},
            'numpy._core.numeric': {'_frombuffer'},
            'builtins': {'set', 'frozenset', 'slice', 'complex'},
            '_codecs': {'encode'},
        }
        if name not in allowed.get(module, set()):
            raise pickle.UnpicklingError(f'Unapproved pickle global: {module}.{name}')
        return getattr(importlib.import_module(module), name)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def attachment(path):
    return next((part.split('-')[0] for part in path.parts if part.startswith('附件')), '题面')


def canonical_path(path):
    parts = path.parts
    start = next((i for i, part in enumerate(parts) if part.startswith('附件')), len(parts) - 1)
    return '/'.join(parts[start:])


def numeric_summary(value):
    a = np.asarray(value)
    result = {'shape': list(a.shape), 'dtype': str(a.dtype)}
    if a.dtype.kind not in 'biuf':
        return result
    bad = 0
    minimum, maximum = None, None
    flat = a.reshape(-1)
    for start in range(0, flat.size, 1_000_000):
        block = flat[start:start + 1_000_000]
        finite = np.isfinite(block)
        bad += int((~finite).sum())
        if finite.any():
            low, high = float(block[finite].min()), float(block[finite].max())
            minimum = low if minimum is None else min(minimum, low)
            maximum = high if maximum is None else max(maximum, high)
    result.update(nonfinite=bad, minimum=minimum, maximum=maximum)
    if a.ndim >= 2:
        result['all_zero_feature_rows'] = sum(
            int(np.all(a[i:i + 32] == 0, axis=-1).sum()) for i in range(0, len(a), 32))
    return result


def describe(value):
    if isinstance(value, dict):
        return {str(k): describe(v) for k, v in value.items()}
    if isinstance(value, (np.ndarray, list, tuple)):
        try:
            return numeric_summary(value)
        except (ValueError, TypeError):
            return {'type': type(value).__name__, 'length': len(value), 'ragged': True}
    return {'type': type(value).__name__}


def rows_from_sheet(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    summaries = []
    labels = []
    for sheet in wb.worksheets:
        iterator = sheet.iter_rows(values_only=True)
        headers = list(next(iterator, ()))
        rows = [dict(zip(headers, row)) for row in iterator if any(x is not None for x in row)]
        info = {'sheet': sheet.title, 'rows': len(rows), 'headers': headers,
                'range': sheet.calculate_dimension()}
        if {'video_id', 'clip_id', 'label'}.issubset(headers):
            labels = rows
            keys = [(str(r['video_id']), normalize_clip(r['clip_id'])) for r in rows]
            info['duplicate_keys'] = len(keys) - len(set(keys))
            info['empty_texts'] = sum(not str(r.get('text') or '').strip() for r in rows)
            info['label_storage_types'] = dict(Counter(type(r['label']).__name__ for r in rows))
            info['numeric_string_labels'] = sum(isinstance(r['label'], str) and valid_label(r['label'], r.get('annotation')) for r in rows)
            invalid = []
            for index, row in enumerate(rows, 2):
                if not valid_label(row['label'], row.get('annotation')):
                    invalid.append(index)
            info['invalid_label_rows'] = invalid
            if 'mode' in headers:
                info['split_counts'] = dict(Counter(str(row['mode']) for row in rows))
        summaries.append(info)
    wb.close()
    return summaries, labels


def valid_label(value, annotation):
    try:
        score = float(value)
    except (ValueError, TypeError):
        return False
    if not np.isfinite(score) or not -3 <= score <= 3:
        return False
    expected = 'negative' if score < 0 else 'positive' if score > 0 else 'neutral'
    return str(annotation).strip().lower() == expected


def normalize_clip(value):
    if isinstance(value, (int, float)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def parse_id(value):
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 2:
        return str(value[0]), normalize_clip(value[1])
    if isinstance(value, (str, np.str_)) and '$_$' in value:
        video, clip = str(value).split('$_$', 1)
        return video, normalize_clip(clip)
    return None


def token_audit(split, tokenizer):
    a = np.asarray(split['text_bert'])
    if a.ndim != 3 or a.shape[1] != 3:
        return {'error': 'Unexpected text_bert shape', 'shape': list(a.shape)}
    raw = [str(x) for x in split['raw_text']]
    encoded = tokenizer(raw, padding='max_length', truncation=True, max_length=a.shape[-1], return_tensors='np')
    expected = np.stack([encoded['input_ids'], encoded['attention_mask'], encoded['token_type_ids']], axis=1)
    matches = np.all(a == expected, axis=(1, 2))
    observed = a[:, 1, :] == 1
    valid_ids = a[:, 0, :]
    return {
        'samples': len(a), 'all_three_channels_exact': int(matches.sum()),
        'input_ids_exact': int(np.all(a[:, 0] == expected[:, 0], axis=1).sum()),
        'attention_mask_exact': int(np.all(a[:, 1] == expected[:, 1], axis=1).sum()),
        'token_type_ids_exact': int(np.all(a[:, 2] == expected[:, 2], axis=1).sum()),
        'first_mismatch_indices': np.flatnonzero(~matches)[:20].tolist(),
        'noninteger_entries': int(np.count_nonzero(a != np.floor(a))),
        'nonbinary_mask_entries': int(np.count_nonzero((a[:, 1] != 0) & (a[:, 1] != 1))),
        'out_of_vocab_ids': int(np.count_nonzero((valid_ids < 0) | (valid_ids >= tokenizer.vocab_size))),
        'nonzero_ids_under_zero_attention': int(np.count_nonzero(valid_ids[~observed])),
        'observed_lengths_min_max': [int(observed.sum(1).min()), int(observed.sum(1).max())],
    }


def split_audit(data, labels):
    result, ids_by_split, videos_by_split = {}, {}, {}
    label_lookup = {(str(row['video_id']), normalize_clip(row['clip_id'])): row for row in labels}
    for name, split in data.items():
        n = len(split['audio'])
        item = {'samples': n, 'field_length_mismatches': [k for k, v in split.items()
                if isinstance(v, (np.ndarray, list, tuple)) and len(v) != n]}
        parsed = [parse_id(x) for x in split.get('id', [])]
        ids = [x for x in parsed if x is not None]
        ids_by_split[name] = set(ids)
        videos_by_split[name] = {x[0] for x in ids}
        item.update(unparsed_ids=len(parsed) - len(ids), duplicate_ids=len(ids) - len(set(ids)), source_videos=len(videos_by_split[name]))
        if 'regression_labels' in split and 'classification_labels' in split:
            scores = np.asarray(split['regression_labels']).reshape(-1)
            categories = np.asarray(split['classification_labels']).reshape(-1)
            item['class_counts'] = {str(x): int((categories == x).sum()) for x in np.unique(categories)}
            item['class_by_score_sign'] = {str(x): dict(Counter(np.sign(scores[categories == x]).astype(int).tolist())) for x in np.unique(categories)}
            item['scores_out_of_range'] = int(np.count_nonzero((scores < -3) | (scores > 3) | ~np.isfinite(scores)))
            missing, mismatched = [], []
            for i, key in enumerate(parsed):
                if key not in label_lookup:
                    missing.append(i)
                elif not np.isclose(float(label_lookup[key]['label']), scores[i], atol=1e-6):
                    mismatched.append(i)
            item.update(excel_missing_ids=len(missing), excel_score_mismatches=len(mismatched))
        result[name] = item
    overlap = {}
    names = list(ids_by_split)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap[f'{left}/{right}'] = {'sample_ids': len(ids_by_split[left] & ids_by_split[right]),
                                          'source_videos': len(videos_by_split[left] & videos_by_split[right])}
    return {'splits': result, 'overlap': overlap}


def video_audit(path):
    result = {'path': str(path)}
    try:
        probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                                'format=duration:stream=codec_type,codec_name,width,height,sample_rate,channels,nb_frames,avg_frame_rate',
                                '-of', 'json', str(path)], capture_output=True, text=True, timeout=45)
        result['probe_exit'] = probe.returncode
        if probe.returncode:
            result['error'] = probe.stderr[:500]
            return result
        result.update(json.loads(probe.stdout))
        decoded = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-threads', '1', '-i', str(path),
                                  '-map', '0:v:0', '-map', '0:a:0?', '-f', 'null', '-'],
                                 capture_output=True, text=True, timeout=120)
        result['decode_exit'] = decoded.returncode
        if decoded.stderr:
            result['decode_messages'] = decoded.stderr[:1000]
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Retry failed feature reads with unchanged source files.')
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if output == source or source in output.parents:
        raise ValueError('Reports must be outside the input directory')
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in source.rglob('*') if p.is_file())
    inventory = json.loads((output / 'inventory.json').read_text(encoding='utf-8')) if args.resume else []
    if args.resume:
        assert {r['path'] for r in inventory} == {p.relative_to(source).as_posix() for p in files}
        for row in inventory:
            info = (source / row['path']).stat()
            if (info.st_size, info.st_mtime_ns) != (row['size'], row['mtime_ns']):
                raise RuntimeError(f'Input changed since prior audit: {row["path"]}')
    for path in ([] if args.resume else files):
        before = path.stat()
        sha = digest(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Input changed during hashing: {path.name}')
        inventory.append({'path': path.relative_to(source).as_posix(), 'key': canonical_path(path.relative_to(source)),
                          'attachment': attachment(path.relative_to(source)), 'size': after.st_size,
                          'sha256': sha, 'mtime_ns': after.st_mtime_ns})
    write_json(output / 'inventory.json', inventory)
    print(f'Inventory: {len(files)} files, {sum(x["size"] for x in inventory)} bytes', flush=True)
    if args.inventory_only:
        return
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'file_count': len(files),
              'file_counts': dict(Counter(f'{attachment(p)}:{p.suffix}' for p in files)),
              'workbooks': {}, 'features': {}, 'tokenizer': {}, 'issues': []}
    if args.resume:
        report = json.loads((output / 'audit.json').read_text(encoding='utf-8'))
        report['previous_reader_issues'] = report['issues']
        report['issues'] = []
        report['resumed_utc'] = datetime.now(timezone.utc).isoformat()
    labels2 = []
    for path in files:
        if path.suffix == '.xlsx':
            sheets, labels = rows_from_sheet(path)
            report['workbooks'][canonical_path(path)] = sheets
            if attachment(path) == '附件2':
                labels2 = labels
            if attachment(path) == '附件1':
                videos = [p for p in files if p.suffix == '.mp4' and attachment(p) == '附件1']
                video_keys = {(p.parent.name, p.stem) for p in videos}
                row_keys = {(str(r['video_id']), normalize_clip(r['clip_id'])) for r in labels}
                report['attachment1_matching'] = {'video_count': len(videos), 'label_count': len(labels),
                                                  'missing_videos': len(row_keys - video_keys),
                                                  'unlabelled_videos': len(video_keys - row_keys)}
        if path.suffix == '.docx':
            with ZipFile(path) as archive:
                xml = ET.fromstring(archive.read('word/document.xml'))
                paragraphs = [''.join(e.text or '' for e in para.iter() if e.tag.endswith('}t'))
                              for para in xml.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p')]
            (output / 'problem-statement.txt').write_text('\n'.join(paragraphs), encoding='utf-8')
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    for path in files:
        if path.suffix != '.pkl':
            continue
        name = canonical_path(path)
        if args.resume and name in report['features']:
            continue
        print(f'Feature audit: {name}', flush=True)
        try:
            with path.open('rb') as handle:
                data = ArrayUnpickler(handle).load()
                trailing = handle.read(1)
            report['features'][name] = {'schema': describe(data), 'trailing_bytes': bool(trailing)}
            if attachment(path) == '附件2':
                report['features'][name]['integrity'] = split_audit(data, labels2)
                if tokenizer and path.name == 'aligned_50.pkl':
                    for split in ('train', 'valid'):
                        report['tokenizer'][split] = token_audit(data[split], tokenizer)
            del data
            gc.collect()
        except Exception as exc:
            report['issues'].append({'file': name, 'error': f'{type(exc).__name__}: {exc}'})
        write_json(output / 'audit.json', report)
    videos = [p for p in files if p.suffix == '.mp4']
    if args.resume and (output / 'videos.json').exists():
        checked = json.loads((output / 'videos.json').read_text(encoding='utf-8'))
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            checked = list(pool.map(video_audit, videos))
        for row in checked:
            row['path'] = str(Path(row['path']).relative_to(source))
    write_json(output / 'videos.json', checked)
    report['videos'] = {'count': len(checked),
                        'failed': [r['path'] for r in checked if r.get('probe_exit') != 0 or r.get('decode_exit') != 0],
                        'no_audio': [r['path'] for r in checked if not any(s.get('codec_type') == 'audio' for s in r.get('streams', []))]}
    changed = [row['path'] for row in inventory
               if ((source / row['path']).stat().st_size != row['size']
                   or (source / row['path']).stat().st_mtime_ns != row['mtime_ns'])]
    report['source_files_changed_during_audit'] = changed
    write_json(output / 'audit.json', report)
    print(json.dumps({'tokenizer': report['tokenizer'], 'videos': report['videos'],
                      'issues': report['issues'], 'source_files_changed': changed}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
