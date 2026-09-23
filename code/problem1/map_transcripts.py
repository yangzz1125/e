"""Map the supplied transcript's MP4 edit-list segment onto original full-media PTS.

No word alignment is asserted by this metadata-only pass. Output remains a candidate
until the cropped and full audio are independently matched and heard by a human.
"""
import argparse
import csv
import json
from pathlib import Path
import struct

ROOT = Path('/hy-tmp/E/problem1/media')


def boxes(buf, start=0, end=None):
    if end is None:
        end = len(buf)
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack_from('>I4s', buf, pos)
        header = 8
        if size == 1:
            if pos + 16 > end:
                raise ValueError('truncated extended MP4 box')
            size = struct.unpack_from('>Q', buf, pos + 8)[0]
            header = 16
        if size == 0:
            size = end - pos
        if size < header or pos + size > end:
            raise ValueError('invalid MP4 box length')
        yield kind.decode('ascii', 'replace'), pos + header, pos + size
        pos += size
    if pos != end:
        raise ValueError('trailing incomplete MP4 box')


def child(buf, lo, hi, kind):
    return [pair for name, *pair in boxes(buf, lo, hi) if name == kind]


def time_scale(buf, lo, name):
    version = buf[lo]
    offset = lo + (20 if version == 1 else 12)
    if version not in (0, 1):
        raise ValueError('unknown ' + name + ' version')
    scale = struct.unpack_from('>I', buf, offset)[0]
    if not scale:
        raise ValueError('zero ' + name + ' timescale')
    return scale


def edit_entries(buf, lo, movie_scale, media_scale):
    version, count = buf[lo], struct.unpack_from('>I', buf, lo + 4)[0]
    if version not in (0, 1):
        raise ValueError('unknown edit-list version')
    out = []
    off = lo + 8
    for _ in range(count):
        if version == 0:
            duration, media, rate_i, rate_f = struct.unpack_from('>Ii hh', buf, off)
            off += 12
        else:
            duration, media, rate_i, rate_f = struct.unpack_from('>Qq hh', buf, off)
            off += 20
        out.append({'movie_duration_sec': duration / movie_scale,
                    'source_start_sec': None if media < 0 else media / media_scale,
                    'rate': [rate_i, rate_f]})
    return out


def inspect(path):
    buf = path.read_bytes()
    moov = child(buf, 0, len(buf), 'moov')
    if len(moov) != 1:
        raise ValueError('not one movie header')
    lo, hi = moov[0]
    movie_scale = time_scale(buf, child(buf, lo, hi, 'mvhd')[0][0], 'mvhd')
    tracks = {}
    for t_lo, t_hi in child(buf, lo, hi, 'trak'):
        mdia = child(buf, t_lo, t_hi, 'mdia')[0]
        mdhd = child(buf, *mdia, 'mdhd')[0]
        hdlr = child(buf, *mdia, 'hdlr')[0]
        kind = buf[hdlr[0] + 8:hdlr[0] + 12]
        if kind not in (b'soun', b'vide'):
            continue
        name = 'audio' if kind == b'soun' else 'video'
        scale = time_scale(buf, mdhd[0], 'mdhd')
        edts = child(buf, t_lo, t_hi, 'edts')
        elst = child(buf, *edts[0], 'elst') if edts else []
        tracks[name] = {'media_timescale': scale, 'movie_timescale': movie_scale,
                        'edits': edit_entries(buf, elst[0][0], movie_scale, scale) if elst else []}
    if set(tracks) != {'audio', 'video'}:
        raise ValueError('audio/video track not found')
    return tracks


def main(media_dir):
    with (media_dir / 'manifest.csv').open(encoding='utf-8', newline='') as f:
        manifest = list(csv.DictReader(f))
    if len(manifest) != 100 or any(row['state'] != 'ok' for row in manifest):
        raise ValueError('media manifest not complete')
    report = []
    for row in manifest:
        info = {'sample_id': row['sample_id'], 'transcript': row['raw_text'], 'source': row['source']}
        try:
            tracks = inspect(Path(row['source']))
            info['tracks'] = tracks
            audio_edits = tracks['audio']['edits']
            video_edits = tracks['video']['edits']
            if len(audio_edits) != 1 or len(video_edits) != 1:
                info['state'] = 'review_multiple_or_absent_edits'
            elif any(e['source_start_sec'] is None or e['rate'] != [1, 0] for e in (audio_edits[0], video_edits[0])):
                info['state'] = 'review_nonunit_or_empty_edit'
            else:
                audio, video = audio_edits[0], video_edits[0]
                info['candidate_audio_interval_sec'] = [audio['source_start_sec'], audio['source_start_sec'] + audio['movie_duration_sec']]
                info['candidate_video_interval_sec'] = [video['source_start_sec'], video['source_start_sec'] + video['movie_duration_sec']]
                info['state'] = 'candidate_only_unverified_audio'
        except Exception as exc:
            info['state'] = 'error'
            info['error'] = str(exc)[:250]
        report.append(info)
    dest = media_dir / 'transcript-segments.json'
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    totals = {s: sum(r['state'] == s for r in report) for s in sorted({r['state'] for r in report})}
    print('STATUS', totals, 'OUTPUT', dest)
    for row in report:
        if row['state'] != 'candidate_only_unverified_audio':
            print('REVIEW', row['sample_id'], row['state'], row.get('error', ''))
    if any(r['state'] == 'error' for r in report):
        raise SystemExit('MP4 parsing errors; do not batch-align')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media-dir', type=Path, default=ROOT, help='output of preprocess_media.py')
    main(parser.parse_args().media_dir)
