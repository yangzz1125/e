"""Whisper ASR diagnostic for selected Problem 1 clips. Never changes provided transcripts.

Base.en ASR output is NOT ground truth or word alignment. Run only on local audio.
"""
import argparse
import hashlib
import json
from pathlib import Path

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main(args):
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite diagnostic output: ' + str(args.output))
    rows = {r['sample_id']: r for r in json.loads(args.alignment.read_text(encoding='utf-8'))}
    ids = [x.strip() for x in args.ids_file.read_text(encoding='utf-8').splitlines() if x.strip()]
    if len(ids) != len(set(ids)) or not ids or not set(ids) <= set(rows):
        raise ValueError('duplicate or unknown sample IDs')
    processor = WhisperProcessor.from_pretrained(str(args.model_dir), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(str(args.model_dir),
                                    local_files_only=True, use_safetensors=True).eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    output = []
    for sid in ids:
        row = rows[sid]
        source_run = Path(row.get('provenance', {}).get('retry_source',
                     args.alignment.parent / 'all' / 'alignment.json'))
        folder = source_run.parent / 'corpus' / row['speaker']
        cropped = folder / (row['base'] + '.wav')
        paths = [('edited_audio', cropped)]
        if sid in args.full_audio_ids:
            paths.append(('full_audio', args.media_dir / sid / 'full-16k-mono.wav'))
        for kind, path in paths:
            waveform, sr = sf.read(path, dtype='float32')
            if sr != 16000 or waveform.ndim != 1 or not 0 < len(waveform) <= 30 * sr:
                raise ValueError('expected 16k mono <=30s: ' + str(path))
            prepared = processor(waveform, sampling_rate=sr, return_tensors='pt',
                                 return_attention_mask=True)
            with torch.inference_mode():
                ids_out = model.generate(prepared.input_features.to(device),
                    attention_mask=prepared.attention_mask.to(device),
                    max_new_tokens=128, num_beams=1)
            text = processor.batch_decode(ids_out, skip_special_tokens=True)[0].strip()
            entry = {'sample_id': sid, 'audio_view': kind, 'audio_sha256': sha256(path),
                     'duration_sec': round(len(waveform) / sr, 6), 'provided_text': row['raw_text'],
                     'asr_text': text, 'model': 'openai/whisper-base.en',
                     'revision': '911407f4214e0e1d82085af863093ec0b66f9cd6',
                     'note': 'ASR diagnostic only; not a correction or a word timestamp'}
            output.append(entry)
            print(sid, kind, 'ASR:', text, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print('OUTPUT',args.output,'records',len(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment', type=Path, required=True)
    parser.add_argument('--media-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--ids-file', type=Path, required=True)
    parser.add_argument('--full-audio-ids', nargs='*', default=[])
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
