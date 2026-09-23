"""Compare six train/validation records with the frozen, selected BERT encoder."""
import argparse
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel
from audit_inputs import ArrayUnpickler, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
paths = list(args.input.rglob('aligned_50.pkl'))
assert len(paths) == 1
with paths[0].open('rb') as source:
    data = ArrayUnpickler(source).load()
torch.backends.cuda.matmul.allow_tf32 = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModel.from_pretrained(str(args.model), local_files_only=True,
                                  use_safetensors=True, attn_implementation='eager').to(device).eval()
results = []
for split_name in ('train', 'valid'):
    split = data[split_name]
    for index in (0, len(split['text_bert']) // 2, len(split['text_bert']) - 1):
        tokens = np.asarray(split['text_bert'][index])
        inputs = {name: torch.as_tensor(tokens[channel:channel + 1], dtype=torch.long, device=device)
                  for channel, name in enumerate(('input_ids', 'attention_mask', 'token_type_ids'))}
        with torch.inference_mode():
            actual = model(**inputs).last_hidden_state[0].cpu().numpy()
        expected = np.asarray(split['text'][index])
        assert actual.shape == expected.shape and np.isfinite(actual).all()
        observed = tokens[1] == 1
        difference = np.abs(actual - expected)
        results.append({'split': split_name, 'sample_index': index,
                        'shape': list(actual.shape),
                        'nonpadding_max_absolute_difference': float(difference[observed].max()),
                        'nonpadding_mean_absolute_difference': float(difference[observed].mean()),
                        'nonpadding_allclose_atol_1e-4_rtol_1e-4': bool(np.allclose(actual[observed], expected[observed], atol=1e-4, rtol=1e-4))})
        print(results[-1], flush=True)
write_json(args.output, {'device': device, 'model': 'google-bert/bert-base-uncased',
                        'revision': '86b5e0934494bd15c9632b12f734a8a67f723594',
                        'sample_selection': 'first, middle, last of train and valid; test excluded',
                        'results': results,
                        'limitation': 'Six input-feature comparisons do not establish full historical encoder provenance.'})
