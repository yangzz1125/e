"""Small runnable check for the full-audio offset calculation."""
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf

from align_words import audio_match


def test_offset():
    sample_rate = 16000
    offset = 6.573125
    rng = np.random.default_rng(3)
    cropped = rng.normal(0, .05, sample_rate * 2).astype('float32')
    full = np.zeros(sample_rate * 10, dtype='float32')
    i = round(offset * sample_rate)
    full[i:i + len(cropped)] = cropped
    with tempfile.TemporaryDirectory() as directory:
        a, b = Path(directory) / 'edited.wav', Path(directory) / 'full.wav'
        sf.write(a, cropped, sample_rate)
        sf.write(b, full, sample_rate)
        result = audio_match(a, b, offset)
    assert result['status'] == 'pass', result
    assert abs(result['offset_error_sec']) < 1 / sample_rate, result


if __name__ == '__main__':
    test_offset()
    print('PASS audio edit offset self-check')
