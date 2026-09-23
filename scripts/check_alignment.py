"""Verify the installed MFA stack using generated speech, not contest data."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
work = ROOT / "reports" / "mfa-smoke"
corpus = work / "corpus" / "speaker"
corpus.mkdir(parents=True, exist_ok=True)
text = "This is a simple test of speech alignment."
(corpus / "sample.lab").write_text(text, encoding="utf-8")
subprocess.run(["espeak-ng", "-v", "en-us", "-s", "140", "-w",
                str(work / "synthetic.wav"), text], check=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(work / "synthetic.wav"),
                "-ar", "16000", "-ac", "1", str(corpus / "sample.wav")], check=True)
environment = os.environ.copy()
environment["PATH"] = str(ROOT / "tools/mfa/bin") + os.pathsep + environment["PATH"]
environment["MFA_ROOT_DIR"] = str(ROOT / "models/mfa")
environment["OMP_NUM_THREADS"] = "2"
subprocess.run([str(ROOT / "tools/mfa/bin/mfa"), "align", str(work / "corpus"),
                str(ROOT / "models/mfa/english_us_arpa.dict"),
                str(ROOT / "models/mfa/english_us_arpa.zip"), str(work / "aligned"),
                "--single_speaker", "--num_jobs", "1", "--clean"],
               env=environment, check=True)
from praatio import textgrid

outputs = list((work / "aligned").rglob("*.TextGrid"))
assert len(outputs) == 1, outputs
grid = textgrid.openTextgrid(str(outputs[0]), includeEmptyIntervals=False)
words = [entry for name in grid.tierNames if "word" in name
         for entry in grid.getTier(name).entries if entry.label.strip()]
assert [entry.label for entry in words] == text.lower().rstrip(".").split(), words
assert all(0 <= entry.start < entry.end <= grid.maxTimestamp for entry in words)
assert all(left.end <= right.start + 1e-6 for left, right in zip(words, words[1:]))
print("MFA synthetic speech alignment PASS:", [(w.label, w.start, w.end) for w in words])
print("Synthetic smoke test only; real-data timing accuracy remains to be evaluated.")
