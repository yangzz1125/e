"""Verify the project runtime without downloading model weights."""

from __future__ import annotations

import importlib
import argparse
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path


REQUIRED_MODULES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "scikit-learn": "sklearn",
    "openpyxl": "openpyxl",
    "soundfile": "soundfile",
    "opensmile": "opensmile",
    "opencv-contrib-python": "cv2",
    "mediapipe": "mediapipe",
    "transformers": "transformers",
    "scienceplots": "scienceplots",
    "adjustText": "adjustText",
    "psutil": "psutil",
    "pyyaml": "yaml",
    "tqdm": "tqdm",
    "pytest": "pytest",
}

DEEP_MODULES = {
    "torch": "torch", "torchvision": "torchvision", "torchaudio": "torchaudio",
    "accelerate": "accelerate", "captum": "captum",
}


def package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def check_imports(include_deep: bool) -> list[str]:
    failures: list[str] = []
    modules = REQUIRED_MODULES | (DEEP_MODULES if include_deep else {})
    for distribution, module in modules.items():
        try:
            importlib.import_module(module)
            print(f"[OK] {distribution}=={package_version(distribution)}")
        except Exception as exc:  # Keep checking so one failure does not hide others.
            failures.append(f"{distribution}: {type(exc).__name__}: {exc}")
            print(f"[FAIL] {failures[-1]}")
    return failures


def check_torch(cuda_build: str) -> list[str]:
    failures: list[str] = []
    try:
        import torch
        import torchaudio
        import torchvision

        versions = (torch.__version__, torchvision.__version__, torchaudio.__version__)
        if not all(f"+{cuda_build}" in version for version in versions):
            failures.append(f"PyTorch builds are not consistently {cuda_build}: {versions}")

        if not torch.cuda.is_available():
            failures.append("torch.cuda.is_available() is False")
        else:
            left = torch.randn((256, 256), device="cuda")
            right = torch.randn((256, 256), device="cuda")
            result = left @ right
            torch.cuda.synchronize()
            if not bool(torch.isfinite(result).all()):
                failures.append("CUDA matrix multiplication returned a non-finite value")
            if bool(torch.isfinite(result).all()):
                print(
                    "[OK] CUDA matrix multiplication; "
                    f"device={torch.cuda.get_device_name(0)!r}, cuda={torch.version.cuda}"
                )
            layer = torch.nn.Linear(8, 1).cuda()
            optimizer = torch.optim.AdamW(layer.parameters())
            loss = layer(torch.ones((2, 8), device='cuda')).square().mean()
            loss.backward()
            optimizer.step()
            assert all(torch.isfinite(parameter).all() for parameter in layer.parameters())
            print('[OK] CUDA backward and optimizer step')
        from transformers import BertConfig, BertModel
        model = BertModel(BertConfig(vocab_size=128, hidden_size=32,
                                    num_hidden_layers=1, num_attention_heads=4,
                                    intermediate_size=64)).eval()
        with torch.no_grad():
            output = model(torch.tensor([[1, 2, 3]])).last_hidden_state
        assert output.shape == (1, 3, 32) and torch.isfinite(output).all()
        print("[OK] Random tiny BERT forward pass (no pretrained weights)")
        from captum.attr import IntegratedGradients
        inputs = torch.tensor([[1.0, 2.0, 3.0]])
        attributions = IntegratedGradients(lambda value: value.sum(dim=1)).attribute(inputs)
        assert torch.allclose(attributions, inputs)
        audio = torchaudio.functional.resample(torch.ones(16000), 16000, 8000)
        assert audio.shape == (8000,) and torch.isfinite(audio).all()
        selected = torchvision.ops.nms(torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]]),
                                       torch.tensor([0.9, 0.8]), 0.5)
        assert selected.tolist() == [0]
        print("[OK] Captum IG; torchaudio resampling; torchvision NMS")
    except Exception as exc:
        failures.append(f"PyTorch runtime: {type(exc).__name__}: {exc}")
    return failures


def check_commands() -> list[str]:
    failures: list[str] = []
    for command in ("ffmpeg", "ffprobe"):
        executable = shutil.which(command)
        if executable is None:
            failures.append(f"{command} is not on PATH")
            continue
        completed = subprocess.run(
            [executable, "-version"],
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            failures.append(f"{command} -version exited with {completed.returncode}")
        else:
            first_line = completed.stdout.splitlines()[0]
            print(f"[OK] {command}: {first_line}")
    return failures


def check_features() -> list[str]:
    try:
        import numpy as np
        import cv2
        from mediapipe.tasks.python.vision import FaceLandmarker
        from sklearn.linear_model import Ridge
        from opensmile_compat import egemaps_lld

        signal = (0.2 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype(np.float32)
        with egemaps_lld() as extractor:
            features = extractor.process_signal(signal, 16000)
        assert len(features) > 0 and features.shape[1] == 25
        assert np.isfinite(features.to_numpy()).all()
        gray = cv2.cvtColor(np.zeros((32, 32, 3), np.uint8), cv2.COLOR_BGR2GRAY)
        assert gray.shape == (32, 32)
        model = Ridge().fit(np.arange(20).reshape(10, 2), np.arange(10))
        assert np.isfinite(model.predict([[2, 3]])).all()
        print(f"[OK] openSMILE synthetic audio: {features.shape}; OpenCV; Ridge")
        print("[OK] MediaPipe FaceLandmarker API; pretrained model NOT tested")
        return []
    except Exception as exc:
        return [f"Feature smoke test: {type(exc).__name__}: {exc}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-only', action='store_true',
                        help='Check only base tools; does not certify GPU readiness.')
    parser.add_argument('--models', action='store_true', help='Also check the downloaded MediaPipe model.')
    parser.add_argument('--cuda-build', default='cu128', help='Expected PyTorch build (server: cu121).')
    args = parser.parse_args()
    print(f"Python {platform.python_version()} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    failures = check_imports(not args.base_only) + check_commands() + check_features()
    if not args.base_only:
        failures += check_torch(args.cuda_build)
    if args.models:
        try:
            import mediapipe as mp
            import numpy as np
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
            model_path = Path(__file__).resolve().parents[1] / 'models/mediapipe/face_landmarker.task'
            options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model_path)),
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True)
            with FaceLandmarker.create_from_options(options) as detector:
                result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                   data=np.zeros((256, 256, 3), np.uint8)))
            assert not result.face_landmarks
            print('[OK] MediaPipe model load and blank image inference')
        except Exception as exc:
            failures.append(f'MediaPipe model: {type(exc).__name__}: {exc}')
    if failures:
        print("\nEnvironment check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nBase environment check passed; GPU not checked." if args.base_only
          else "\nFull environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
