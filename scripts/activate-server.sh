#!/usr/bin/env bash
# Source this file; tool-specific environments apply only to their commands.
EMOTION_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EMOTION_PROJECT_ROOT
source "$EMOTION_PROJECT_ROOT/.venv/bin/activate"
export HF_HOME="$EMOTION_PROJECT_ROOT/.cache/huggingface"
export MFA_ROOT_DIR="$EMOTION_PROJECT_ROOT/models/mfa"

mfa() {
    env PATH="$EMOTION_PROJECT_ROOT/tools/mfa/bin:$PATH" \
        "$EMOTION_PROJECT_ROOT/tools/mfa/bin/mfa" "$@"
}

FeatureExtraction() {
    env OMP_NUM_THREADS=4 \
        "$EMOTION_PROJECT_ROOT/tools/OpenFace-linux/build/bin/FeatureExtraction" "$@"
}
