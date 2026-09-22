"""
Model Weights Manager & Downloader for Face-motion-gpu.
Downloads or links pre-trained ONNX and PyTorch weights for:
- SCRFD Face Detection (ONNX)
- 6DRepNet Head Pose (ONNX)
- Silero VAD (ONNX)
- Gaze-LLE ViT-B (PyTorch / HuggingFace)
- Aniemore Wav2Vec2 Crosslingual (HuggingFace)
"""

import os
import shutil
import urllib.request
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
FACE_MOTION_LEGACY_CACHE = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "Face-motion", "models_cache")
)

# Download links for model weights
MODEL_URLS = {
    "sixdrepnet.onnx": "https://github.com/AscTr/6DRepNet/releases/download/v0.1/6DRepNet_300W_LP.onnx",
    "silero_vad.onnx": "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
    "scrfd_2.5g_kps.onnx": "https://huggingface.co/RuteNL/SCRFD-face-detection-ONNX/resolve/main/2.5g_bnkps.onnx",
}


def download_file(url: str, target_path: str):
    """Download a file with progress output."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        print(f"[CACHE HIT] {os.path.basename(target_path)} already exists.")
        return

    print(f"[DOWNLOADING] {os.path.basename(target_path)} from {url}...")
    try:
        urllib.request.urlretrieve(url, target_path)
        print(f"[DONE] Saved to {target_path}")
    except Exception as e:
        print(f"[WARNING] Could not auto-download from {url}: {e}")


def get_model_path(model_filename: str) -> str:
    """
    Resolves the model path with the following precedence:
    1. Local Face-motion-gpu/models/checkpoints/
    2. Legacy Face-motion/models_cache/
    3. Auto-download from remote release
    """
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    local_path = os.path.join(CHECKPOINTS_DIR, model_filename)

    # 1. Check local checkpoints
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    # 2. Check legacy cache from Face-motion
    legacy_path = os.path.join(FACE_MOTION_LEGACY_CACHE, model_filename)
    if os.path.exists(legacy_path) and os.path.getsize(legacy_path) > 0:
        print(f"[REUSE] Found {model_filename} in legacy cache: {legacy_path}")
        try:
            shutil.copy2(legacy_path, local_path)
            return local_path
        except Exception:
            return legacy_path

    # 3. Attempt download if known URL
    if model_filename in MODEL_URLS:
        download_file(MODEL_URLS[model_filename], local_path)
        if os.path.exists(local_path):
            return local_path

    return local_path


if __name__ == "__main__":
    print("Pre-fetching base models...")
    for model_name in MODEL_URLS:
        p = get_model_path(model_name)
        print(f"-> {model_name}: {p}")
