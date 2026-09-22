"""
Vision Module Performance & VRAM Benchmark.
Measures FPS, Latency (ms), and VRAM usage on GPU/CPU for SCRFD, EmotiEffLib, 6DRepNet, and Gaze-LLE.
"""

import time
import argparse
import numpy as np
import torch
import sys
import os

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(rows, headers, tablefmt=""):
        res = [" | ".join(headers), "-" * 60]
        for r in rows:
            res.append(" | ".join(str(c) for c in r))
        return "\n".join(res)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from vision.face_detection import SCRFDDetector
from vision.face_emotion import FaceEmotionModel
from vision.head_pose import HeadPose6DRepNetModel
from vision.eye_gaze import GazeLLEModel


def get_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Benchmark Vision Models on GPU.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    print("=" * 70)
    print(f"       BENCHMARKING VISION MODULE (Device: {args.device}, Batch: {args.batch_size})")
    print("=" * 70)

    # Synthetic dummy images for testing throughput
    dummy_frame_640 = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    dummy_crops_224 = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(args.batch_size)]
    dummy_poses = [{"pitch": 0.0, "yaw": 0.0, "roll": 0.0} for _ in range(args.batch_size)]

    table = []

    # 1. SCRFD Face Detection
    vram_before = get_vram_mb()
    det = SCRFDDetector(device=args.device)
    # Warmup
    det.detect_single(dummy_frame_640)
    t0 = time.perf_counter()
    for _ in range(args.iterations):
        det.detect_single(dummy_frame_640)
    lat_scrfd = ((time.perf_counter() - t0) / args.iterations) * 1000
    fps_scrfd = 1000.0 / max(0.001, lat_scrfd)
    vram_scrfd = max(0.0, get_vram_mb() - vram_before)
    table.append(["SCRFD Face Detect", f"{fps_scrfd:.1f}", f"{lat_scrfd:.1f} ms", f"{vram_scrfd:.1f} MB", "High"])

    # 2. EmotiEffLib Face Emotion
    vram_before = get_vram_mb()
    emo = FaceEmotionModel(device=args.device)
    emo.predict_batch(dummy_crops_224)
    t0 = time.perf_counter()
    for _ in range(args.iterations):
        emo.predict_batch(dummy_crops_224)
    lat_emo = ((time.perf_counter() - t0) / args.iterations) * 1000
    fps_emo = (args.batch_size * 1000.0) / max(0.001, lat_emo)
    vram_emo = max(0.0, get_vram_mb() - vram_before)
    table.append(["EmotiEffLib (Batch)", f"{fps_emo:.1f}", f"{lat_emo:.1f} ms", f"{vram_emo:.1f} MB", "SOTA"])

    # 3. 6DRepNet Head Pose
    vram_before = get_vram_mb()
    hp = HeadPose6DRepNetModel(device=args.device)
    hp.predict_batch(dummy_crops_224)
    t0 = time.perf_counter()
    for _ in range(args.iterations):
        hp.predict_batch(dummy_crops_224)
    lat_hp = ((time.perf_counter() - t0) / args.iterations) * 1000
    fps_hp = (args.batch_size * 1000.0) / max(0.001, lat_hp)
    vram_hp = max(0.0, get_vram_mb() - vram_before)
    table.append(["6DRepNet (Batch)", f"{fps_hp:.1f}", f"{lat_hp:.1f} ms", f"{vram_hp:.1f} MB", "High"])

    # 4. Gaze-LLE
    vram_before = get_vram_mb()
    gaze = GazeLLEModel(device=args.device)
    gaze.predict_batch(dummy_crops_224, dummy_poses)
    t0 = time.perf_counter()
    for _ in range(args.iterations):
        gaze.predict_batch(dummy_crops_224, dummy_poses)
    lat_gaze = ((time.perf_counter() - t0) / args.iterations) * 1000
    fps_gaze = (args.batch_size * 1000.0) / max(0.001, lat_gaze)
    vram_gaze = max(0.0, get_vram_mb() - vram_before)
    table.append(["Gaze-LLE (Batch)", f"{fps_gaze:.1f}", f"{lat_gaze:.1f} ms", f"{vram_gaze:.1f} MB", "High"])

    headers = ["Model", "Throughput (FPS)", "Batch Latency", "VRAM Footprint", "Quality"]
    print(tabulate(table, headers=headers, tablefmt="github"))
    print("\n[INFO] Benchmark completed successfully.")


if __name__ == "__main__":
    main()
