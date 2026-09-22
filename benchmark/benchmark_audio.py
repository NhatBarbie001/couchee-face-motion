"""
Audio Module Performance & VRAM Benchmark.
Measures Latency, Real-Time-Factor (RTF), and VRAM for Silero VAD, Aniemore Wav2Vec2, and Prosody.
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

from audio.vad import SileroVADModel
from audio.voice_emotion import VoiceEmotionModel
from audio.prosody import ProsodyAnalyzer


def get_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Benchmark Audio Models on GPU.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--duration-sec", type=float, default=10.0)
    args = parser.parse_args()

    print("=" * 70)
    print(f"       BENCHMARKING AUDIO MODULE (Device: {args.device}, Audio Length: {args.duration_sec}s)")
    print("=" * 70)

    # 10 seconds of synthetic audio at 16kHz
    sr = 16000
    dummy_wav = (np.sin(2 * np.pi * 220 * np.linspace(0, args.duration_sec, int(sr * args.duration_sec))) * 0.5).astype(np.float32)
    dummy_chunk_512 = dummy_wav[:512]

    table = []

    # 1. Silero VAD
    vram_before = get_vram_mb()
    vad = SileroVADModel(device=args.device)
    t0 = time.perf_counter()
    for _ in range(100):
        vad.predict_chunk(dummy_chunk_512)
    elapsed_512 = (time.perf_counter() - t0) / 100
    rtf_vad = elapsed_512 / (512 / 16000.0)
    vram_vad = max(0.0, get_vram_mb() - vram_before)
    table.append(["Silero VAD", f"{rtf_vad:.4f}x RTF", f"{elapsed_512 * 1000:.2f} ms", f"{vram_vad:.1f} MB", "Ultra Fast"])

    # 2. Aniemore Wav2Vec2 Voice Emotion
    vram_before = get_vram_mb()
    emo = VoiceEmotionModel(device=args.device)
    speech_segment = dummy_wav[: sr * 3]  # 3-second speech segment
    t0 = time.perf_counter()
    emo.predict_segment(speech_segment)
    elapsed_emo = time.perf_counter() - t0
    rtf_emo = elapsed_emo / 3.0
    vram_emo = max(0.0, get_vram_mb() - vram_before)
    table.append(["Aniemore Wav2Vec2", f"{rtf_emo:.4f}x RTF", f"{elapsed_emo * 1000:.1f} ms", f"{vram_emo:.1f} MB", "SOTA Crosslingual"])

    # 3. Prosody Extraction
    pros = ProsodyAnalyzer(sample_rate=sr)
    t0 = time.perf_counter()
    pros.analyze(
        waveform=dummy_wav,
        speech_segments=[{"start": 0.0, "end": args.duration_sec, "duration": args.duration_sec, "speech": True}],
        pause_ratio=0.1,
        total_duration_sec=args.duration_sec
    )
    elapsed_pros = time.perf_counter() - t0
    rtf_pros = elapsed_pros / args.duration_sec
    table.append(["Prosody (F0/Energy/Rate)", f"{rtf_pros:.4f}x RTF", f"{elapsed_pros * 1000:.1f} ms", "0.0 MB (CPU)", "Acoustic Gold"])

    headers = ["Model / Submodule", "Real-Time Factor (RTF)", "Execution Time", "VRAM Footprint", "Quality Tier"]
    print(tabulate(table, headers=headers, tablefmt="github"))
    print("\n[INFO] Benchmark completed successfully.")


if __name__ == "__main__":
    main()
