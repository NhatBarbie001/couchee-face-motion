#!/usr/bin/env python3
"""
Master CLI Runner for Face-motion-gpu (High-Throughput Offline Multimodal Sale AI Pipeline).

Usage:
    python run_pipeline.py --video "C:\\Users\\loidi\\Downloads\\tiktok1.mp4" --output-dir "results/sale_run_1" --device cuda --batch-size 32
"""

import os
import sys
import argparse
import time

# Windows UTF-8 stdout setup
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from service import MultimodalEvaluatorService


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run High-Throughput Offline Multimodal Sale AI Assessment (GPU Accelerated)."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to student sales roleplay video file (.mp4, .mov, etc.)"
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to separate audio file (.wav). If omitted, audio will be extracted automatically from --video."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/session_run",
        help="Directory where evaluation reports (JSON + Markdown) will be saved."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device ('cuda' for NVIDIA GPU, 'cpu' for CPU fallback)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of video frames per GPU batch (default: 32 for RTX 3060 12GB)."
    )
    parser.add_argument(
        "--step",
        type=int,
        default=2,
        help="Frame sampling step (default: 2, 2x speedup)."
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Optional session identifier (defaults to media filename)."
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable generating annotated output video"
    )
    parser.add_argument(
        "--no-hud",
        action="store_true",
        help="Disable HUD dashboard overlay on output video"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    session_id = args.session_id or os.path.splitext(os.path.basename(args.video))[0]
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("      FACE-MOTION-GPU: HIGH-THROUGHPUT OFFLINE SALE AI EVALUATOR")
    print("=" * 80)
    print(f"Session ID       : {session_id}")
    print(f"Input Video      : {args.video}")
    print(f"Output Directory : {args.output_dir}")
    print(f"Target Device    : {args.device}")
    print(f"GPU Batch Size   : {args.batch_size}")
    print(f"Frame Step       : {args.step}")
    print(f"Save Video       : {not args.no_video}")
    print(f"Show HUD Overlay : {not args.no_hud}")
    print("=" * 80)

    if not os.path.exists(args.video):
        print(f"[ERROR] Video file not found: {args.video}")
        sys.exit(1)

    evaluator_service = MultimodalEvaluatorService.get_instance(device=args.device)

    result = evaluator_service.evaluate(
        video_path=args.video,
        audio_path=args.audio,
        output_dir=args.output_dir,
        session_id=session_id,
        batch_size=args.batch_size,
        step=args.step,
        save_video=not args.no_video,
        show_hud=not args.no_hud,
        show_progress=True
    )

    perf = result.get("performance", {})
    overall = result.get("overall_evaluation", {})
    scores = overall.get("breakdown", {})
    meta = result.get("metadata", {})
    kbe = result.get("key_behavioral_events", {})

    print("\n" + "=" * 80)
    print(f"                         EVALUATION SUMMARY: {overall.get('grade', 'N/A')}")
    print("=" * 80)
    print(f"Total Score         : {overall.get('total_score', 0)} / 100")
    print(f"1. Confidence       : {scores.get('confidence_score', 0)} / 25")
    print(f"2. Active Listening : {scores.get('active_listening_score', 0)} / 25")
    print(f"3. Vocal Dynamism   : {scores.get('vocal_dynamism_score', 0)} / 25")
    print(f"4. Facial Warmth    : {scores.get('facial_warmth_score', 0)} / 25")
    print("-" * 80)
    if kbe.get("longest_distraction") and kbe["longest_distraction"].get("duration_seconds", 0) > 0:
        ld = kbe["longest_distraction"]
        print(f"Longest Distraction : {ld['duration_seconds']}s (Lý do: {ld.get('reason')} từ {ld.get('start_time')}s đến {ld.get('end_time')}s)")
    print(f"Turns Analyzed      : {meta.get('total_turns', 0)} turns ({meta.get('speaking_turns', 0)} speaking, {meta.get('listening_turns', 0)} listening)")
    if result.get("files", {}).get("annotated_video"):
        print(f"Annotated Video     : {result['files']['annotated_video']}")
    if result.get("files", {}).get("json_report"):
        print(f"Detailed JSON       : {result['files']['json_report']}")
    if result.get("files", {}).get("markdown_report"):
        print(f"Executive Report    : {result['files']['markdown_report']}")
    print("-" * 80)
    print(f"Total Execution Time: {perf.get('execution_time_sec', 0)}s (Throughput: {perf.get('realtime_multiplier', 'N/A')} Real-time)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
