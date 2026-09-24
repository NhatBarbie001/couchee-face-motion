"""
Unit & Integration Test for Face-motion-gpu Auto-Turn Sensory Engine.
Tests:
1. Extracting behavioral evidence without pre-defined turn markers.
2. ASR + VAD 0.3s pause splitting (turns with role='speaker').
3. Validation against BehavioralEvidenceResponse schema.
"""

import os
import sys
import json
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from schemas_v2 import AnalyzeAutoRequest, BehavioralEvidenceResponse
from service import MultimodalEvaluatorService


def test_auto_turns_pipeline():
    print("=" * 70)
    print("TESTING AUTO-TURN SENSORY ENGINE (VAD 0.3s + ASR)")
    print("=" * 70)

    sample_video = r"C:\Users\loidi\Downloads\tiktok1.mp4"
    if not os.path.exists(sample_video):
        print(f"[SKIP] Sample video not found: {sample_video}")
        return

    # Initialize warm service
    device = os.environ.get("DEVICE", "cuda")
    print(f"[Init] Initializing MultimodalEvaluatorService on '{device}'...")
    evaluator = MultimodalEvaluatorService.get_instance(device=device)

    # 1. Test AnalyzeAutoRequest Schema
    req = AnalyzeAutoRequest(
        video_path=sample_video,
        session_id="test_auto_session",
        vad_threshold=0.3,
        pause_threshold_sec=0.3,
        step=6,
        batch_size=32,
        include_timeline_1s=True
    )
    print(f"[Schema] AnalyzeAutoRequest validated successfully:")
    print(f"  - Video path: {req.video_path}")
    print(f"  - VAD Threshold: {req.vad_threshold}")
    print(f"  - Pause Threshold: {req.pause_threshold_sec}s")

    # 2. Run Auto-Turn Evidence Extraction
    t0 = time.time()
    evidence = evaluator.extract_behavioral_evidence(
        video_path=req.video_path,
        turn_markers=None,
        media_type="video",
        step=req.step,
        batch_size=req.batch_size,
        session_id=req.session_id,
        include_timeline_1s=req.include_timeline_1s,
        vad_threshold=req.vad_threshold,
        pause_threshold_sec=req.pause_threshold_sec
    )
    elapsed = round(time.time() - t0, 2)

    # 3. Validate against Pydantic model
    validated = BehavioralEvidenceResponse(**evidence)
    print(f"\n[PASS] Analysis completed in {elapsed}s (Realtime: {evidence['performance']['realtime_multiplier']})")
    print(f"  - Duration: {validated.duration_seconds}s")
    print(f"  - Eye Contact: {validated.overall_metrics.eye_contact_ratio}")
    print(f"  - Nodding Count: {validated.overall_metrics.nodding_count}")
    print(f"  - Speech Rate WPM: {validated.overall_metrics.speech_rate_wpm}")
    print(f"  - Total Turns Detected: {len(validated.turn_evidence)}")

    assert len(validated.turn_evidence) > 0, "Expected at least 1 turn detected from video!"

    print("\n--- TURN DETAILS ---")
    for t in validated.turn_evidence:
        print(f"  Turn {t.turn_index} [{t.role}]: {t.start_sec}s -> {t.end_sec}s ({t.duration_sec}s)")
        print(f"    Text: '{t.text}'")
        print(f"    Eye Contact: {t.eye_contact_ratio} | Smile: {t.smile_ratio} | Nodding: {t.nodding_count} | WPM: {t.speech_rate_wpm}")
        assert t.role == "speaker", f"Expected role='speaker', got '{t.role}'"

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! AUTO-TURN SENSORY ENGINE OPERATIONAL.")
    print("=" * 70)


if __name__ == "__main__":
    test_auto_turns_pipeline()
