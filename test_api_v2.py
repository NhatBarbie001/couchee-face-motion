"""
Unit and Integration Test for Face-motion-gpu Sensory Engine v2.
Validates:
1. Video analysis with ground-truth turn_markers.
2. Audio-only analysis (skipping GPU vision).
3. Pydantic schema validation for v2 request/response.
"""

import os
import sys
import json
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ.setdefault("DEVICE", "cpu")

from schemas_v2 import AnalyzeRequestV2, BehavioralEvidenceResponse, TurnMarker
from service import MultimodalEvaluatorService


def test_v2_pipeline():
    print("=" * 70)
    print("TESTING FACE-MOTION-GPU SENSORY ENGINE v2")
    print("=" * 70)

    sample_video = r"C:\Users\loidi\Downloads\tiktok1.mp4"
    if not os.path.exists(sample_video):
        print(f"[SKIP] Sample video not found: {sample_video}")
        return

    # Initialize warm service
    device = os.environ.get("DEVICE", "cuda")
    evaluator = MultimodalEvaluatorService.get_instance(device=device)

    # -------------------------------------------------------------
    # TEST 1: Full Video + Ground-Truth Turn Markers
    # -------------------------------------------------------------
    print("\n>>> TEST 1: Full Video with Turn Markers")
    turn_markers = [
        TurnMarker(role="student", start_sec=0.0, end_sec=15.0, text="Chào mừng quý khách..."),
        TurnMarker(role="ai", start_sec=15.0, end_sec=25.0, text="Dạ cảm ơn bạn..."),
        TurnMarker(role="student", start_sec=25.0, end_sec=35.0, text="Để em giới thiệu thêm...")
    ]

    req_v2 = AnalyzeRequestV2(
        session_id="test_session_v2",
        video_path=sample_video,
        media_type="video",
        turn_markers=turn_markers,
        step=6,
        batch_size=32,
        include_timeline_1s=True
    )

    t0 = time.time()
    evidence = evaluator.extract_behavioral_evidence(
        video_path=req_v2.video_path,
        turn_markers=req_v2.turn_markers,
        media_type=req_v2.media_type,
        step=req_v2.step,
        batch_size=req_v2.batch_size,
        session_id=req_v2.session_id,
        include_timeline_1s=req_v2.include_timeline_1s
    )
    elapsed = round(time.time() - t0, 2)

    # Validate against Pydantic model
    validated = BehavioralEvidenceResponse(**evidence)
    print(f"[PASS] Video analyzed in {elapsed}s (Realtime: {evidence['performance']['realtime_multiplier']})")
    print(f"  - Duration: {validated.duration_seconds}s")
    print(f"  - Eye Contact Ratio: {validated.overall_metrics.eye_contact_ratio}")
    print(f"  - Nodding Count: {validated.overall_metrics.nodding_count}")
    print(f"  - Speech Rate WPM: {validated.overall_metrics.speech_rate_wpm}")
    print(f"  - Distraction Moments: {len(validated.anomalies.distraction_moments)}")
    print(f"  - Hesitation Moments: {len(validated.anomalies.hesitation_moments)}")
    print(f"  - Turn Evidence Count: {len(validated.turn_evidence)}")

    assert len(validated.turn_evidence) == 3, "Expected 3 turns aligned with input markers!"
    assert validated.turn_evidence[0].role == "student"
    assert validated.turn_evidence[1].role == "ai"
    print("  [OK] Ground-truth turn alignment verified!")

    # -------------------------------------------------------------
    # TEST 2: Audio-Only Mode (Camera Disabled)
    # -------------------------------------------------------------
    print("\n>>> TEST 2: Audio-Only Mode")
    req_audio = AnalyzeRequestV2(
        session_id="test_audio_session",
        video_path=sample_video,  # Using the same file to read audio track
        media_type="audio",
        turn_markers=[TurnMarker(role="student", start_sec=0.0, end_sec=15.0)],
        include_timeline_1s=False
    )

    t0 = time.time()
    audio_evidence = evaluator.extract_behavioral_evidence(
        video_path=req_audio.video_path,
        turn_markers=req_audio.turn_markers,
        media_type=req_audio.media_type,
        session_id=req_audio.session_id,
        include_timeline_1s=req_audio.include_timeline_1s
    )
    elapsed = round(time.time() - t0, 2)

    validated_audio = BehavioralEvidenceResponse(**audio_evidence)
    print(f"[PASS] Audio-only analyzed in {elapsed}s")
    print(f"  - Media type: {validated_audio.media_type}")
    print(f"  - Speech Rate WPM: {validated_audio.overall_metrics.speech_rate_wpm}")
    print(f"  - Pitch Variance: {validated_audio.overall_metrics.pitch_variance}")
    print(f"  - Eye contact (should be None): {validated_audio.overall_metrics.eye_contact_ratio}")

    assert validated_audio.overall_metrics.eye_contact_ratio is None, "Vision metrics should be None in audio mode!"
    print("  [OK] Audio-only bypass verified successfully!")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! SENSORY ENGINE v2 READY FOR INTEGRATION.")
    print("=" * 70)


if __name__ == "__main__":
    test_v2_pipeline()
