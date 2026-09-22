"""
Eye Gaze Inference & Sale Attention Metrics Extractor.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from .model import GazeLLEModel


class BatchEyeGazeAnalyzer:
    """
    Analyzes visual attention and eye contact patterns for sale training.
    """

    def __init__(self, device: str = "cuda", weights_path: Optional[str] = None):
        self.model = GazeLLEModel(device=device, weights_path=weights_path)

    def analyze_batch(
        self,
        face_crops_rgb: List[np.ndarray],
        head_poses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run batch estimation."""
        return self.model.predict_batch(face_crops_rgb, head_poses)

    def aggregate_session(
        self,
        gaze_results: List[Dict[str, Any]],
        fps: float = 30.0,
        step: int = 1
    ) -> Dict[str, Any]:
        """
        Calculates conversation-level gaze metrics:
        - eye_contact_ratio
        - gaze_away_ratio
        - longest_eye_contact (seconds)
        - gaze_break_count
        """
        if not gaze_results:
            return {
                "eye_contact_ratio": 0.0,
                "gaze_away_ratio": 1.0,
                "longest_eye_contact": 0.0,
                "gaze_break_count": 0,
            }

        total_frames = len(gaze_results)
        frame_interval_sec = (1.0 / max(1.0, fps)) * max(1, step)

        eye_contact_count = 0
        gaze_away_count = 0

        longest_contact_frames = 0
        current_contact_frames = 0
        gaze_break_count = 0
        prev_contact = False

        for r in gaze_results:
            is_contact = r["is_eye_contact"]
            if is_contact:
                eye_contact_count += 1
                current_contact_frames += 1
                if current_contact_frames > longest_contact_frames:
                    longest_contact_frames = current_contact_frames
            else:
                gaze_away_count += 1
                if prev_contact:
                    gaze_break_count += 1
                current_contact_frames = 0

            prev_contact = is_contact

        longest_contact_sec = round(longest_contact_frames * frame_interval_sec, 2)

        return {
            "eye_contact_ratio": round(eye_contact_count / total_frames, 4),
            "gaze_away_ratio": round(gaze_away_count / total_frames, 4),
            "longest_eye_contact": longest_contact_sec,
            "gaze_break_count": gaze_break_count,
        }
