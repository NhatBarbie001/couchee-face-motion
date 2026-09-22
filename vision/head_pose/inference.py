"""
Head Pose Inference and Behavior Metric Extraction.
"""

from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from .model import HeadPose6DRepNetModel


class BatchHeadPoseAnalyzer:
    """
    Analyzes head pose sequences, categorizes orientations, and detects nodding.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        yaw_thresh: float = 15.0,
        pitch_thresh: float = 15.0
    ):
        self.model = HeadPose6DRepNetModel(model_path=model_path, device=device)
        self.yaw_thresh = yaw_thresh
        self.pitch_thresh = pitch_thresh

    def classify_orientation(self, pitch: float, yaw: float) -> Tuple[str, str]:
        """Classify pitch (up/down/center) and yaw (left/right/center)."""
        if yaw < -self.yaw_thresh:
            yaw_label = "left"
        elif yaw > self.yaw_thresh:
            yaw_label = "right"
        else:
            yaw_label = "center"

        if pitch < -self.pitch_thresh:
            pitch_label = "down"
        elif pitch > self.pitch_thresh:
            pitch_label = "up"
        else:
            pitch_label = "center"

        return yaw_label, pitch_label

    def analyze_batch(
        self,
        face_crops: List[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        Runs batch prediction and returns structured pose dictionaries.
        """
        if not face_crops:
            return []

        euler_angles = self.model.predict_batch(face_crops)
        results = []
        for p, y, r in euler_angles:
            yaw_lbl, pitch_lbl = self.classify_orientation(p, y)
            is_center = (yaw_lbl == "center" and pitch_lbl == "center")
            results.append({
                "pitch": round(p, 2),
                "yaw": round(y, 2),
                "roll": round(r, 2),
                "yaw_label": yaw_lbl,
                "pitch_label": pitch_lbl,
                "is_center": is_center
            })

        return results

    def detect_nodding(self, pitch_series: List[float], min_prominence: float = 8.0) -> int:
        """
        Counts nodding events (pitch oscillation up and down).
        """
        if len(pitch_series) < 6:
            return 0

        nod_count = 0
        diffs = np.diff(pitch_series)
        # Check zero-crossings with adequate amplitude
        sign_changes = np.where(np.diff(np.sign(diffs)))[0]
        
        for idx in sign_changes:
            # Check local excursion magnitude
            start = max(0, idx - 2)
            end = min(len(pitch_series), idx + 3)
            amplitude = np.max(pitch_series[start:end]) - np.min(pitch_series[start:end])
            if amplitude >= min_prominence:
                nod_count += 1

        # Each nod consists of a down-and-up cycle (roughly 2 inflection points)
        return nod_count // 2

    def aggregate_session(
        self,
        pose_results: List[Dict[str, Any]],
        fps: float = 30.0
    ) -> Dict[str, Any]:
        """
        Calculates conversation-level head pose metrics.
        """
        if not pose_results:
            return {
                "head_center_ratio": 1.0,
                "head_left_ratio": 0.0,
                "head_right_ratio": 0.0,
                "head_down_ratio": 0.0,
                "head_movement_rate": 0.0,
                "nodding_count": 0,
            }

        total = len(pose_results)
        center_count = sum(1 for r in pose_results if r["is_center"])
        left_count = sum(1 for r in pose_results if r["yaw_label"] == "left")
        right_count = sum(1 for r in pose_results if r["yaw_label"] == "right")
        down_count = sum(1 for r in pose_results if r["pitch_label"] == "down")

        pitch_series = [r["pitch"] for r in pose_results]
        yaw_series = [r["yaw"] for r in pose_results]

        # Calculate movement rate (std deviation of angles)
        movement_rate = float(np.std(pitch_series) + np.std(yaw_series))
        nodding = self.detect_nodding(pitch_series)

        return {
            "head_center_ratio": round(center_count / total, 4),
            "head_left_ratio": round(left_count / total, 4),
            "head_right_ratio": round(right_count / total, 4),
            "head_down_ratio": round(down_count / total, 4),
            "head_movement_rate": round(movement_rate, 2),
            "nodding_count": nodding,
        }
