"""
Temporal Window Synchronization and 1-Second Time-Binned Timeline Builder.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class TemporalAligner:
    """
    Builds second-by-second timeline (1s bins) aligning vision and audio streams.
    """

    def __init__(self, bin_size_sec: float = 1.0):
        self.bin_size_sec = bin_size_sec

    def build_timeline_1s(
        self,
        duration_sec: float,
        frame_timeline: List[Dict[str, Any]],
        speech_segments: Optional[List[Dict[str, Any]]] = None,
        prosody_info: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Creates a uniform 1-second interval timeline array.
        """
        num_seconds = max(1, int(np.ceil(duration_sec)))
        timeline_1s = []

        speech_segments = speech_segments or []
        prosody_info = prosody_info or {}
        base_pitch = prosody_info.get("pitch_mean", 175.0)

        for s in range(num_seconds):
            t_start = s * self.bin_size_sec
            t_end = min(duration_sec, (s + 1) * self.bin_size_sec)

            # Vision frames in this 1s slice
            sec_frames = [
                f for f in frame_timeline if t_start <= f["timestamp"] <= t_end
            ]

            if sec_frames:
                is_focused = not any(f.get("is_distracted", False) for f in sec_frames)
                emotions = [f.get("dominant_emotion", "neutral") for f in sec_frames]
                dom_emo = max(set(emotions), key=emotions.count) if emotions else "neutral"
                avg_val = round(float(np.mean([f.get("valence", 0.0) for f in sec_frames])), 2)
                avg_yaw = round(float(np.mean([f.get("yaw", 0.0) for f in sec_frames])), 1)
                avg_pitch = round(float(np.mean([f.get("pitch", 0.0) for f in sec_frames])), 1)
                gaze_lbls = [f.get("gaze_label", "away") for f in sec_frames]
                dom_gaze = max(set(gaze_lbls), key=gaze_lbls.count) if gaze_lbls else "away"
            else:
                is_focused = True
                dom_emo = "neutral"
                avg_val = 0.0
                avg_yaw = 0.0
                avg_pitch = 0.0
                dom_gaze = "camera"

            # Check if this second overlaps with any speech interval
            is_speech = any(
                max(t_start, seg["start"]) < min(t_end, seg["end"])
                for seg in speech_segments
            )

            timeline_1s.append({
                "second": s,
                "time_range": [round(t_start, 2), round(t_end, 2)],
                "vision": {
                    "is_focused": is_focused,
                    "dominant_emotion": dom_emo,
                    "valence": avg_val,
                    "head_yaw": avg_yaw,
                    "head_pitch": avg_pitch,
                    "gaze": dom_gaze
                },
                "audio": {
                    "is_speech": is_speech,
                    "pitch": round(base_pitch if is_speech else 0.0, 1),
                    "energy": 0.12 if is_speech else 0.01,
                    "tone": "enthusiasm" if (is_speech and avg_val > 0.2) else ("neutral" if is_speech else "silence")
                }
            })

        return timeline_1s
