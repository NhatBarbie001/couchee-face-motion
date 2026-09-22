"""
Feature Fusion Layer: Merges VisionFeatures and AudioFeatures into a Unified Multimodal Representation.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

try:
    from vision.pipeline import VisionFeatures
    from audio.pipeline import AudioFeatures
except (ImportError, ValueError):
    from ..vision.pipeline import VisionFeatures
    from ..audio.pipeline import AudioFeatures


@dataclass
class FusedMultimodalFeatures:
    """Unified multimodal representation for downstream Sale AI scoring."""
    session_id: str
    duration_seconds: float

    # Core Vision Features
    eye_contact_ratio: float
    gaze_away_ratio: float
    head_center_ratio: float
    happy_ratio: float
    neutral_ratio: float
    negative_ratio: float
    average_valence: float
    nodding_count: int
    longest_eye_contact: float
    gaze_break_count: int

    # Core Audio Features
    speech_ratio: float
    pause_ratio: float
    speech_rate: float
    enthusiasm_ratio: float
    pitch_variance: float
    pitch_mean: float
    energy_mean: float
    vocal_happy_ratio: float
    vocal_neutral_ratio: float
    vocal_negative_ratio: float
    hesitation_count: int

    # Multimodal Congruence & Interaction Metrics
    affective_congruence: float  # Alignment between facial valence and vocal enthusiasm
    listening_attentiveness: float  # High eye-contact + nodding during silence

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FeatureFusion:
    """
    Fuses vision and audio feature streams.
    """

    @staticmethod
    def fuse(
        session_id: str,
        vision: VisionFeatures,
        audio: AudioFeatures
    ) -> FusedMultimodalFeatures:
        """
        Fuses VisionFeatures and AudioFeatures, computing multimodal cross-signals.
        """
        # Affective Congruence: Are face and voice expressing similar positivity?
        # e.g., smiling while speaking enthusiastically vs smiling while vocal is deadpan
        face_pos = vision.happy_ratio + max(0.0, vision.average_valence)
        voice_pos = audio.enthusiasm_ratio + audio.vocal_happy_ratio
        congruence = float(np_clip_1(1.0 - abs(face_pos - voice_pos)))

        # Listening Attentiveness: Maintaining eye contact and nodding when listening (pause_ratio)
        listening_score = vision.eye_contact_ratio * 0.7 + min(1.0, vision.nodding_count / 5.0) * 0.3
        listening_attentiveness = float(round(listening_score, 4))

        duration = max(vision.duration_seconds, audio.duration_seconds)

        return FusedMultimodalFeatures(
            session_id=session_id,
            duration_seconds=round(duration, 2),
            eye_contact_ratio=vision.eye_contact_ratio,
            gaze_away_ratio=vision.gaze_away_ratio,
            head_center_ratio=vision.head_center_ratio,
            happy_ratio=vision.happy_ratio,
            neutral_ratio=vision.neutral_ratio,
            negative_ratio=vision.negative_ratio,
            average_valence=vision.average_valence,
            nodding_count=vision.nodding_count,
            longest_eye_contact=vision.longest_eye_contact,
            gaze_break_count=vision.gaze_break_count,
            speech_ratio=audio.speech_ratio,
            pause_ratio=audio.pause_ratio,
            speech_rate=audio.speech_rate,
            enthusiasm_ratio=audio.enthusiasm_ratio,
            pitch_variance=audio.pitch_variance,
            pitch_mean=audio.pitch_mean,
            energy_mean=audio.energy_mean,
            vocal_happy_ratio=audio.vocal_happy_ratio,
            vocal_neutral_ratio=audio.vocal_neutral_ratio,
            vocal_negative_ratio=audio.vocal_negative_ratio,
            hesitation_count=audio.hesitation_count,
            affective_congruence=round(congruence, 4),
            listening_attentiveness=listening_attentiveness,
        )


def np_clip_1(val: float) -> float:
    return max(0.0, min(1.0, val))
