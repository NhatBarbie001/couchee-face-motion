"""
Voice Emotion Inference and Conversation Aggregation.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from .model import VoiceEmotionModel


class BatchVoiceEmotionAnalyzer:
    """
    Analyzes emotional states of speech segments.
    """

    def __init__(
        self,
        model_name: str = "Aniemore/wav2vec2-xlsr-53-russian-emotion-distil",
        device: str = "cuda"
    ):
        self.model = VoiceEmotionModel(model_name=model_name, device=device)

    def analyze_segments(
        self,
        waveform_16k: np.ndarray,
        speech_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Runs emotion analysis only on active speech segments.
        """
        results = []
        for seg in speech_segments:
            start_sample = int(seg["start"] * 16000)
            end_sample = int(seg["end"] * 16000)
            chunk = waveform_16k[start_sample:end_sample]

            if len(chunk) < 800:
                continue

            scores = self.model.predict_segment(chunk)
            dominant = max(scores.items(), key=lambda x: x[1])

            results.append({
                "start": seg["start"],
                "end": seg["end"],
                "duration": seg["duration"],
                "scores": scores,
                "dominant_emotion": dominant[0],
                "confidence": dominant[1]
            })

        return results

    def aggregate_session(
        self,
        segment_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Aggregates duration-weighted vocal emotion metrics for sale evaluation.
        """
        if not segment_results:
            return {
                "vocal_neutral_ratio": 1.0,
                "vocal_happy_ratio": 0.0,
                "vocal_enthusiasm_ratio": 0.0,
                "vocal_negative_ratio": 0.0,
            }

        total_speech_sec = sum(r["duration"] for r in segment_results)
        if total_speech_sec <= 0:
            total_speech_sec = 1.0

        neutral_sec = 0.0
        happy_sec = 0.0
        enthusiasm_sec = 0.0
        neg_sec = 0.0

        for r in segment_results:
            dur = r["duration"]
            scores = r["scores"]
            neutral_sec += dur * scores.get("neutral", 0.0)
            happy_sec += dur * scores.get("happiness", 0.0)
            enthusiasm_sec += dur * scores.get("enthusiasm", 0.0)
            neg_sec += dur * (scores.get("sadness", 0.0) + scores.get("anger", 0.0))

        return {
            "vocal_neutral_ratio": round(neutral_sec / total_speech_sec, 4),
            "vocal_happy_ratio": round(happy_sec / total_speech_sec, 4),
            "vocal_enthusiasm_ratio": round(enthusiasm_sec / total_speech_sec, 4),
            "vocal_negative_ratio": round(neg_sec / total_speech_sec, 4),
        }
