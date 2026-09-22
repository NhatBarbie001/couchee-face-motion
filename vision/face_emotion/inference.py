"""
Face Emotion Inference & Temporal Aggregation Engine.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from .model import FaceEmotionModel


class BatchFaceEmotionAnalyzer:
    """
    Analyzes emotion sequences across frames and aggregates conversation-level features.
    """

    def __init__(
        self,
        model_name: str = "enet_b0_8_best_vgaf",
        device: str = "cuda",
        engine: str = "onnx"
    ):
        self.model = FaceEmotionModel(
            model_name=model_name,
            device=device,
            engine=engine
        )

    def calculate_valence(self, scores: Dict[str, float]) -> float:
        """
        Computes psychological valence [-1.0, 1.0]:
        Positive: Happiness, mild Surprise
        Negative: Anger, Sadness, Fear, Disgust, Contempt
        """
        pos = scores.get("happiness", 0.0) + 0.3 * scores.get("surprise", 0.0)
        neg = (
            scores.get("anger", 0.0)
            + scores.get("sadness", 0.0)
            + scores.get("fear", 0.0)
            + scores.get("disgust", 0.0)
            + 0.5 * scores.get("contempt", 0.0)
        )
        return float(np.clip(pos - neg, -1.0, 1.0))

    def analyze_batch(
        self,
        face_crops_rgb: List[np.ndarray],
        window_size: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Runs batch emotion analysis and applies temporal smoothing.
        """
        if not face_crops_rgb:
            return []

        raw_scores_list = self.model.predict_batch(face_crops_rgb)
        smoothed_results = []

        # Temporal smoothing buffer
        history: List[Dict[str, float]] = []

        for scores in raw_scores_list:
            history.append(scores)
            if len(history) > window_size:
                history.pop(0)

            # Average over window
            avg_scores = {}
            for k in scores.keys():
                avg_scores[k] = float(np.mean([h[k] for h in history]))

            dominant = max(avg_scores.items(), key=lambda item: item[1])
            valence = self.calculate_valence(avg_scores)

            smoothed_results.append({
                "scores": avg_scores,
                "dominant_emotion": dominant[0],
                "confidence": dominant[1],
                "valence": valence,
            })

        return smoothed_results

    def aggregate_session(
        self,
        frame_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Aggregates frame-level emotion features to conversation-level sale metrics.
        """
        if not frame_results:
            return {
                "happy_ratio": 0.0,
                "neutral_ratio": 1.0,
                "negative_ratio": 0.0,
                "average_valence": 0.0,
            }

        total_frames = len(frame_results)
        happy_count = 0
        neutral_count = 0
        sad_count = 0
        angry_count = 0
        surprise_count = 0
        fear_count = 0
        disgust_count = 0
        neg_count = 0
        valences = []

        for r in frame_results:
            dom = r["dominant_emotion"]
            valences.append(r["valence"])
            if dom == "happiness":
                happy_count += 1
            elif dom == "neutral":
                neutral_count += 1
            elif dom == "sadness":
                sad_count += 1
                neg_count += 1
            elif dom == "anger":
                angry_count += 1
                neg_count += 1
            elif dom == "surprise":
                surprise_count += 1
            elif dom == "fear":
                fear_count += 1
                neg_count += 1
            elif dom in ["disgust", "contempt"]:
                disgust_count += 1
                neg_count += 1

        return {
            "happy_ratio": round(happy_count / total_frames, 4),
            "neutral_ratio": round(neutral_count / total_frames, 4),
            "sad_ratio": round(sad_count / total_frames, 4),
            "angry_ratio": round(angry_count / total_frames, 4),
            "surprise_ratio": round(surprise_count / total_frames, 4),
            "fear_ratio": round(fear_count / total_frames, 4),
            "disgust_ratio": round(disgust_count / total_frames, 4),
            "negative_ratio": round(neg_count / total_frames, 4),
            "average_valence": round(float(np.mean(valences)), 4),
        }
