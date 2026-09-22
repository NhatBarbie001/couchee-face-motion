"""
Silero VAD Inference & Speech Segment Extractor.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from .model import SileroVADModel


class BatchVADAnalyzer:
    """
    Analyzes full audio waveforms to segment speech vs silence intervals.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300,
        device: str = "cuda"
    ):
        self.model = SileroVADModel(model_path=model_path, threshold=threshold, device=device)
        self.min_speech_samples = int(16000 * min_speech_duration_ms / 1000)
        self.min_silence_samples = int(16000 * min_silence_duration_ms / 1000)

    def analyze_waveform(
        self,
        waveform_16k: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyzes a 16kHz mono audio array.
        Returns speech intervals, silence intervals, hesitations > 2.0s, and speech ratio.
        """
        self.model.reset_states()
        total_samples = len(waveform_16k)
        duration_sec = total_samples / 16000.0

        if total_samples < 512:
            return {
                "speech_segments": [],
                "speech_ratio": 0.0,
                "pause_ratio": 1.0,
                "hesitation_count": 0,
                "total_speech_sec": 0.0,
            }

        chunk_size = 512
        num_chunks = total_samples // chunk_size
        speech_probs = []

        for i in range(num_chunks):
            chunk = waveform_16k[i * chunk_size : (i + 1) * chunk_size]
            prob = self.model.predict_chunk(chunk)
            speech_probs.append(prob)

        # Hysteresis thresholding to extract segments
        speech_segments = []
        is_speaking = False
        start_chunk = 0

        for i, p in enumerate(speech_probs):
            if p >= self.model.threshold and not is_speaking:
                is_speaking = True
                start_chunk = i
            elif p < (self.model.threshold * 0.7) and is_speaking:
                is_speaking = False
                start_sec = round((start_chunk * chunk_size) / 16000.0, 2)
                end_sec = round((i * chunk_size) / 16000.0, 2)
                if (end_sec - start_sec) >= (self.min_speech_samples / 16000.0):
                    speech_segments.append({
                        "start": start_sec,
                        "end": end_sec,
                        "duration": round(end_sec - start_sec, 2),
                        "speech": True
                    })

        # Close segment if still speaking at end
        if is_speaking:
            start_sec = round((start_chunk * chunk_size) / 16000.0, 2)
            end_sec = round(duration_sec, 2)
            speech_segments.append({
                "start": start_sec,
                "end": end_sec,
                "duration": round(end_sec - start_sec, 2),
                "speech": True
            })

        # Calculate speech duration & pauses
        total_speech_sec = sum(s["duration"] for s in speech_segments)
        speech_ratio = min(1.0, total_speech_sec / max(0.1, duration_sec))
        pause_ratio = max(0.0, 1.0 - speech_ratio)

        # Count long hesitations (> 2.0s)
        hesitation_count = 0
        for i in range(len(speech_segments) - 1):
            pause_len = speech_segments[i + 1]["start"] - speech_segments[i]["end"]
            if pause_len >= 2.0:
                hesitation_count += 1

        return {
            "speech_segments": speech_segments,
            "speech_ratio": round(speech_ratio, 4),
            "pause_ratio": round(pause_ratio, 4),
            "hesitation_count": hesitation_count,
            "total_speech_sec": round(total_speech_sec, 2),
            "duration_sec": round(duration_sec, 2)
        }
