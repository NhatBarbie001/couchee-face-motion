"""
AudioModule: Standalone Audio Analytics Pipeline for High-Throughput Speech Processing.
Coordinates Silero VAD + Aniemore Wav2Vec2 + Prosody Extractor.
"""

import os
import subprocess
import soundfile as sf
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

from .vad import BatchVADAnalyzer
from .voice_emotion import BatchVoiceEmotionAnalyzer
from .prosody import ProsodyAnalyzer
from .asr_engine import ZipformerASR


@dataclass
class AudioFeatures:
    """Standardized conversation-level audio features for Sale AI."""
    speech_ratio: float
    pause_ratio: float
    speech_rate: float
    enthusiasm_ratio: float
    pitch_variance: float
    pitch_mean: float
    pitch_std: float
    energy_mean: float
    vocal_happy_ratio: float
    vocal_neutral_ratio: float
    vocal_negative_ratio: float
    hesitation_count: int
    total_speech_seconds: float
    duration_seconds: float
    speech_segments: Optional[List[Dict[str, Any]]] = None
    voice_emotion_segments: Optional[List[Dict[str, Any]]] = None
    transcript: str = ""
    word_timestamps: Optional[List[Dict[str, Any]]] = None
    speech_rate_wpm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AudioModule:
    """
    Independent Audio Pipeline supporting direct audio or video files.
    """

    def __init__(
        self,
        device: str = "cuda",
        sample_rate: int = 16000
    ):
        self.device = device
        self.sample_rate = sample_rate
        print(f"[AudioModule] Initializing Audio Pipeline on device '{self.device}'...")

        self.vad_analyzer = BatchVADAnalyzer(device=self.device)
        self.voice_emotion_analyzer = BatchVoiceEmotionAnalyzer(device=self.device)
        self.prosody_analyzer = ProsodyAnalyzer(sample_rate=self.sample_rate)
        self.asr_engine = ZipformerASR(sample_rate=self.sample_rate)

    def extract_audio_if_video(self, media_path: str, temp_wav_dir: str = "temp_audio") -> str:
        """Extracts 16kHz mono WAV from media file via FFmpeg or returns path if already WAV."""
        ext = os.path.splitext(media_path)[1].lower()
        if ext == ".wav":
            return media_path

        os.makedirs(temp_wav_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(media_path))[0]
        out_wav = os.path.join(temp_wav_dir, f"{base}_16k_mono.wav")

        # Use FFmpeg to convert to 16kHz mono PCM 16-bit
        cmd = [
            "ffmpeg", "-y", "-i", media_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            out_wav
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return out_wav
        except Exception:
            # If ffmpeg is not available, try reading directly
            return media_path

    def load_waveform_16k(self, audio_path: str) -> Tuple[np.ndarray, float]:
        """Loads and normalizes 16kHz float32 mono audio."""
        try:
            data, sr = sf.read(audio_path, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)  # Convert stereo to mono

            if sr != self.sample_rate:
                # Basic linear resample
                num_target_samples = int(len(data) * self.sample_rate / sr)
                data = np.interp(
                    np.linspace(0.0, 1.0, num_target_samples, endpoint=False),
                    np.linspace(0.0, 1.0, len(data), endpoint=False),
                    data
                ).astype(np.float32)

            duration = len(data) / self.sample_rate
            return data, duration
        except Exception as e:
            print(f"[AudioModule] Error reading audio file {audio_path}: {e}")
            dummy_data = np.zeros(self.sample_rate * 5, dtype=np.float32)
            return dummy_data, 5.0

    def process(
        self,
        media_path: str,
        vad_threshold: Optional[float] = None,
        pause_threshold_sec: float = 0.3
    ) -> AudioFeatures:
        """
        Processes an audio or video file and returns standardized AudioFeatures.
        
        Args:
            media_path: Path to audio (.wav, .mp3) or video (.mp4) file
            vad_threshold: Optional sensitivity threshold for VAD (e.g. 0.3)
            pause_threshold_sec: Silence duration (seconds) to split utterances/turns (default: 0.3)
        """
        if not os.path.exists(media_path):
            raise FileNotFoundError(f"Media file not found: {media_path}")

        wav_path = self.extract_audio_if_video(media_path)
        waveform, duration_sec = self.load_waveform_16k(wav_path)

        # 1. Silero VAD (Acoustic Voice Activity Detection)
        min_silence_ms = max(100, int(pause_threshold_sec * 1000))
        vad_results = self.vad_analyzer.analyze_waveform(
            waveform,
            threshold=vad_threshold,
            min_silence_duration_ms=min_silence_ms
        )
        speech_segments = vad_results.get("speech_segments") or []

        # 2. Zipformer ASR Speech-to-Text Transcription (Linguistic Ground Truth)
        asr_result = self.asr_engine.transcribe(
            waveform,
            sample_rate=self.sample_rate
        )
        word_timestamps = [w.to_dict() for w in asr_result.words]

        # 3. Multimodal Speech Segmentation Reconciliation (VAD + ASR Ground Truth)
        # Groups ASR words based on pause_threshold_sec (>= 0.3s silence cuts span)
        if word_timestamps:
            asr_spans = []
            c_start = word_timestamps[0]["start"]
            c_end = word_timestamps[0]["end"]
            for w in word_timestamps[1:]:
                if w["start"] - c_end < pause_threshold_sec:
                    c_end = max(c_end, w["end"])
                else:
                    asr_spans.append({
                        "start": round(c_start, 2),
                        "end": round(c_end, 2),
                        "duration": round(c_end - c_start, 2),
                        "speech": True
                    })
                    c_start = w["start"]
                    c_end = w["end"]
            asr_spans.append({
                "start": round(c_start, 2),
                "end": round(c_end, 2),
                "duration": round(c_end - c_start, 2),
                "speech": True
            })

            if not speech_segments:
                speech_segments = asr_spans
            else:
                combined = sorted(speech_segments + asr_spans, key=lambda s: s["start"])
                merged = []
                cur = combined[0]
                for nxt in combined[1:]:
                    if nxt["start"] <= cur["end"] + (pause_threshold_sec * 0.8):
                        cur["end"] = max(cur["end"], nxt["end"])
                        cur["duration"] = round(cur["end"] - cur["start"], 2)
                    else:
                        merged.append(cur)
                        cur = nxt
                merged.append(cur)
                speech_segments = merged

        total_speech_sec = round(sum(s["duration"] for s in speech_segments), 2)
        speech_ratio = round(min(1.0, total_speech_sec / max(0.1, duration_sec)), 4)
        pause_ratio = round(max(0.0, 1.0 - speech_ratio), 4)

        # Hesitation count (pauses >= 2.0s between speech intervals)
        hesitation_count = 0
        for i in range(len(speech_segments) - 1):
            pause_len = speech_segments[i + 1]["start"] - speech_segments[i]["end"]
            if pause_len >= 2.0:
                hesitation_count += 1

        # 4. Aniemore Wav2Vec2 Voice Emotion (Inference on reconciled speech segments)
        voice_emotions = self.voice_emotion_analyzer.analyze_segments(waveform, speech_segments)
        emotion_stats = self.voice_emotion_analyzer.aggregate_session(voice_emotions)

        # 5. Prosody Extraction (Pitch, Energy, Speaking Rate, Variance)
        prosody_stats = self.prosody_analyzer.analyze(
            waveform=waveform,
            speech_segments=speech_segments,
            pause_ratio=pause_ratio,
            total_duration_sec=duration_sec
        )

        return AudioFeatures(
            speech_ratio=speech_ratio,
            pause_ratio=pause_ratio,
            speech_rate=prosody_stats["speech_rate"],
            enthusiasm_ratio=emotion_stats["vocal_enthusiasm_ratio"],
            pitch_variance=prosody_stats["pitch_variance"],
            pitch_mean=prosody_stats["pitch_mean"],
            pitch_std=prosody_stats["pitch_std"],
            energy_mean=prosody_stats["energy_mean"],
            vocal_happy_ratio=emotion_stats["vocal_happy_ratio"],
            vocal_neutral_ratio=emotion_stats["vocal_neutral_ratio"],
            vocal_negative_ratio=emotion_stats["vocal_negative_ratio"],
            hesitation_count=hesitation_count,
            total_speech_seconds=total_speech_sec,
            duration_seconds=round(duration_sec, 2),
            speech_segments=speech_segments,
            voice_emotion_segments=voice_emotions,
            transcript=asr_result.full_transcript,
            word_timestamps=word_timestamps,
            speech_rate_wpm=asr_result.speech_rate_wpm
        )
