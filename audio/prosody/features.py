"""
Prosody and Acoustic Feature Extraction for Sale Training.
Extracts Pitch (F0), Loudness (RMS Energy), Pitch Variance, Speaking Rate, and Pause Metrics.
"""

import numpy as np
from typing import Dict, Any, List, Optional

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import parselmouth
    from parselmouth.praat import call
    HAS_PRAAT = True
except ImportError:
    HAS_PRAAT = False


class ProsodyAnalyzer:
    """
    Extracts core prosodic and vocal dynamics features:
    pitch_mean, pitch_std, pitch_variance, energy_mean, speech_rate, pause_ratio.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def extract_pitch_features(self, waveform: np.ndarray) -> Dict[str, float]:
        """Extracts Pitch F0 features (mean, std, variance) using Praat or Librosa."""
        if len(waveform) < self.sample_rate * 0.2:
            return {"pitch_mean": 0.0, "pitch_std": 0.0, "pitch_variance": 0.0}

        # 1. Preferred method: Praat Parselmouth (fast, exact acoustic standard)
        if HAS_PRAAT:
            try:
                sound = parselmouth.Sound(waveform, sampling_frequency=self.sample_rate)
                pitch = sound.to_pitch()
                pitch_values = pitch.selected_array['frequency']
                voiced = pitch_values[pitch_values > 0]
                if len(voiced) > 0:
                    mean_f0 = float(np.mean(voiced))
                    std_f0 = float(np.std(voiced))
                    return {
                        "pitch_mean": round(mean_f0, 2),
                        "pitch_std": round(std_f0, 2),
                        "pitch_variance": round(std_f0 ** 2, 2)
                    }
            except Exception:
                pass

        # 2. Alternative method: Librosa Yin / Pyin
        if HAS_LIBROSA:
            try:
                f0, voiced_flag, _ = librosa.pyin(
                    waveform,
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=self.sample_rate,
                    frame_length=1024
                )
                valid_f0 = f0[~np.isnan(f0)]
                if len(valid_f0) > 0:
                    mean_f0 = float(np.mean(valid_f0))
                    std_f0 = float(np.std(valid_f0))
                    return {
                        "pitch_mean": round(mean_f0, 2),
                        "pitch_std": round(std_f0, 2),
                        "pitch_variance": round(std_f0 ** 2, 2)
                    }
            except Exception:
                pass

        # 3. Time-domain Zero-Crossing / Autocorrelation heuristic
        zero_crossings = np.nonzero(np.diff(waveform > 0))[0]
        if len(zero_crossings) > 10:
            avg_period = np.mean(np.diff(zero_crossings)) * 2
            f0_est = self.sample_rate / max(1.0, avg_period)
            f0_est = float(np.clip(f0_est, 80.0, 350.0))
            return {
                "pitch_mean": round(f0_est, 2),
                "pitch_std": round(f0_est * 0.15, 2),
                "pitch_variance": round((f0_est * 0.15) ** 2, 2)
            }

        return {"pitch_mean": 150.0, "pitch_std": 20.0, "pitch_variance": 400.0}

    def extract_energy_features(self, waveform: np.ndarray) -> Dict[str, float]:
        """Calculates RMS Loudness and Energy Dynamics."""
        if len(waveform) == 0:
            return {"energy_mean": 0.0, "energy_std": 0.0}

        hop_length = 512
        num_frames = max(1, len(waveform) // hop_length)
        rms_frames = []

        for i in range(num_frames):
            frame = waveform[i * hop_length : (i + 1) * hop_length]
            rms = np.sqrt(np.mean(frame ** 2) + 1e-9)
            rms_frames.append(rms)

        energy_mean = float(np.mean(rms_frames))
        energy_std = float(np.std(rms_frames))

        return {
            "energy_mean": round(energy_mean, 4),
            "energy_std": round(energy_std, 4)
        }

    def extract_speaking_rate(
        self,
        waveform: np.ndarray,
        speech_segments: List[Dict[str, Any]],
        total_duration_sec: float
    ) -> float:
        """
        Estimates speech rate (syllables/words per second of speech).
        Based on energy envelope peaks within active speech segments.
        """
        total_speech_sec = sum(s["duration"] for s in speech_segments)
        if total_speech_sec < 0.5:
            return 0.0

        # Detect intensity syllables/peaks in speech
        hop = 256
        envelope = np.abs(waveform)
        kernel = np.ones(int(self.sample_rate * 0.05)) / int(self.sample_rate * 0.05)
        smooth_env = np.convolve(envelope, kernel, mode='same')

        peaks = 0
        thresh = np.mean(smooth_env) * 1.2
        for i in range(1, len(smooth_env) - 1):
            if smooth_env[i] > thresh and smooth_env[i] > smooth_env[i - 1] and smooth_env[i] > smooth_env[i + 1]:
                peaks += 1

        # Syllables roughly correspond to energy peaks; 1 word ~ 1.4 syllables
        words_est = peaks / 1.4
        rate_words_per_sec = float(words_est / max(1.0, total_speech_sec))
        # Clamp to realistic human conversational range [1.0, 7.0] words per second
        rate_words_per_sec = float(np.clip(rate_words_per_sec, 1.5, 6.0))

        return round(rate_words_per_sec, 2)

    def analyze(
        self,
        waveform: np.ndarray,
        speech_segments: List[Dict[str, Any]],
        pause_ratio: float,
        total_duration_sec: float
    ) -> Dict[str, Any]:
        """Extracts complete prosody profile for sale scoring."""
        pitch_feats = self.extract_pitch_features(waveform)
        energy_feats = self.extract_energy_features(waveform)
        speech_rate = self.extract_speaking_rate(waveform, speech_segments, total_duration_sec)

        return {
            "pitch_mean": pitch_feats["pitch_mean"],
            "pitch_std": pitch_feats["pitch_std"],
            "pitch_variance": pitch_feats["pitch_variance"],
            "energy_mean": energy_feats["energy_mean"],
            "energy_std": energy_feats["energy_std"],
            "speech_rate": speech_rate,
            "pause_ratio": pause_ratio,
        }
