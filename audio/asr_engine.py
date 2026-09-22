"""
Zipformer ASR Engine using sherpa-onnx for high-speed, accurate Vietnamese speech recognition
with word timestamps and speech rate (WPM) calculation.
"""

import os
import tarfile
import urllib.request
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union
import numpy as np

try:
    import sherpa_onnx
    HAS_SHERPA = True
except ImportError:
    HAS_SHERPA = False


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ASRResult:
    full_transcript: str
    total_words: int
    speech_rate_wpm: float
    words: List[WordTimestamp]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "full_transcript": self.full_transcript,
            "total_words": self.total_words,
            "speech_rate_wpm": self.speech_rate_wpm,
            "words": [w.to_dict() for w in self.words]
        }


class ZipformerASR:
    """
    Offline Automatic Speech Recognition engine powered by Zipformer transducer in INT8 ONNX format.
    Optimized for CPU & GPU server execution.
    """

    DEFAULT_MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2"
    DEFAULT_MODEL_DIR_NAME = "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09"

    def __init__(
        self,
        model_dir: Optional[str] = None,
        num_threads: int = 4,
        sample_rate: int = 16000,
        decoding_method: str = "greedy_search"
    ):
        self.sample_rate = sample_rate
        self.num_threads = num_threads
        self.decoding_method = decoding_method
        self.recognizer = None

        if not HAS_SHERPA:
            print("[ZipformerASR] Warning: 'sherpa-onnx' not installed. ASR will return empty transcript.")
            return

        self.model_dir = self._resolve_model_dir(model_dir)

        try:
            self._ensure_model_exists()
            encoder_path = os.path.join(self.model_dir, "encoder.int8.onnx")
            decoder_path = os.path.join(self.model_dir, "decoder.onnx")
            joiner_path = os.path.join(self.model_dir, "joiner.int8.onnx")
            tokens_path = os.path.join(self.model_dir, "tokens.txt")

            for p in [encoder_path, decoder_path, joiner_path, tokens_path]:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Zipformer required model file not found: {p}")

            print(f"[ZipformerASR] Initializing sherpa-onnx transducer recognizer (threads={num_threads})...")
            self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=encoder_path,
                decoder=decoder_path,
                joiner=joiner_path,
                tokens=tokens_path,
                num_threads=self.num_threads,
                sample_rate=self.sample_rate,
                feature_dim=80,
                decoding_method=self.decoding_method,
                debug=False
            )
            print(f"[ZipformerASR] Loaded Vietnamese Zipformer model successfully.")
        except Exception as e:
            print(f"[ZipformerASR] Warning: Failed to load Zipformer model ({e}). Continuing with graceful fallback.")
            self.recognizer = None

    def _resolve_model_dir(self, explicit_dir: Optional[str]) -> str:
        """Finds or resolves the directory for the Zipformer model."""
        if explicit_dir and os.path.exists(explicit_dir):
            return os.path.abspath(explicit_dir)

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        local_dir = os.path.join(base_dir, "models", "checkpoints", self.DEFAULT_MODEL_DIR_NAME)
        if os.path.exists(local_dir):
            return local_dir

        legacy_cache_dir = os.path.abspath(
            os.path.join(base_dir, "..", "Face-motion", "models_cache", self.DEFAULT_MODEL_DIR_NAME)
        )
        if os.path.exists(legacy_cache_dir):
            return legacy_cache_dir

        return local_dir

    def _ensure_model_exists(self):
        """Downloads and extracts the model archive if not present."""
        if os.path.exists(self.model_dir):
            return

        parent_dir = os.path.dirname(self.model_dir)
        os.makedirs(parent_dir, exist_ok=True)
        tar_path = os.path.join(parent_dir, f"{self.DEFAULT_MODEL_DIR_NAME}.tar.bz2")

        # Check if archive exists in legacy cache
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        legacy_tar = os.path.abspath(
            os.path.join(base_dir, "..", "Face-motion", "models_cache", "zipformer_vi.tar.bz2")
        )
        if os.path.exists(legacy_tar):
            print(f"[ZipformerASR] Extracting existing model archive from {legacy_tar}...")
            with tarfile.open(legacy_tar, "r:bz2") as tar:
                tar.extractall(parent_dir)
            return

        print(f"[ZipformerASR] Downloading Vietnamese Zipformer model (26MB)...")
        urllib.request.urlretrieve(self.DEFAULT_MODEL_URL, tar_path)
        print(f"[ZipformerASR] Extracting model archive...")
        with tarfile.open(tar_path, "r:bz2") as tar:
            tar.extractall(parent_dir)
        if os.path.exists(tar_path):
            try:
                os.remove(tar_path)
            except OSError:
                pass

    def transcribe(
        self,
        media_or_waveform: Union[str, np.ndarray],
        sample_rate: int = 16000,
        speech_duration_sec: Optional[float] = None
    ) -> ASRResult:
        """
        Transcribes the complete audio waveform and returns transcript with word timestamps.
        """
        if self.recognizer is None:
            return ASRResult(full_transcript="", total_words=0, speech_rate_wpm=0.0, words=[])

        if isinstance(media_or_waveform, str):
            try:
                import soundfile as sf
                data, sr = sf.read(media_or_waveform)
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                waveform = data.astype(np.float32)
                sample_rate = sr
            except Exception:
                return ASRResult(full_transcript="", total_words=0, speech_rate_wpm=0.0, words=[])
        else:
            waveform = media_or_waveform.astype(np.float32)

        if sample_rate != self.sample_rate:
            num_target_samples = int(len(waveform) * self.sample_rate / sample_rate)
            waveform = np.interp(
                np.linspace(0.0, 1.0, num_target_samples, endpoint=False),
                np.linspace(0.0, 1.0, len(waveform), endpoint=False),
                waveform
            ).astype(np.float32)
            sample_rate = self.sample_rate

        total_audio_sec = len(waveform) / float(sample_rate)
        if total_audio_sec < 0.1:
            return ASRResult(full_transcript="", total_words=0, speech_rate_wpm=0.0, words=[])

        try:
            stream = self.recognizer.create_stream()
            stream.accept_waveform(sample_rate, waveform)
            self.recognizer.decode_stream(stream)
            res = stream.result

            raw_text = (res.text or "").strip()
            tokens = list(res.tokens) if hasattr(res, "tokens") and res.tokens else []
            timestamps = list(res.timestamps) if hasattr(res, "timestamps") and res.timestamps else []

            words = self._extract_words_with_timestamps(tokens, timestamps, raw_text)
            word_count = len(words) if words else len(raw_text.split())

            active_time_sec = speech_duration_sec if (speech_duration_sec and speech_duration_sec > 0) else total_audio_sec
            active_minutes = active_time_sec / 60.0
            wpm = round((word_count / active_minutes) if active_minutes > 0 else 0.0, 1)

            return ASRResult(
                full_transcript=raw_text,
                total_words=word_count,
                speech_rate_wpm=wpm,
                words=words
            )
        except Exception as e:
            print(f"[ZipformerASR] Error during transcription: {e}")
            return ASRResult(full_transcript="", total_words=0, speech_rate_wpm=0.0, words=[])

    def _extract_words_with_timestamps(
        self,
        tokens: List[str],
        timestamps: List[float],
        full_text: str
    ) -> List[WordTimestamp]:
        """
        Reconstructs words and attaches start and end timestamps from BPE tokens.
        """
        words: List[WordTimestamp] = []
        if not tokens or not timestamps or len(tokens) != len(timestamps):
            split_words = full_text.split()
            if not split_words:
                return []
            start_t = timestamps[0] if timestamps else 0.0
            end_t = timestamps[-1] if timestamps else 1.0
            step = (end_t - start_t) / max(1, len(split_words))
            for i, w in enumerate(split_words):
                w_start = round(start_t + i * step, 2)
                w_end = round(w_start + step, 2)
                words.append(WordTimestamp(word=w, start=w_start, end=w_end))
            return words

        current_word_chars: List[str] = []
        word_start: Optional[float] = None
        word_end: float = 0.0

        for token, ts in zip(tokens, timestamps):
            is_new_word = token.startswith(" ") or token.startswith(" ")
            clean_token = token.replace(" ", "").replace(" ", "")

            if is_new_word and current_word_chars:
                w_str = "".join(current_word_chars).strip()
                if w_str:
                    words.append(WordTimestamp(
                        word=w_str,
                        start=round(word_start if word_start is not None else 0.0, 2),
                        end=round(word_end, 2)
                    ))
                current_word_chars = []
                word_start = ts

            if word_start is None:
                word_start = ts
            word_end = ts + 0.35

            if clean_token:
                current_word_chars.append(clean_token)

        if current_word_chars:
            w_str = "".join(current_word_chars).strip()
            if w_str:
                words.append(WordTimestamp(
                    word=w_str,
                    start=round(word_start if word_start is not None else 0.0, 2),
                    end=round(word_end, 2)
                ))

        return words
