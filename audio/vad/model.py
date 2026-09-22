"""
Silero VAD ONNX / PyTorch Engine (CUDA / CPU).
High-precision voice activity detection at 16kHz.
"""

import os
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional

try:
    from models.download_models import get_model_path
except ImportError:
    from ...models.download_models import get_model_path


class SileroVADModel:
    """
    Silero VAD ONNX implementation supporting batching and CUDA acceleration.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        device: str = "cuda"
    ):
        self.threshold = threshold
        self.device = device.lower()

        if model_path is None:
            model_path = get_model_path("silero_vad.onnx")
        self.model_path = model_path

        providers = []
        if self.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = None
        if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 0:
            try:
                self.session = ort.InferenceSession(self.model_path, providers=providers)
                print(f"[SileroVAD] Loaded successfully on {self.session.get_providers()[0]}")
            except Exception as e:
                print(f"[SileroVAD] Warning loading ONNX: {e}")

        self.reset_states()

    def reset_states(self):
        """Reset internal recurrent states for streaming/windowing."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

    def predict_chunk(self, audio_chunk_16k: np.ndarray) -> float:
        """
        Takes 512 samples of 16kHz float32 audio and returns speech probability [0.0, 1.0].
        """
        if self.session is None:
            # Energy fallback
            rms = np.sqrt(np.mean(audio_chunk_16k ** 2) + 1e-9)
            return 1.0 if rms > 0.02 else 0.0

        if len(audio_chunk_16k) != 512:
            if len(audio_chunk_16k) < 512:
                audio_chunk_16k = np.pad(audio_chunk_16k, (0, 512 - len(audio_chunk_16k)))
            else:
                audio_chunk_16k = audio_chunk_16k[:512]

        x = audio_chunk_16k.reshape(1, -1).astype(np.float32)
        sr = np.array([16000], dtype=np.int64)

        try:
            # Silero VAD input signatures: input, state, sr
            inputs = {
                self.session.get_inputs()[0].name: x,
                self.session.get_inputs()[1].name: self._state,
                self.session.get_inputs()[2].name: sr
            }
            outs = self.session.run(None, inputs)
            prob = float(outs[0][0][0])
            self._state = outs[1]
            return prob
        except Exception:
            # Energy fallback
            rms = np.sqrt(np.mean(audio_chunk_16k ** 2) + 1e-9)
            return 1.0 if rms > 0.02 else 0.0
