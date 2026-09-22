"""
6DRepNet Head Pose Estimation Model (ONNX Runtime CUDA / TensorRT / CPU).
Predicts 3D Euler angles (Pitch, Yaw, Roll) from 224x224 RGB face crops.
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional

try:
    from models.download_models import get_model_path
except ImportError:
    from ...models.download_models import get_model_path


class HeadPose6DRepNetModel:
    """
    6DRepNet head pose estimator.
    Takes batch of (224, 224, 3) RGB face crops and returns (pitch, yaw, roll).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda"
    ):
        self.device = device.lower()
        if model_path is None:
            model_path = get_model_path("sixdrepnet.onnx")
        self.model_path = model_path

        providers = []
        if self.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append((
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 1536 * 1024 * 1024,  # 1.5 GB limit
                }
            ))
        providers.append("CPUExecutionProvider")

        self.session = None
        if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 0:
            try:
                self.session = ort.InferenceSession(self.model_path, providers=providers)
                print(f"[6DRepNet] Loaded successfully on {self.session.get_providers()[0]}")
            except Exception as e:
                print(f"[6DRepNet] Warning loading ONNX: {e}")
        else:
            print(f"[6DRepNet] Checkpoint not found at {self.model_path}")

        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1))
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1))

    def preprocess_batch(self, face_crops: List[np.ndarray]) -> np.ndarray:
        """Preprocess batch of RGB images to normalized NCHW tensor."""
        processed = []
        for img in face_crops:
            if img.shape[:2] != (224, 224):
                img = cv2.resize(img, (224, 224))
            # HWC -> CHW, [0, 255] -> [0.0, 1.0]
            chw = img.astype(np.float32) / 255.0
            chw = np.transpose(chw, (2, 0, 1))
            processed.append(chw)
        batch = np.stack(processed, axis=0)
        batch = (batch - self.mean) / self.std
        return batch

    @staticmethod
    def _matrix_to_euler_batch(R: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Convert batch of 3x3 rotation matrices R (shape N, 3, 3) to Euler angles
        (pitch, yaw, roll) in degrees using vectorization.
        """
        if R.ndim == 2:
            R = np.expand_dims(R, axis=0)

        sy = np.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
        singular = sy < 1e-6

        pitch_rad = np.where(
            ~singular,
            np.arctan2(R[:, 2, 1], R[:, 2, 2]),
            np.arctan2(-R[:, 1, 2], R[:, 1, 1])
        )
        yaw_rad = np.arctan2(-R[:, 2, 0], sy)
        roll_rad = np.where(
            ~singular,
            np.arctan2(R[:, 1, 0], R[:, 0, 0]),
            0.0
        )

        pitch_deg = pitch_rad * (180.0 / np.pi)
        yaw_deg = yaw_rad * (180.0 / np.pi)
        roll_deg = roll_rad * (180.0 / np.pi)

        return [
            (float(p), float(y), float(r))
            for p, y, r in zip(pitch_deg, yaw_deg, roll_deg)
        ]

    def predict_batch(
        self,
        face_crops: List[np.ndarray]
    ) -> List[Tuple[float, float, float]]:
        """
        Runs full GPU batch prediction.
        Returns list of (pitch, yaw, roll) in degrees.
        """
        if not face_crops:
            return []

        if self.session is None:
            return [(0.0, 0.0, 0.0) for _ in face_crops]

        input_tensor = self.preprocess_batch(face_crops)
        input_name = self.session.get_inputs()[0].name

        try:
            # 1. Full-batch single ONNX call
            outs = self.session.run(None, {input_name: input_tensor})
            raw_out = outs[0]

            if raw_out.ndim == 3 and raw_out.shape[1] == 3 and raw_out.shape[2] == 3:
                return self._matrix_to_euler_batch(raw_out)
            elif raw_out.ndim == 2 and raw_out.shape[1] == 3:
                return [(float(row[0]), float(row[1]), float(row[2])) for row in raw_out]
            else:
                return [(0.0, 0.0, 0.0) for _ in face_crops]

        except Exception as e:
            # Fallback to single-item iteration if batch fails
            results = []
            for single_chw in input_tensor:
                try:
                    outs = self.session.run(None, {input_name: np.expand_dims(single_chw, 0)})
                    raw_out = outs[0]
                    if raw_out.ndim == 3 and raw_out.shape[1] == 3 and raw_out.shape[2] == 3:
                        angles = self._matrix_to_euler_batch(raw_out)
                        results.append(angles[0])
                    elif raw_out.ndim == 2 and raw_out.shape[1] == 3:
                        results.append((float(raw_out[0][0]), float(raw_out[0][1]), float(raw_out[0][2])))
                    else:
                        results.append((0.0, 0.0, 0.0))
                except Exception:
                    results.append((0.0, 0.0, 0.0))
            return results
