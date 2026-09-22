"""
Face Detection Inference Engine (Batching & Crop Extraction).
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from .model import SCRFDDetector


class BatchFaceDetector:
    """
    High-throughput face detection and alignment for offline batch processing.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thresh: float = 0.5,
        device: str = "cuda",
        crop_size: int = 224,
        detect_interval: int = 3
    ):
        self.detector = SCRFDDetector(
            model_path=model_path,
            conf_thresh=conf_thresh,
            device=device
        )
        self.crop_size = crop_size
        self.detect_interval = max(1, detect_interval)
        self._last_face: Optional[Dict[str, Any]] = None
        self._frames_since_detect: int = 0

    def extract_face_crop(
        self,
        frame_rgb: np.ndarray,
        bbox: List[int],
        margin: float = 0.2
    ) -> np.ndarray:
        """Crop and square-pad face region with a safety margin."""
        h, w = frame_rgb.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1

        # Add margin
        mx = int(bw * margin)
        my = int(bh * margin)
        cx1 = max(0, x1 - mx)
        cy1 = max(0, y1 - my)
        cx2 = min(w, x2 + mx)
        cy2 = min(h, y2 + my)

        crop = frame_rgb[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)

        return cv2.resize(crop, (self.crop_size, self.crop_size))

    def process_frames(
        self,
        frames_rgb: List[np.ndarray]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Processes a batch of frames using high-speed tracking:
        Runs full SCRFD detection every `detect_interval` frames and crops
        intermediate frames using recent face bbox with fresh pixels.
        """
        results = []
        for frame in frames_rgb:
            should_detect = (
                self._last_face is None
                or self._frames_since_detect >= self.detect_interval
            )

            if should_detect:
                detections = self.detector.detect_single(frame)
                if detections:
                    primary = max(
                        detections,
                        key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]) * d["confidence"]
                    )
                    crop = self.extract_face_crop(frame, primary["bbox"])
                    primary["face_crop"] = crop
                    self._last_face = {
                        "bbox": primary["bbox"],
                        "confidence": primary["confidence"],
                        "landmarks": primary["landmarks"]
                    }
                    self._frames_since_detect = 0
                    results.append(primary)
                else:
                    self._last_face = None
                    self._frames_since_detect = 0
                    results.append(None)
            else:
                self._frames_since_detect += 1
                crop = self.extract_face_crop(frame, self._last_face["bbox"])
                results.append({
                    "bbox": self._last_face["bbox"],
                    "confidence": self._last_face["confidence"],
                    "landmarks": self._last_face["landmarks"],
                    "face_crop": crop
                })

        return results
