"""
Multimodal Frame Annotator for High-Visual Video Rendering.
Draws corner-accented bounding box, 3D Euler pose axes, Eye Contact badge, and semi-transparent HUD Dashboard.
"""

from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np


class MultimodalAnnotator:
    """
    Renders rich multimodal visual overlays directly onto video frames.
    """

    # Vibrant harmonious color palette (BGR)
    EMOTION_COLORS = {
        "anger": (50, 50, 220),       # Red
        "contempt": (140, 60, 180),   # Magenta
        "disgust": (60, 140, 60),     # Forest Green
        "fear": (180, 100, 40),       # Amber / Deep Orange
        "happiness": (30, 215, 255),  # Gold / Radiant Yellow
        "neutral": (180, 180, 180),   # Cool Slate Gray
        "sadness": (220, 120, 50),    # Soft Blue
        "surprise": (220, 220, 30),   # Cyan / Mint
    }
    DEFAULT_COLOR = (200, 200, 200)

    def __init__(self, show_hud: bool = True):
        self.show_hud = show_hud

    def draw_corner_box(
        self,
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
        length: int = 20
    ):
        """Draws bounding box with modern corner accents."""
        x1, y1 = pt1
        x2, y2 = pt2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

        l = min(length, max(5, (x2 - x1) // 4), max(5, (y2 - y1) // 4))
        # Top-Left
        cv2.line(img, (x1, y1), (x1 + l, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + l), color, thickness)
        # Top-Right
        cv2.line(img, (x2, y1), (x2 - l, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + l), color, thickness)
        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + l, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - l), color, thickness)
        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - l, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - l), color, thickness)

    def draw_headpose_axis(
        self,
        img: np.ndarray,
        pitch: float,
        yaw: float,
        roll: float,
        tdx: int,
        tdy: int,
        size: int = 50
    ):
        """
        Draws 3D pose vectors (Pitch: Red, Yaw: Green, Roll: Blue) projecting from face center.
        """
        p = pitch * np.pi / 180
        y = -(yaw * np.pi / 180)
        r = roll * np.pi / 180

        # X-Axis pointing right (Pitch)
        x1 = size * (np.cos(r) * np.cos(y)) + tdx
        y1 = size * (np.cos(p) * np.sin(r) + np.cos(r) * np.sin(p) * np.sin(y)) + tdy

        # Y-Axis pointing down (Yaw)
        x2 = size * (-np.sin(r) * np.cos(y)) + tdx
        y2 = size * (np.cos(p) * np.cos(r) - np.sin(p) * np.sin(r) * np.sin(y)) + tdy

        # Z-Axis pointing outward (Roll)
        x3 = size * (np.sin(y)) + tdx
        y3 = size * (-np.cos(y) * np.sin(p)) + tdy

        cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.2)
        cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.2)
        cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2, cv2.LINE_AA, tipLength=0.2)

    def draw_hud(
        self,
        frame: np.ndarray,
        scores: Dict[str, float],
        dominant_emotion: str,
        confidence: float,
        valence: float,
        is_eye_contact: bool,
        gaze_label: str
    ):
        """
        Draws semi-transparent HUD dashboard in the top-right corner.
        """
        fh, fw = frame.shape[:2]
        hud_w = 260
        hud_h = 240
        margin = 16

        x1 = fw - hud_w - margin
        y1 = margin
        x2 = fw - margin
        y2 = margin + hud_h

        if x1 < 0 or y1 < 0:
            return

        # Semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 22, 28), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 65, 80), 1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # Title
        cv2.putText(frame, "EMOTION & ATTENTION HUD", (x1 + 12, y1 + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, (220, 225, 235), 1, cv2.LINE_AA)

        # Eye Contact Indicator
        eye_color = (60, 220, 100) if is_eye_contact else (40, 120, 240)
        eye_text = "EYE CONTACT: YES" if is_eye_contact else f"GAZE: {gaze_label.upper()}"
        cv2.circle(frame, (x1 + 18, y1 + 42), 5, eye_color, -1)
        cv2.putText(frame, eye_text, (x1 + 30, y1 + 46),
                    cv2.FONT_HERSHEY_DUPLEX, 0.40, eye_color, 1, cv2.LINE_AA)

        # Valence Bar (-1.0 to 1.0)
        val_norm = (valence + 1.0) / 2.0  # [0.0, 1.0]
        val_w = int(hud_w - 40)
        val_x = x1 + 20
        val_y = y1 + 62
        cv2.rectangle(frame, (val_x, val_y), (val_x + val_w, val_y + 6), (50, 55, 65), -1)
        fill_w = int(val_norm * val_w)
        val_color = (60, 220, 100) if valence >= 0 else (50, 50, 220)
        cv2.rectangle(frame, (val_x, val_y), (val_x + fill_w, val_y + 6), val_color, -1)
        cv2.putText(frame, f"Valence: {valence:+.2f}", (val_x, val_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 185, 195), 1, cv2.LINE_AA)

        # 6 Primary emotion probability bars
        display_emotions = ["happiness", "neutral", "surprise", "sadness", "anger", "fear"]
        bar_start_y = y1 + 96
        bar_height = 8
        bar_spacing = 20

        for i, emo in enumerate(display_emotions):
            prob = scores.get(emo, 0.0)
            by = bar_start_y + i * bar_spacing
            cv2.putText(frame, emo[:7].capitalize(), (x1 + 16, by + 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (190, 195, 205), 1, cv2.LINE_AA)

            bx = x1 + 80
            bw = 110
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bar_height), (45, 50, 60), -1)
            fill_len = int(prob * bw)
            c = self.EMOTION_COLORS.get(emo, self.DEFAULT_COLOR)
            if fill_len > 0:
                cv2.rectangle(frame, (bx, by), (bx + fill_len, by + bar_height), c, -1)

            pct_text = f"{int(prob * 100)}%"
            cv2.putText(frame, pct_text, (bx + bw + 8, by + 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 185, 195), 1, cv2.LINE_AA)

    def annotate_frame(
        self,
        frame_rgb: np.ndarray,
        detection: Optional[Dict[str, Any]],
        emotion_res: Dict[str, Any],
        pose_res: Dict[str, Any],
        gaze_res: Dict[str, Any]
    ) -> np.ndarray:
        """
        Renders complete multimodal annotations on a single RGB frame.
        Returns annotated RGB frame.
        """
        # Convert to BGR for OpenCV rendering
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        dom_emo = emotion_res.get("dominant_emotion", "neutral")
        conf = emotion_res.get("confidence", 0.0)
        valence = emotion_res.get("valence", 0.0)
        scores = emotion_res.get("scores", {})

        pitch = pose_res.get("pitch", 0.0)
        yaw = pose_res.get("yaw", 0.0)
        roll = pose_res.get("roll", 0.0)

        is_contact = gaze_res.get("is_eye_contact", False)
        gaze_lbl = gaze_res.get("label", "away")

        # 1. Draw Bounding Box & Face Badges if face detected
        if detection is not None and "bbox" in detection:
            bbox = detection["bbox"]
            x1, y1, x2, y2 = bbox
            color = self.EMOTION_COLORS.get(dom_emo, self.DEFAULT_COLOR)

            self.draw_corner_box(frame_bgr, (x1, y1), (x2, y2), color, thickness=3, length=24)

            # Emotion Badge above box
            badge_text = f"{dom_emo.upper()}: {int(conf * 100)}%"
            font = cv2.FONT_HERSHEY_DUPLEX
            (tw, th), _ = cv2.getTextSize(badge_text, font, 0.55, 1)

            bx1 = x1
            by1 = max(0, y1 - th - 10)
            bx2 = min(frame_bgr.shape[1], x1 + tw + 16)
            by2 = y1

            overlay = frame_bgr.copy()
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (20, 20, 24), -1)
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, 1)
            cv2.addWeighted(overlay, 0.8, frame_bgr, 0.2, 0, frame_bgr)
            cv2.putText(frame_bgr, badge_text, (bx1 + 8, by2 - 5), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # 2. Draw 3D Euler Pose Vectors from face center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            axis_len = int(min(x2 - x1, y2 - y1) * 0.38)
            self.draw_headpose_axis(frame_bgr, pitch, yaw, roll, cx, cy, size=axis_len)

        # 3. Draw HUD Dashboard (Top-Right)
        if self.show_hud:
            self.draw_hud(frame_bgr, scores, dom_emo, conf, valence, is_contact, gaze_lbl)

        # Convert back to RGB
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
