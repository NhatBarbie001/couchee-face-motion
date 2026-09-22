"""
Annotated Video Writer & Audio Remuxer.
Encodes annotated video stream and remuxes original audio track using FFmpeg.
"""

import os
import shutil
import subprocess
import cv2
import numpy as np
from typing import Optional, Tuple


class AnnotatedVideoWriter:
    """
    High-throughput video writer that produces annotated video with original audio remuxed.
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: Tuple[int, int],
        source_video_path: Optional[str] = None
    ):
        self.output_path = os.path.abspath(output_path)
        self.fps = fps
        self.frame_size = frame_size  # (width, height)
        self.source_video_path = source_video_path

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        # Temp file for video stream before audio remuxing
        base_dir = os.path.dirname(self.output_path)
        base_name = os.path.splitext(os.path.basename(self.output_path))[0]
        self.temp_video_path = os.path.join(base_dir, f"{base_name}_temp_raw.mp4")

        # OpenCV VideoWriter with mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.temp_video_path,
            fourcc,
            self.fps,
            self.frame_size
        )
        if not self.writer.isOpened():
            # Fallback codec
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.writer = cv2.VideoWriter(
                self.temp_video_path,
                fourcc,
                self.fps,
                self.frame_size
            )

    def write_frame_rgb(self, frame_rgb: np.ndarray):
        """Writes an RGB frame to the video stream."""
        if self.writer is not None and self.writer.isOpened():
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if (frame_bgr.shape[1], frame_bgr.shape[0]) != self.frame_size:
                frame_bgr = cv2.resize(frame_bgr, self.frame_size)
            self.writer.write(frame_bgr)

    def close(self):
        """Closes video writer and remuxes audio from source video."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        if not os.path.exists(self.temp_video_path):
            return

        # Attempt remuxing audio using FFmpeg
        remux_success = False
        if self.source_video_path and os.path.exists(self.source_video_path):
            cmd = [
                "ffmpeg", "-y",
                "-i", self.temp_video_path,
                "-i", self.source_video_path,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                self.output_path
            ]
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                remux_success = True
            except Exception:
                # Direct stream copy fallback
                cmd_copy = [
                    "ffmpeg", "-y",
                    "-i", self.temp_video_path,
                    "-i", self.source_video_path,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-map", "0:v:0",
                    "-map", "1:a:0?",
                    "-shortest",
                    self.output_path
                ]
                try:
                    subprocess.run(cmd_copy, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    remux_success = True
                except Exception:
                    pass

        if not remux_success:
            # If FFmpeg failed or unavailable, just move raw temp video to output path
            if os.path.exists(self.output_path):
                os.remove(self.output_path)
            shutil.move(self.temp_video_path, self.output_path)
        else:
            if os.path.exists(self.temp_video_path):
                os.remove(self.temp_video_path)

        print(f"[ANNOTATED VIDEO SAVED] -> {self.output_path}")
