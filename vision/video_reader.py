"""
Threaded Asynchronous Video Frame Reader (Producer-Consumer Queue).
Offloads OpenCV frame decoding and BGR2RGB color conversion to a dedicated background thread,
allowing GPU inference and video decoding to run fully in parallel.
"""

import cv2
import queue
import threading
import numpy as np
from typing import List, Tuple, Optional


class ThreadedVideoReader:
    """
    Asynchronous video reader using background thread and bounded Queue.
    Decodes video frames ahead of time so GPU inference never stalls on CPU disk/decode I/O.
    """

    _SENTINEL = object()

    def __init__(
        self,
        video_path: str,
        step: int = 1,
        max_queue_size: int = 128
    ):
        self.video_path = video_path
        self.step = max(1, step)
        self.max_queue_size = max_queue_size

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = (self.total_video_frames / self.fps) if self.fps > 0 else 0.0

        self.frame_queue: queue.Queue = queue.Queue(maxsize=self.max_queue_size)
        self.stopped = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        """Background worker thread continuously decoding frames into queue."""
        frame_idx = 0
        try:
            while not self.stopped.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    break

                if frame_idx % self.step == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    timestamp = frame_idx / self.fps
                    # Block with timeout to check stopped flag
                    while not self.stopped.is_set():
                        try:
                            self.frame_queue.put((frame_rgb, timestamp), timeout=0.1)
                            break
                        except queue.Full:
                            continue

                frame_idx += 1
        finally:
            self.frame_queue.put(self._SENTINEL)
            try:
                self.cap.release()
            except Exception:
                pass

    def read_batch(self, batch_size: int) -> Tuple[List[np.ndarray], List[float], bool]:
        """
        Reads up to batch_size frames from queue.
        Returns (batch_frames_rgb, batch_timestamps, is_finished).
        """
        batch_frames = []
        batch_timestamps = []
        is_finished = False

        while len(batch_frames) < batch_size:
            try:
                item = self.frame_queue.get(timeout=2.0)
                if item is self._SENTINEL:
                    is_finished = True
                    # Re-put sentinel so subsequent calls immediately recognize EOF
                    self.frame_queue.put(self._SENTINEL)
                    break

                frame_rgb, timestamp = item
                batch_frames.append(frame_rgb)
                batch_timestamps.append(timestamp)
            except queue.Empty:
                if not self.worker_thread.is_alive():
                    is_finished = True
                    break

        return batch_frames, batch_timestamps, is_finished

    def close(self):
        """Signals worker to stop and clears remaining queue items."""
        self.stopped.set()
        # Empty queue to unblock worker if waiting on put
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        self.worker_thread.join(timeout=1.0)
        try:
            self.cap.release()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
