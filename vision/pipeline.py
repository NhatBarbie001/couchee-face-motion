"""
VisionModule: Standalone Multimodal Vision Pipeline for High-Throughput Video Processing.
Coordinates SCRFD + EmotiEffLib + 6DRepNet + Gaze-LLE.
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from .face_detection import BatchFaceDetector
from .face_emotion import BatchFaceEmotionAnalyzer
from .head_pose import BatchHeadPoseAnalyzer
from .eye_gaze import BatchEyeGazeAnalyzer
from .annotator import MultimodalAnnotator
from .video_writer import AnnotatedVideoWriter
from .video_reader import ThreadedVideoReader


@dataclass
class VisionFeatures:
    """Standardized conversation-level vision features for Sale AI."""
    eye_contact_ratio: float
    gaze_away_ratio: float
    head_center_ratio: float
    happy_ratio: float
    neutral_ratio: float
    sad_ratio: float
    angry_ratio: float
    surprise_ratio: float
    fear_ratio: float
    disgust_ratio: float
    negative_ratio: float
    average_valence: float
    head_left_ratio: float
    head_right_ratio: float
    head_down_ratio: float
    head_movement_rate: float
    nodding_count: int
    longest_eye_contact: float
    gaze_break_count: int
    total_frames_analyzed: int
    duration_seconds: float
    no_face_count: int = 0
    no_face_duration_sec: float = 0.0
    no_face_ratio: float = 0.0
    no_face_episodes: Optional[List[Dict[str, Any]]] = None
    annotated_video_path: Optional[str] = None
    frame_timeline: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VisionModule:
    """
    High-Throughput Vision Pipeline orchestrator.
    Runs batch inference across SCRFD, EmotiEffLib, 6DRepNet, and Gaze-LLE.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        print(f"[Vision GPU] Initializing vision pipeline on '{device}'...")

        self.face_detector = BatchFaceDetector(device=device)
        self.emotion_analyzer = BatchFaceEmotionAnalyzer(device=device)
        self.head_pose_analyzer = BatchHeadPoseAnalyzer(device=device)
        self.gaze_analyzer = BatchEyeGazeAnalyzer()
        print("[Vision GPU] All 4 vision models loaded successfully.")

    def process(
        self,
        video_path: str,
        batch_size: int = 32,
        step: int = 2,
        save_video: bool = False,
        output_video_path: Optional[str] = None,
        show_hud: bool = True,
        show_progress: bool = True
    ) -> VisionFeatures:
        """
        Processes a video file in batches and returns high-level VisionFeatures.
        Optionally renders and saves an annotated video with original audio remuxed.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        with ThreadedVideoReader(video_path, step=step) as reader:
            fps = reader.fps
            width = reader.width
            height = reader.height
            total_video_frames = reader.total_video_frames
            duration_sec = reader.duration_sec
            out_fps = max(1.0, fps / max(1, step))

            # Setup video writer if save_video is requested
            video_writer = None
            annotator = None
            if save_video:
                if output_video_path is None:
                    base, _ = os.path.splitext(video_path)
                    output_video_path = f"{base}_annotated.mp4"
                annotator = MultimodalAnnotator(show_hud=show_hud)
                video_writer = AnnotatedVideoWriter(
                    output_path=output_video_path,
                    fps=out_fps,
                    frame_size=(width, height),
                    source_video_path=video_path
                )

            # Collect batches
            all_emotion_results: List[Dict[str, Any]] = []
            all_pose_results: List[Dict[str, Any]] = []
            all_gaze_results: List[Dict[str, Any]] = []
            all_frame_timeline: List[Dict[str, Any]] = []

            pbar = tqdm(
                total=total_video_frames // step,
                desc="[Vision GPU] Processing Frames",
                disable=not show_progress
            )

            while True:
                batch_frames, batch_timestamps, is_finished = reader.read_batch(batch_size)
                if batch_frames:
                    self._process_batch(
                        batch_frames,
                        batch_timestamps,
                        all_emotion_results,
                        all_pose_results,
                        all_gaze_results,
                        all_frame_timeline,
                        annotator=annotator,
                        video_writer=video_writer
                    )
                    pbar.update(len(batch_frames))

                if is_finished:
                    break

            pbar.close()

        # Close video writer & remux audio
        if video_writer is not None:
            video_writer.close()

        # Aggregate metrics
        emotion_stats = self.emotion_analyzer.aggregate_session(all_emotion_results)
        pose_stats = self.head_pose_analyzer.aggregate_session(all_pose_results, fps=fps)
        gaze_stats = self.gaze_analyzer.aggregate_session(all_gaze_results, fps=fps, step=step)

        # Aggregate no-face episodes
        no_face_frames = [f for f in all_frame_timeline if not f.get("face_detected", True)]
        total_f = max(1, len(all_frame_timeline))
        no_face_ratio = round(len(no_face_frames) / total_f, 4)

        no_face_episodes = []
        cur_episode = []
        for f in all_frame_timeline:
            if not f.get("face_detected", True):
                cur_episode.append(f)
            else:
                if cur_episode:
                    t_start = cur_episode[0]["timestamp"]
                    t_end = cur_episode[-1]["timestamp"]
                    dur = round(max(0.1, t_end - t_start), 2)
                    if dur >= 0.5:
                        no_face_episodes.append({
                            "start_sec": t_start,
                            "end_sec": t_end,
                            "duration": dur,
                            "reason": f"Không phát hiện khuôn mặt trong {dur}s"
                        })
                    cur_episode = []
        if cur_episode:
            t_start = cur_episode[0]["timestamp"]
            t_end = cur_episode[-1]["timestamp"]
            dur = round(max(0.1, t_end - t_start), 2)
            if dur >= 0.5:
                no_face_episodes.append({
                    "start_sec": t_start,
                    "end_sec": t_end,
                    "duration": dur,
                    "reason": f"Không phát hiện khuôn mặt trong {dur}s"
                })

        no_face_duration_sec = round(sum(ep["duration"] for ep in no_face_episodes), 2) if no_face_episodes else round(len(no_face_frames) * (duration_sec / total_f), 2)
        no_face_count = len(no_face_episodes)

        return VisionFeatures(
            eye_contact_ratio=gaze_stats["eye_contact_ratio"],
            gaze_away_ratio=gaze_stats["gaze_away_ratio"],
            head_center_ratio=pose_stats["head_center_ratio"],
            happy_ratio=emotion_stats["happy_ratio"],
            neutral_ratio=emotion_stats["neutral_ratio"],
            sad_ratio=emotion_stats.get("sad_ratio", 0.0),
            angry_ratio=emotion_stats.get("angry_ratio", 0.0),
            surprise_ratio=emotion_stats.get("surprise_ratio", 0.0),
            fear_ratio=emotion_stats.get("fear_ratio", 0.0),
            disgust_ratio=emotion_stats.get("disgust_ratio", 0.0),
            negative_ratio=emotion_stats["negative_ratio"],
            average_valence=emotion_stats["average_valence"],
            head_left_ratio=pose_stats["head_left_ratio"],
            head_right_ratio=pose_stats["head_right_ratio"],
            head_down_ratio=pose_stats["head_down_ratio"],
            head_movement_rate=pose_stats["head_movement_rate"],
            nodding_count=pose_stats["nodding_count"],
            longest_eye_contact=gaze_stats["longest_eye_contact"],
            gaze_break_count=gaze_stats["gaze_break_count"],
            total_frames_analyzed=len(all_emotion_results),
            duration_seconds=round(duration_sec, 2),
            no_face_count=no_face_count,
            no_face_duration_sec=no_face_duration_sec,
            no_face_ratio=no_face_ratio,
            no_face_episodes=no_face_episodes,
            annotated_video_path=output_video_path if save_video else None,
            frame_timeline=all_frame_timeline
        )

    def _process_batch(
        self,
        batch_frames_rgb: List[np.ndarray],
        batch_timestamps: List[float],
        all_emotion_results: List[Dict[str, Any]],
        all_pose_results: List[Dict[str, Any]],
        all_gaze_results: List[Dict[str, Any]],
        all_frame_timeline: List[Dict[str, Any]],
        annotator: Optional[MultimodalAnnotator] = None,
        video_writer: Optional[AnnotatedVideoWriter] = None
    ):
        """Processes a single batch of RGB frames through the 4 vision models and optionally annotates."""
        # 1. Detect faces and extract crops
        detections = self.face_detector.process_frames(batch_frames_rgb)

        valid_crops = []
        valid_indices = []
        for i, det in enumerate(detections):
            if det is not None:
                valid_crops.append(det["face_crop"])
                valid_indices.append(i)

        if not valid_crops:
            # No face detected in this batch
            for idx, frame in enumerate(batch_frames_rgb):
                t = batch_timestamps[idx] if idx < len(batch_timestamps) else 0.0
                emo_dict = {"dominant_emotion": "neutral", "confidence": 1.0, "valence": 0.0, "scores": {"neutral": 1.0}}
                pose_dict = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "yaw_label": "center", "pitch_label": "center", "is_center": True}
                gaze_dict = {"is_eye_contact": False, "label": "away"}
                all_emotion_results.append(emo_dict)
                all_pose_results.append(pose_dict)
                all_gaze_results.append(gaze_dict)
                all_frame_timeline.append({
                    "timestamp": round(t, 3),
                    "face_detected": False,
                    "bbox": None,
                    "dominant_emotion": "neutral",
                    "valence": 0.0,
                    "scores": {"neutral": 1.0},
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "roll": 0.0,
                    "is_eye_contact": False,
                    "gaze_label": "away",
                    "is_distracted": True
                })
                if video_writer is not None and annotator is not None:
                    annotated = annotator.annotate_frame(frame, None, emo_dict, pose_dict, gaze_dict)
                    video_writer.write_frame_rgb(annotated)
            return

        # 2. Parallel model inference on GPU
        emotions = self.emotion_analyzer.analyze_batch(valid_crops)
        poses = self.head_pose_analyzer.analyze_batch(valid_crops)
        gazes = self.gaze_analyzer.analyze_batch(valid_crops, poses)

        # 3. Align batch back to original sequence & write frames
        crop_ptr = 0
        for i, frame in enumerate(batch_frames_rgb):
            t = batch_timestamps[i] if i < len(batch_timestamps) else 0.0
            if i in valid_indices:
                cur_emo = emotions[crop_ptr]
                cur_pose = poses[crop_ptr]
                cur_gaze = gazes[crop_ptr]
                cur_det = detections[i]
                crop_ptr += 1
            else:
                cur_emo = {"dominant_emotion": "neutral", "confidence": 1.0, "valence": 0.0, "scores": {"neutral": 1.0}}
                cur_pose = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "yaw_label": "center", "pitch_label": "center", "is_center": True}
                cur_gaze = {"is_eye_contact": False, "label": "away"}
                cur_det = None

            all_emotion_results.append(cur_emo)
            all_pose_results.append(cur_pose)
            all_gaze_results.append(cur_gaze)

            is_contact = cur_gaze.get("is_eye_contact", False)
            yaw = cur_pose.get("yaw", 0.0)
            pitch = cur_pose.get("pitch", 0.0)
            is_distracted = (not is_contact) or (abs(yaw) > 18.0) or (pitch < -14.0)

            all_frame_timeline.append({
                "timestamp": round(t, 3),
                "face_detected": cur_det is not None,
                "bbox": cur_det["bbox"] if cur_det else None,
                "dominant_emotion": cur_emo.get("dominant_emotion", "neutral"),
                "valence": cur_emo.get("valence", 0.0),
                "scores": cur_emo.get("scores", {}),
                "pitch": pitch,
                "yaw": yaw,
                "roll": cur_pose.get("roll", 0.0),
                "is_eye_contact": is_contact,
                "gaze_label": cur_gaze.get("label", "away"),
                "is_distracted": is_distracted
            })

            if video_writer is not None and annotator is not None:
                annotated = annotator.annotate_frame(frame, cur_det, cur_emo, cur_pose, cur_gaze)
                video_writer.write_frame_rgb(annotated)
