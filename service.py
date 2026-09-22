"""
MultimodalEvaluatorService: High-Performance Singleton In-Memory Inference Service.
Keeps all vision & audio models resident in GPU/CPU memory (Warm State) to eliminate cold-start overhead,
and executes Vision & Audio streams concurrently in parallel threads.
"""

import os
import sys
import time
import threading
import concurrent.futures
from typing import Dict, Any, Optional, List

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from vision.pipeline import VisionModule
from audio.pipeline import AudioModule
from fusion.feature_fusion import FeatureFusion
from fusion.turn_analyzer import ConversationalTurnAnalyzer
from fusion.temporal import TemporalAligner
from scoring.scorer import SaleScorer


class MultimodalEvaluatorService:
    """
    Production-grade Evaluator Service managing VRAM model lifecycle.
    Can be called repeatedly with zero model re-loading latency.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, device: str = "cuda"):
        self.device = device.lower()
        print("=" * 80)
        print(f"[EvaluatorService] Initializing models on device '{self.device}'...")
        t0 = time.perf_counter()

        self.vision_module = VisionModule(device=self.device)
        self.audio_module = AudioModule(device=self.device)
        self.turn_analyzer = ConversationalTurnAnalyzer()
        self.temporal_aligner = TemporalAligner(bin_size_sec=1.0)
        self.scorer = SaleScorer()

        t_load = round(time.perf_counter() - t0, 2)
        print(f"[EvaluatorService] All models warm and ready in {t_load}s.")
        print("=" * 80)

    @classmethod
    def get_instance(cls, device: str = "cuda") -> "MultimodalEvaluatorService":
        """Singleton accessor ensuring only 1 model set is loaded into GPU VRAM."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(device=device)
            return cls._instance

    def evaluate(
        self,
        video_path: str,
        audio_path: Optional[str] = None,
        output_dir: str = "results",
        session_id: Optional[str] = None,
        batch_size: int = 32,
        step: int = 2,
        save_video: bool = False,
        show_hud: bool = True,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Executes full multimodal assessment on a video.
        Runs Vision and Audio pipelines concurrently in parallel threads.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        session_id = session_id or os.path.splitext(os.path.basename(video_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        annotated_video_path = os.path.join(output_dir, f"{session_id}_annotated.mp4") if save_video else None
        audio_source = audio_path if (audio_path and os.path.exists(audio_path)) else video_path

        t_start = time.perf_counter()

        # Step 1 & 2: Concurrent Vision & Audio Execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_vision = executor.submit(
                self.vision_module.process,
                video_path=video_path,
                batch_size=batch_size,
                step=step,
                save_video=save_video,
                output_video_path=annotated_video_path,
                show_hud=show_hud,
                show_progress=show_progress
            )
            future_audio = executor.submit(
                self.audio_module.process,
                audio_source
            )
            vision_features = future_vision.result()
            audio_features = future_audio.result()

        # Step 3: Multimodal Signal Fusion
        fused_features = FeatureFusion.fuse(
            session_id=session_id,
            vision=vision_features,
            audio=audio_features
        )

        # Step 4: Conversational Turns & Deep Behavioral Attention Analysis
        attention_stats = self.turn_analyzer.analyze_attention_and_distraction(
            frame_timeline=vision_features.frame_timeline,
            total_duration_sec=vision_features.duration_seconds
        )
        turns = self.turn_analyzer.segment_conversational_turns(
            speech_segments=audio_features.speech_segments or [],
            frame_timeline=vision_features.frame_timeline,
            total_duration_sec=vision_features.duration_seconds,
            word_timestamps=audio_features.word_timestamps or []
        )
        key_behavioral_events = self.turn_analyzer.extract_key_behavioral_events(
            attention_stats=attention_stats,
            speech_segments=audio_features.speech_segments or [],
            frame_timeline=vision_features.frame_timeline
        )

        timeline_1s = self.temporal_aligner.build_timeline_1s(
            duration_sec=vision_features.duration_seconds,
            frame_timeline=vision_features.frame_timeline,
            speech_segments=audio_features.speech_segments or [],
            prosody_info={
                "pitch_mean": audio_features.pitch_mean,
                "pitch_variance": audio_features.pitch_variance
            }
        )

        # Step 5: Build Final Behavioral Metrics & Metadata
        metadata = {
            "session_id": session_id,
            "duration_seconds": round(vision_features.duration_seconds, 2),
            "total_frames_analyzed": len(vision_features.frame_timeline or []),
            "total_turns": len(turns),
            "speaking_turns": sum(1 for t in turns if t["type"] == "speaking"),
            "listening_turns": sum(1 for t in turns if t["type"] == "listening"),
            "total_words_spoken": len(audio_features.word_timestamps or []),
            "annotated_video_path": vision_features.annotated_video_path
        }

        final_behavioral_metrics = {
            "facial_emotions": {
                "happy_ratio": vision_features.happy_ratio,
                "neutral_ratio": vision_features.neutral_ratio,
                "sad_ratio": vision_features.sad_ratio,
                "angry_ratio": vision_features.angry_ratio,
                "surprise_ratio": vision_features.surprise_ratio,
                "fear_ratio": vision_features.fear_ratio,
                "disgust_ratio": vision_features.disgust_ratio,
                "negative_ratio": vision_features.negative_ratio,
                "average_valence": vision_features.average_valence
            },
            "attention_and_focus": {
                "focus_ratio": attention_stats["focus_ratio"],
                "distracted_ratio": attention_stats["distracted_ratio"],
                "longest_focus_streak_seconds": attention_stats["longest_focus_streak_seconds"],
                "longest_distraction_seconds": attention_stats["longest_distraction_seconds"],
                "longest_distraction_episode": attention_stats.get("longest_distraction_episode"),
                "eye_contact_ratio": vision_features.eye_contact_ratio,
                "nodding_count": vision_features.nodding_count
            },
            "vocal_interaction": {
                "speech_ratio": audio_features.speech_ratio,
                "pause_ratio": audio_features.pause_ratio,
                "speech_rate_wps": audio_features.speech_rate,
                "pitch_variance": audio_features.pitch_variance,
                "vocal_enthusiasm_ratio": audio_features.enthusiasm_ratio,
                "vocal_neutral_ratio": audio_features.vocal_neutral_ratio,
                "vocal_negative_ratio": audio_features.vocal_negative_ratio
            }
        }

        # Step 6: Sale Scoring Rubric Evaluation
        report = self.scorer.score_session(
            features=fused_features,
            metadata=metadata,
            final_behavioral_metrics=final_behavioral_metrics,
            key_behavioral_events=key_behavioral_events,
            conversational_turns=turns,
            timeline_1s=timeline_1s
        )

        json_path, md_path = self.scorer.save_reports(report, output_dir)
        total_time = round(time.perf_counter() - t_start, 2)
        realtime_multiplier = round(vision_features.duration_seconds / max(0.01, total_time), 2)

        report_dict = report.to_dict()
        report_dict["files"] = {
            "json_report": json_path,
            "markdown_report": md_path,
            "annotated_video": vision_features.annotated_video_path
        }
        report_dict["performance"] = {
            "execution_time_sec": total_time,
            "realtime_multiplier": f"{realtime_multiplier}x"
        }
        return report_dict

    @staticmethod
    def _classify_vocal_tone(
        pitch_variance: float,
        energy_mean: float,
        enthusiasm_ratio: float,
        hesitation_count: int = 0
    ) -> str:
        """
        Classifies vocal tone into 4 practical categories for sales coaching:
        - 'monotone': Flat pitch variance (< 2500), robotic or dull delivery
        - 'nervous': Erratic pitch variance (> 14000) with low energy or frequent hesitations
        - 'enthusiastic': High vocal energy and dynamic pitch variation
        - 'confident': Balanced pitch, steady energy and professional pace
        """
        if pitch_variance < 2500 or (energy_mean < 0.015 and pitch_variance < 4000):
            return "monotone"
        if (pitch_variance > 14000 and energy_mean < 0.035) or hesitation_count >= 3:
            return "nervous"
        if enthusiasm_ratio >= 0.20 or (pitch_variance >= 6000 and energy_mean >= 0.03):
            return "enthusiastic"
        return "confident"

    def extract_behavioral_evidence(
        self,
        video_path: str,
        turn_markers: Optional[List[Any]] = None,
        media_type: str = "video",
        step: int = 6,
        batch_size: int = 32,
        session_id: Optional[str] = None,
        include_timeline_1s: bool = True
    ) -> Dict[str, Any]:
        """
        Sensory Engine v2: Extracts objective multimodal behavioral evidence
        without hardcoded scoring, for consumption by LLMs or downstream analytics.
        Supports both 'video' and 'audio' (audio-only mode when webcam is off).
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Media file not found: {video_path}")

        session_id = session_id or os.path.splitext(os.path.basename(video_path))[0]
        media_type = (media_type or "video").lower()
        t_start = time.perf_counter()

        # =========================================================================
        # CASE A: AUDIO-ONLY MODE (Camera disabled by student)
        # =========================================================================
        if media_type == "audio":
            # Process audio only, skip GPU vision entirely
            audio_features = self.audio_module.process(video_path)
            duration_sec = round(audio_features.duration_seconds, 2)

            # Hesitations from speech segments
            hesitation_events = []
            speech_segs = audio_features.speech_segments or []
            for i in range(len(speech_segs) - 1):
                pause_len = round(speech_segs[i + 1]["start"] - speech_segs[i]["end"], 2)
                if pause_len >= 1.5:
                    hesitation_events.append({
                        "start_sec": round(speech_segs[i]["end"], 2),
                        "end_sec": round(speech_segs[i + 1]["start"], 2),
                        "duration": pause_len,
                        "reason": f"Ngập ngừng ngắt quãng kéo dài {pause_len}s"
                    })

            # Turn-by-turn evidence
            if turn_markers:
                turn_evidence = self.turn_analyzer.analyze_ground_truth_turns(
                    turn_markers=turn_markers,
                    frame_timeline=[],
                    audio_features=audio_features,
                    nodding_moments=[]
                )
            else:
                raw_turns = self.turn_analyzer.segment_conversational_turns(
                    speech_segments=speech_segs,
                    frame_timeline=[],
                    total_duration_sec=duration_sec,
                    word_timestamps=audio_features.word_timestamps or []
                )
                turn_evidence = [
                    {
                        "turn_index": t["turn_id"],
                        "role": "student" if t["type"] == "speaking" else "ai",
                        "start_sec": t["start_time"],
                        "end_sec": t["end_time"],
                        "duration_sec": t["duration"],
                        "text": t.get("text", ""),
                        "eye_contact_ratio": None,
                        "smile_ratio": None,
                        "dominant_emotion": None,
                        "nodding_count": None,
                        "attentive_gaze_ratio": None,
                        "speech_rate_wpm": audio_features.speech_rate_wpm if t["type"] == "speaking" else 0.0,
                        "hesitation_seconds": 0.0,
                        "is_speaking": t["type"] == "speaking"
                    }
                    for t in raw_turns
                ]

            vocal_tone = self._classify_vocal_tone(
                pitch_variance=audio_features.pitch_variance,
                energy_mean=audio_features.energy_mean,
                enthusiasm_ratio=audio_features.enthusiasm_ratio,
                hesitation_count=len(hesitation_events)
            )

            for t in turn_evidence:
                if t.get("role") == "student":
                    t["vocal_tone"] = vocal_tone

            total_time = round(time.perf_counter() - t_start, 2)
            realtime_multiplier = round(duration_sec / max(0.01, total_time), 2)

            return {
                "session_id": session_id,
                "status": "completed",
                "media_type": "audio",
                "duration_seconds": duration_sec,
                "performance": {
                    "execution_time_sec": total_time,
                    "realtime_multiplier": f"{realtime_multiplier}x"
                },
                "overall_metrics": {
                    "eye_contact_ratio": None,
                    "distracted_ratio": None,
                    "nodding_count": None,
                    "smile_ratio": None,
                    "dominant_emotions": None,
                    "speech_rate_wpm": round(audio_features.speech_rate_wpm, 1),
                    "pitch_variance": round(audio_features.pitch_variance, 2),
                    "speech_ratio": round(audio_features.speech_ratio, 4),
                    "pause_ratio": round(audio_features.pause_ratio, 4),
                    "total_hesitations_count": len(hesitation_events),
                    "energy_mean": round(float(audio_features.energy_mean), 4),
                    "vocal_enthusiasm_ratio": round(float(audio_features.enthusiasm_ratio), 4),
                    "vocal_tone": vocal_tone
                },
                "anomalies": {
                    "distraction_moments": [],
                    "hesitation_moments": hesitation_events,
                    "nodding_moments": []
                },
                "turn_evidence": turn_evidence,
                "timeline_1s": None
            }

        # =========================================================================
        # CASE B: FULL MULTIMODAL MODE (Video + Audio)
        # =========================================================================
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f_vision = executor.submit(
                self.vision_module.process,
                video_path=video_path,
                batch_size=batch_size,
                step=step,
                save_video=False,
                output_video_path=None,
                show_hud=False,
                show_progress=False
            )
            f_audio = executor.submit(
                self.audio_module.process,
                video_path
            )
            vision_features = f_vision.result()
            audio_features = f_audio.result()

        duration_sec = round(max(vision_features.duration_seconds, audio_features.duration_seconds), 2)

        # 1. Attention & Distraction stats
        attention_stats = self.turn_analyzer.analyze_attention_and_distraction(
            frame_timeline=vision_features.frame_timeline,
            total_duration_sec=duration_sec
        )

        # 2. Key events (Hesitations, Nodding moments, Distraction episodes)
        events = self.turn_analyzer.extract_key_behavioral_events(
            attention_stats=attention_stats,
            speech_segments=audio_features.speech_segments or [],
            frame_timeline=vision_features.frame_timeline
        )

        # Format distraction moments
        distraction_moments = []
        for ep in events.get("distraction_episodes", []):
            distraction_moments.append({
                "start_sec": ep["start_time"],
                "end_sec": ep["end_time"],
                "duration": ep["duration"],
                "reason": ep["reason"]
            })

        # Format hesitation moments
        hesitation_moments = []
        for h in events.get("hesitation_events", []):
            hesitation_moments.append({
                "start_sec": h["start_time"],
                "end_sec": h["end_time"],
                "duration": h["duration"],
                "reason": h.get("description", f"Ngập ngừng ngắt quãng {h['duration']}s")
            })

        # Format nodding moments
        nodding_moments = events.get("nodding_moments", [])

        # 3. Ground-truth turn analysis or automatic turn segmentation
        if turn_markers and len(turn_markers) > 0:
            turn_evidence = self.turn_analyzer.analyze_ground_truth_turns(
                turn_markers=turn_markers,
                frame_timeline=vision_features.frame_timeline,
                audio_features=audio_features,
                nodding_moments=nodding_moments
            )
        else:
            raw_turns = self.turn_analyzer.segment_conversational_turns(
                speech_segments=audio_features.speech_segments or [],
                frame_timeline=vision_features.frame_timeline,
                total_duration_sec=duration_sec,
                word_timestamps=audio_features.word_timestamps or []
            )
            turn_evidence = [
                {
                    "turn_index": t["turn_id"],
                    "role": "student" if t["type"] == "speaking" else "ai",
                    "start_sec": t["start_time"],
                    "end_sec": t["end_time"],
                    "duration_sec": t["duration"],
                    "text": t.get("text", ""),
                    "eye_contact_ratio": t.get("vision", {}).get("eye_contact_ratio"),
                    "smile_ratio": round(1.0 if t.get("vision", {}).get("dominant_emotion") == "happy" else 0.0, 2),
                    "dominant_emotion": t.get("vision", {}).get("dominant_emotion"),
                    "nodding_count": 0,
                    "attentive_gaze_ratio": t.get("listening", {}).get("attentiveness") if t["type"] == "listening" else None,
                    "speech_rate_wpm": audio_features.speech_rate_wpm if t["type"] == "speaking" else 0.0,
                    "hesitation_seconds": 0.0,
                    "is_speaking": t["type"] == "speaking"
                }
                for t in raw_turns
            ]

        # 4. Optional 1s Timeline for UI visualization
        timeline_1s = None
        if include_timeline_1s:
            timeline_1s = self.temporal_aligner.build_timeline_1s(
                duration_sec=duration_sec,
                frame_timeline=vision_features.frame_timeline,
                speech_segments=audio_features.speech_segments or [],
                prosody_info={
                    "pitch_mean": audio_features.pitch_mean,
                    "pitch_variance": audio_features.pitch_variance
                }
            )

        total_time = round(time.perf_counter() - t_start, 2)
        realtime_multiplier = round(duration_sec / max(0.01, total_time), 2)

        vocal_tone = self._classify_vocal_tone(
            pitch_variance=audio_features.pitch_variance,
            energy_mean=audio_features.energy_mean,
            enthusiasm_ratio=audio_features.enthusiasm_ratio,
            hesitation_count=len(hesitation_moments)
        )

        for t in turn_evidence:
            if t.get("role") == "student":
                t["vocal_tone"] = vocal_tone

        return {
            "session_id": session_id,
            "status": "completed",
            "media_type": "video",
            "duration_seconds": duration_sec,
            "performance": {
                "execution_time_sec": total_time,
                "realtime_multiplier": f"{realtime_multiplier}x"
            },
            "overall_metrics": {
                "eye_contact_ratio": round(vision_features.eye_contact_ratio, 4),
                "distracted_ratio": round(attention_stats.get("distracted_ratio", 0.0), 4),
                "nodding_count": vision_features.nodding_count,
                "smile_ratio": round(vision_features.happy_ratio, 4),
                "dominant_emotions": {
                    "happy": round(vision_features.happy_ratio, 4),
                    "neutral": round(vision_features.neutral_ratio, 4),
                    "sad": round(vision_features.sad_ratio, 4),
                    "negative": round(vision_features.negative_ratio, 4),
                    "average_valence": round(vision_features.average_valence, 4)
                },
                "speech_rate_wpm": round(audio_features.speech_rate_wpm, 1),
                "pitch_variance": round(audio_features.pitch_variance, 2),
                "speech_ratio": round(audio_features.speech_ratio, 4),
                "pause_ratio": round(audio_features.pause_ratio, 4),
                "total_hesitations_count": len(hesitation_moments),
                "energy_mean": round(float(audio_features.energy_mean), 4),
                "vocal_enthusiasm_ratio": round(float(audio_features.enthusiasm_ratio), 4),
                "vocal_tone": vocal_tone
            },
            "anomalies": {
                "distraction_moments": distraction_moments,
                "hesitation_moments": hesitation_moments,
                "nodding_moments": [
                    {"timestamp": n.get("timestamp", 0.0), "amplitude": n.get("amplitude", 0.0)}
                    for n in nodding_moments
                ]
            },
            "turn_evidence": turn_evidence,
            "timeline_1s": timeline_1s
        }

