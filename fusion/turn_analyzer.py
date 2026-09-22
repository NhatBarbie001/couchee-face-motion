"""
Conversational Turn & Deep Behavioral Attention Analyzer.
Calculates:
- Longest distraction episode combining 3D Head Pose and Eye Gaze
- Distraction episodes >= 2.5s and longest focus streak
- Turn-by-turn conversation analysis (Speaking vs Listening turns)
- Key behavioral events (Hesitations, Nodding moments, Distractions)
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class ConversationalTurnAnalyzer:
    """
    Orchestrates temporal turn segmentation, multimodal attention tracking,
    and granular event extraction.
    """

    def __init__(
        self,
        min_distraction_sec: float = 2.5,
        turn_pause_threshold_sec: float = 1.2,
        yaw_distraction_thresh: float = 18.0,
        pitch_down_thresh: float = -14.0
    ):
        self.min_distraction_sec = min_distraction_sec
        self.turn_pause_threshold_sec = turn_pause_threshold_sec
        self.yaw_distraction_thresh = yaw_distraction_thresh
        self.pitch_down_thresh = pitch_down_thresh

    def analyze_attention_and_distraction(
        self,
        frame_timeline: List[Dict[str, Any]],
        total_duration_sec: float
    ) -> Dict[str, Any]:
        """
        Analyzes consecutive distraction frames by combining 3D Head Pose and Eye Gaze.
        Returns:
            - longest_distraction_seconds
            - longest_distraction_episode (start, end, duration, reason, dominant angles)
            - distraction_episodes (all episodes >= min_distraction_sec)
            - focus_ratio & distracted_ratio
            - longest_focus_streak_seconds
        """
        if not frame_timeline:
            return {
                "focus_ratio": 1.0,
                "distracted_ratio": 0.0,
                "longest_focus_streak_seconds": round(total_duration_sec, 2),
                "longest_distraction_seconds": 0.0,
                "longest_distraction_episode": None,
                "distraction_episodes": []
            }

        total_frames = len(frame_timeline)
        distracted_frames_count = 0

        # Scan for distraction and focus streaks
        episodes: List[Dict[str, Any]] = []
        current_episode_frames: List[Dict[str, Any]] = []

        longest_focus_frames = 0
        current_focus_frames = 0

        for f in frame_timeline:
            is_distracted = f.get("is_distracted", False)
            if is_distracted:
                distracted_frames_count += 1
                current_episode_frames.append(f)
                if current_focus_frames > longest_focus_frames:
                    longest_focus_frames = current_focus_frames
                current_focus_frames = 0
            else:
                current_focus_frames += 1
                if current_episode_frames:
                    episodes.append(self._format_distraction_episode(current_episode_frames))
                    current_episode_frames = []

        if current_focus_frames > longest_focus_frames:
            longest_focus_frames = current_focus_frames

        if current_episode_frames:
            episodes.append(self._format_distraction_episode(current_episode_frames))

        frame_interval = total_duration_sec / max(1, total_frames)
        longest_focus_sec = round(longest_focus_frames * frame_interval, 2)

        # Find the single longest distraction episode
        longest_episode = None
        longest_distraction_sec = 0.0
        if episodes:
            longest_episode = max(episodes, key=lambda e: e["duration"])
            longest_distraction_sec = longest_episode["duration"]

        # Filter prominent episodes >= min_distraction_sec
        significant_episodes = [
            e for e in episodes if e["duration"] >= self.min_distraction_sec
        ]

        distracted_ratio = round(distracted_frames_count / total_frames, 4)
        focus_ratio = round(max(0.0, 1.0 - distracted_ratio), 4)

        return {
            "focus_ratio": focus_ratio,
            "distracted_ratio": distracted_ratio,
            "longest_focus_streak_seconds": longest_focus_sec,
            "longest_distraction_seconds": longest_distraction_sec,
            "longest_distraction_episode": longest_episode,
            "distraction_episodes": significant_episodes
        }

    def _format_distraction_episode(
        self,
        frames: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Formats an episode of consecutive distracted frames."""
        t_start = frames[0]["timestamp"]
        t_end = frames[-1]["timestamp"]
        dur = round(max(0.1, t_end - t_start), 2)

        avg_yaw = float(np.mean([f.get("yaw", 0.0) for f in frames]))
        avg_pitch = float(np.mean([f.get("pitch", 0.0) for f in frames]))

        # Count gaze directions
        gaze_labels = [f.get("gaze_label", "away") for f in frames]
        dom_gaze = max(set(gaze_labels), key=gaze_labels.count) if gaze_labels else "away"

        # Determine human-friendly explanation
        reasons = []
        if avg_yaw < -self.yaw_distraction_thresh:
            reasons.append(f"Quay đầu sang trái (Yaw: {avg_yaw:.1f}°)")
        elif avg_yaw > self.yaw_distraction_thresh:
            reasons.append(f"Quay đầu sang phải (Yaw: {avg_yaw:.1f}°)")

        if avg_pitch < self.pitch_down_thresh:
            reasons.append(f"Cúi đầu nhìn xuống (Pitch: {avg_pitch:.1f}°)")

        if dom_gaze in ["left", "right", "down", "away"]:
            reasons.append(f"Mắt hướng về '{dom_gaze}'")

        reason_str = "; ".join(reasons) if reasons else "Không duy trì giao tiếp mắt trực tiếp"

        return {
            "start_time": t_start,
            "end_time": t_end,
            "duration": dur,
            "reason": reason_str,
            "head_yaw": round(avg_yaw, 1),
            "head_pitch": round(avg_pitch, 1),
            "gaze_direction": dom_gaze
        }

    def segment_conversational_turns(
        self,
        speech_segments: List[Dict[str, Any]],
        frame_timeline: List[Dict[str, Any]],
        total_duration_sec: float,
        word_timestamps: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Groups speech into speaking turns and listening turns, merging short pauses,
        and assigns transcribed text with timestamps to each speaking turn.
        """
        word_timestamps = word_timestamps or []
        if not speech_segments and word_timestamps:
            # Reconstruct speech segments from word timestamp clusters (ASR-guided fallback)
            cur_start = word_timestamps[0]["start"]
            cur_end = word_timestamps[0]["end"]
            for w in word_timestamps[1:]:
                if w["start"] - cur_end < self.turn_pause_threshold_sec:
                    cur_end = max(cur_end, w["end"])
                else:
                    speech_segments.append({
                        "start": round(cur_start, 2),
                        "end": round(cur_end, 2),
                        "duration": round(cur_end - cur_start, 2),
                        "speech": True
                    })
                    cur_start = w["start"]
                    cur_end = w["end"]
            speech_segments.append({
                "start": round(cur_start, 2),
                "end": round(cur_end, 2),
                "duration": round(cur_end - cur_start, 2),
                "speech": True
            })

        if not speech_segments:
            # Entire video is silence or listening
            turn = self._build_turn_data(
                turn_id=1,
                turn_type="listening",
                t_start=0.0,
                t_end=total_duration_sec,
                frame_timeline=frame_timeline,
                word_timestamps=word_timestamps
            )
            return [turn]

        # 1. Merge speech segments separated by < turn_pause_threshold_sec into Speaking Turns
        raw_speaking_spans: List[Tuple[float, float]] = []
        cur_start = speech_segments[0]["start"]
        cur_end = speech_segments[0]["end"]

        for seg in speech_segments[1:]:
            gap = seg["start"] - cur_end
            if gap < self.turn_pause_threshold_sec:
                cur_end = seg["end"]
            else:
                raw_speaking_spans.append((cur_start, cur_end))
                cur_start = seg["start"]
                cur_end = seg["end"]
        raw_speaking_spans.append((cur_start, cur_end))

        # 2. Interleave Speaking Turns and Listening Turns
        turns: List[Dict[str, Any]] = []
        turn_id = 1
        last_t = 0.0

        for sp_start, sp_end in raw_speaking_spans:
            # Listening turn before speaking if gap >= 1.0s
            if sp_start - last_t >= 1.0:
                listening_turn = self._build_turn_data(
                    turn_id=turn_id,
                    turn_type="listening",
                    t_start=last_t,
                    t_end=sp_start,
                    frame_timeline=frame_timeline,
                    word_timestamps=word_timestamps
                )
                turns.append(listening_turn)
                turn_id += 1

            # Speaking turn
            speaking_turn = self._build_turn_data(
                turn_id=turn_id,
                turn_type="speaking",
                t_start=sp_start,
                t_end=sp_end,
                frame_timeline=frame_timeline,
                word_timestamps=word_timestamps
            )
            turns.append(speaking_turn)
            turn_id += 1
            last_t = sp_end

        # Trailing listening turn if video continues after speaking
        if total_duration_sec - last_t >= 1.0:
            trailing_turn = self._build_turn_data(
                turn_id=turn_id,
                turn_type="listening",
                t_start=last_t,
                t_end=total_duration_sec,
                frame_timeline=frame_timeline,
                word_timestamps=word_timestamps
            )
            turns.append(trailing_turn)

        return turns

    def _build_turn_data(
        self,
        turn_id: int,
        turn_type: str,
        t_start: float,
        t_end: float,
        frame_timeline: List[Dict[str, Any]],
        word_timestamps: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Calculates vision and vocal metrics within a single turn time-window."""
        dur = round(max(0.1, t_end - t_start), 2)
        turn_frames = [
            f for f in frame_timeline if t_start <= f["timestamp"] <= t_end
        ]

        # Vision metrics in this turn
        if turn_frames:
            focused_count = sum(1 for f in turn_frames if not f.get("is_distracted", False))
            eye_contact_count = sum(1 for f in turn_frames if f.get("is_eye_contact", False))
            focus_ratio = round(focused_count / len(turn_frames), 4)
            eye_contact_ratio = round(eye_contact_count / len(turn_frames), 4)

            emotions = [f.get("dominant_emotion", "neutral") for f in turn_frames]
            dom_emo = max(set(emotions), key=emotions.count) if emotions else "neutral"
            valences = [f.get("valence", 0.0) for f in turn_frames]
            avg_valence = round(float(np.mean(valences)), 4)
        else:
            focus_ratio = 1.0
            eye_contact_ratio = 1.0
            dom_emo = "neutral"
            avg_valence = 0.0

        turn_dict = {
            "turn_id": turn_id,
            "type": turn_type,
            "start_time": round(t_start, 2),
            "end_time": round(t_end, 2),
            "duration": dur,
            "text": "",
            "vision": {
                "focus_ratio": focus_ratio,
                "eye_contact_ratio": eye_contact_ratio,
                "dominant_emotion": dom_emo,
                "average_valence": avg_valence
            }
        }

        if turn_type == "speaking":
            words_in_turn = []
            if word_timestamps:
                for w in word_timestamps:
                    w_start = w.get("start", 0.0)
                    w_end = w.get("end", 0.0)
                    if w_end >= (t_start - 0.25) and w_start <= (t_end + 0.25):
                        words_in_turn.append(w)
            turn_dict["text"] = " ".join(w["word"] for w in words_in_turn).strip()
            turn_dict["audio"] = {
                "is_speaking": True,
                "dominant_tone": "enthusiasm" if avg_valence > 0.2 else "neutral",
                "word_count": len(words_in_turn)
            }
        else:
            turn_dict["text"] = ""
            turn_dict["listening"] = {
                "attentiveness": focus_ratio,
                "attentive_label": "High" if focus_ratio >= 0.75 else "Moderate"
            }

        return turn_dict

    def extract_key_behavioral_events(
        self,
        attention_stats: Dict[str, Any],
        speech_segments: Optional[List[Dict[str, Any]]] = None,
        frame_timeline: Optional[List[Dict[str, Any]]] = None,
        nodding_min_prominence: float = 11.0
    ) -> Dict[str, Any]:
        """
        Gathers key behavioral events:
        - Longest distraction episode (with reason, yaw/pitch, gaze, start/end)
        - Distraction episodes (>= min_distraction_sec)
        - Hesitation pauses (>= 2.0s)
        - Nodding moments
        """
        speech_segments = speech_segments or []
        frame_timeline = frame_timeline or []

        # 1. Hesitations (pauses >= 2.0s between speech chunks)
        hesitations = []
        for i in range(len(speech_segments) - 1):
            pause_len = round(speech_segments[i + 1]["start"] - speech_segments[i]["end"], 2)
            if pause_len >= 2.0:
                hesitations.append({
                    "start_time": round(speech_segments[i]["end"], 2),
                    "end_time": round(speech_segments[i + 1]["start"], 2),
                    "duration": pause_len,
                    "description": f"Ngập ngừng ngắt quãng kéo dài {pause_len}s"
                })

        # 2. Nodding moments from frame_timeline
        nodding_moments = []
        if len(frame_timeline) >= 6:
            pitch_series = [f.get("pitch", 0.0) for f in frame_timeline]
            diffs = np.diff(pitch_series)
            sign_changes = np.where(np.diff(np.sign(diffs)))[0]

            i = 0
            while i < len(sign_changes):
                idx = sign_changes[i]
                start_f = max(0, idx - 2)
                end_f = min(len(pitch_series), idx + 3)
                amp = float(np.max(pitch_series[start_f:end_f]) - np.min(pitch_series[start_f:end_f]))
                if amp >= nodding_min_prominence:
                    t_nod = frame_timeline[idx]["timestamp"]
                    nodding_moments.append({
                        "timestamp": round(t_nod, 2),
                        "amplitude": round(amp, 1),
                        "type": "head_nod"
                    })
                    i += 2
                else:
                    i += 1

        longest_dist = attention_stats.get("longest_distraction_episode")
        if longest_dist:
            longest_dist_formatted = {
                "duration_seconds": longest_dist["duration"],
                "start_time": longest_dist["start_time"],
                "end_time": longest_dist["end_time"],
                "reason": longest_dist["reason"],
                "head_yaw": longest_dist.get("head_yaw", 0.0),
                "head_pitch": longest_dist.get("head_pitch", 0.0),
                "gaze_direction": longest_dist.get("gaze_direction", "away")
            }
        else:
            longest_dist_formatted = {
                "duration_seconds": 0.0,
                "start_time": 0.0,
                "end_time": 0.0,
                "reason": "Duy trì tập trung liên tục",
                "head_yaw": 0.0,
                "head_pitch": 0.0,
                "gaze_direction": "camera"
            }

        return {
            "longest_distraction": longest_dist_formatted,
            "distraction_episodes": attention_stats.get("distraction_episodes", []),
            "hesitation_events": hesitations,
            "nodding_moments": nodding_moments
        }

    def analyze_ground_truth_turns(
        self,
        turn_markers: List[Any],
        frame_timeline: List[Dict[str, Any]],
        audio_features: Optional[Any] = None,
        nodding_moments: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyzes multimodal behavior aligned directly with ground-truth dialogue turns
        provided by caller (e.g. Couchee session.videoTurnMarkers).
        """
        results = []
        frame_timeline = frame_timeline or []
        nodding_moments = nodding_moments or []

        word_timestamps = getattr(audio_features, "word_timestamps", None) or []
        speech_segments = getattr(audio_features, "speech_segments", None) or []

        for idx, marker in enumerate(turn_markers):
            # Support both Pydantic model and dict
            if hasattr(marker, "dict"):
                m_data = marker.dict()
            elif isinstance(marker, dict):
                m_data = marker
            else:
                m_data = {
                    "role": getattr(marker, "role", "student"),
                    "start_sec": getattr(marker, "start_sec", 0.0),
                    "end_sec": getattr(marker, "end_sec", 0.0),
                    "text": getattr(marker, "text", None)
                }

            role = str(m_data.get("role", "student")).lower()
            t_start = float(m_data.get("start_sec", 0.0))
            t_end = float(m_data.get("end_sec", 0.0))
            dur = round(max(0.1, t_end - t_start), 2)
            turn_text = m_data.get("text")

            # Slice frames in this turn
            turn_frames = [
                f for f in frame_timeline if t_start <= f.get("timestamp", 0.0) <= t_end
            ]

            turn_item: Dict[str, Any] = {
                "turn_index": idx + 1,
                "role": role,
                "start_sec": round(t_start, 2),
                "end_sec": round(t_end, 2),
                "duration_sec": dur,
                "text": turn_text,
                "eye_contact_ratio": None,
                "smile_ratio": None,
                "dominant_emotion": None,
                "nodding_count": None,
                "attentive_gaze_ratio": None,
                "speech_rate_wpm": None,
                "hesitation_seconds": 0.0,
                "is_speaking": None,
                "energy_mean": None,
                "vocal_tone": None
            }

            if turn_frames:
                eye_contact_count = sum(1 for f in turn_frames if f.get("is_eye_contact", False))
                focused_count = sum(1 for f in turn_frames if not f.get("is_distracted", False))
                happy_count = sum(1 for f in turn_frames if f.get("dominant_emotion") == "happy" or f.get("valence", 0.0) > 0.3)

                emotions = [f.get("dominant_emotion", "neutral") for f in turn_frames]
                dom_emo = max(set(emotions), key=emotions.count) if emotions else "neutral"

                eye_contact_ratio = round(eye_contact_count / len(turn_frames), 4)
                attentive_ratio = round(focused_count / len(turn_frames), 4)
                smile_ratio = round(happy_count / len(turn_frames), 4)
            else:
                eye_contact_ratio = 1.0
                attentive_ratio = 1.0
                smile_ratio = 0.0
                dom_emo = "neutral"

            # Nodding moments during this turn
            nods_in_turn = sum(
                1 for n in nodding_moments if t_start <= n.get("timestamp", 0.0) <= t_end
            )

            # Speech and words in this turn
            words_in_turn = [
                w for w in word_timestamps
                if (w.get("end", 0.0) >= (t_start - 0.25) and w.get("start", 0.0) <= (t_end + 0.25))
            ]
            if not turn_text and words_in_turn:
                turn_item["text"] = " ".join(w.get("word", "") for w in words_in_turn).strip()

            word_count = len(words_in_turn) if words_in_turn else (len(turn_item["text"].split()) if turn_item["text"] else 0)
            minute_len = max(0.05, dur / 60.0)
            calculated_wpm = round(word_count / minute_len, 1)

            # Hesitations within this turn
            turn_hesitations_sec = 0.0
            turn_segments = [
                s for s in speech_segments
                if s.get("start", 0.0) >= (t_start - 0.5) and s.get("end", 0.0) <= (t_end + 0.5)
            ]
            for s_i in range(len(turn_segments) - 1):
                gap = round(turn_segments[s_i + 1]["start"] - turn_segments[s_i]["end"], 2)
                if gap >= 1.5:
                    turn_hesitations_sec += gap

            if role == "student":
                turn_item["eye_contact_ratio"] = eye_contact_ratio
                turn_item["smile_ratio"] = smile_ratio
                turn_item["dominant_emotion"] = dom_emo
                turn_item["speech_rate_wpm"] = calculated_wpm
                turn_item["hesitation_seconds"] = round(turn_hesitations_sec, 2)
                turn_item["is_speaking"] = word_count > 0 or len(turn_segments) > 0
                turn_item["energy_mean"] = round(float(getattr(audio_features, "energy_mean", 0.0) or 0.0), 4)
            else:
                # AI turn: student is listening (do cả nụ cười, giao tiếp mắt và gật đầu khi lắng nghe)
                turn_item["eye_contact_ratio"] = eye_contact_ratio
                turn_item["smile_ratio"] = smile_ratio
                turn_item["dominant_emotion"] = dom_emo
                turn_item["attentive_gaze_ratio"] = attentive_ratio
                turn_item["nodding_count"] = nods_in_turn
                turn_item["is_speaking"] = False

            results.append(turn_item)

        return results


