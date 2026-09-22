"""
Sale Skill Scoring Engine (4 Pillars: Confidence, Listening, Vocal Dynamism, Warmth).
"""

import json
import os
import sys
from typing import Dict, Any, List, Optional
from .rubric import SaleAssessmentReport, CriterionScore

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from fusion.feature_fusion import FusedMultimodalFeatures
except ImportError:
    try:
        from ..fusion.feature_fusion import FusedMultimodalFeatures
    except Exception:
        FusedMultimodalFeatures = Any


class SaleScorer:
    """
    Evaluates fused multimodal features and produces actionable scoring reports.
    """

    def score_session(
        self,
        features: FusedMultimodalFeatures,
        metadata: Optional[Dict[str, Any]] = None,
        final_behavioral_metrics: Optional[Dict[str, Any]] = None,
        key_behavioral_events: Optional[Dict[str, Any]] = None,
        conversational_turns: Optional[List[Dict[str, Any]]] = None,
        timeline_1s: Optional[List[Dict[str, Any]]] = None
    ) -> SaleAssessmentReport:
        """
        Calculates 4 category scores (25 points each) -> 100 max points.
        """
        strengths = []
        improvements = []

        # =========================================================================
        # 1. CONFIDENCE & PRESENCE (25 pts)
        # =========================================================================
        # Metrics: Eye contact ratio (>= 0.70), Head center ratio (>= 0.75), low hesitations
        conf_pts = 0.0
        conf_feedback = []

        # Eye contact (max 12 pts)
        if features.eye_contact_ratio >= 0.70:
            conf_pts += 12.0
            strengths.append(f"Duy trì giao tiếp ánh mắt rất tốt ({int(features.eye_contact_ratio*100)}%), tạo cảm giác đáng tin cậy.")
        elif features.eye_contact_ratio >= 0.50:
            conf_pts += 8.0
            conf_feedback.append("Cần tăng thời lượng nhìn thẳng vào khách hàng để tăng uy tín.")
        else:
            conf_pts += 4.0
            improvements.append(f"Giao tiếp mắt còn thấp ({int(features.eye_contact_ratio*100)}%), hay đảo mắt sang chỗ khác.")

        # Head stability (max 8 pts)
        if features.head_center_ratio >= 0.75:
            conf_pts += 8.0
        elif features.head_center_ratio >= 0.55:
            conf_pts += 5.5
        else:
            conf_pts += 3.0
            conf_feedback.append("Đầu nghiêng hoặc quay qua lại nhiều khi trình bày.")

        # Speech hesitations (max 5 pts)
        if features.hesitation_count <= 1:
            conf_pts += 5.0
            strengths.append("Tự tin, mạch lạc, hầu như không có khoảng ngập ngừng lúng túng.")
        elif features.hesitation_count <= 3:
            conf_pts += 3.5
        else:
            conf_pts += 1.5
            improvements.append(f"Có {features.hesitation_count} lần ngập ngừng ngắc ngứ kéo dài trên 2 giây.")

        # =========================================================================
        # 2. ACTIVE LISTENING & EMPATHY (25 pts)
        # =========================================================================
        # Metrics: Nodding count (>= 3), Listening attentiveness, low gaze break
        listen_pts = 0.0
        listen_feedback = []

        # Nodding count (max 10 pts)
        if features.nodding_count >= 4:
            listen_pts += 10.0
            strengths.append(f"Gật đầu đồng thuận rất tốt ({features.nodding_count} lần), thể hiện sự lắng nghe và thấu hiểu.")
        elif features.nodding_count >= 2:
            listen_pts += 7.0
        else:
            listen_pts += 3.0
            improvements.append("Nên gật đầu nhẹ khi lắng nghe khách hàng để thể hiện sự quan tâm.")

        # Attentive gaze while listening (max 10 pts)
        if features.listening_attentiveness >= 0.70:
            listen_pts += 10.0
        elif features.listening_attentiveness >= 0.50:
            listen_pts += 7.0
        else:
            listen_pts += 4.0
            listen_feedback.append("Dễ mất tập trung khi khách hàng đang nói.")

        # Gaze break moderation (max 5 pts)
        if features.gaze_away_ratio <= 0.20:
            listen_pts += 5.0
        elif features.gaze_away_ratio <= 0.35:
            listen_pts += 3.5
        else:
            listen_pts += 2.0
            improvements.append(f"Tỷ lệ nhìn lơ đễnh đi chỗ khác khá cao ({int(features.gaze_away_ratio*100)}%).")

        # =========================================================================
        # 3. VOCAL DYNAMISM & ENERGY (25 pts)
        # =========================================================================
        # Metrics: Pitch variance (>= 30), Speech rate (3.0 - 4.5 wps), Enthusiasm ratio
        vocal_pts = 0.0
        vocal_feedback = []

        # Pitch variance (max 10 pts) - Avoid monotone
        if features.pitch_variance >= 35.0:
            vocal_pts += 10.0
            strengths.append("Ngữ điệu biến chuyển linh hoạt, giọng nói truyền cảm hứng và không bị đều đều.")
        elif features.pitch_variance >= 20.0:
            vocal_pts += 7.0
        else:
            vocal_pts += 3.5
            improvements.append("Giọng nói có xu hướng đều đều (monotone), thiếu điểm nhấn nhá khi chào hàng.")

        # Speaking rate (max 8 pts) - Optimal 3.0 to 4.5 words/sec
        if 2.8 <= features.speech_rate <= 4.5:
            vocal_pts += 8.0
            strengths.append(f"Tốc độ nói lý tưởng ({features.speech_rate} từ/giây), khách hàng dễ tiếp thu.")
        elif features.speech_rate < 2.8:
            vocal_pts += 5.0
            vocal_feedback.append("Tốc độ nói hơi chậm, cần chủ động đẩy nhịp hội thoại.")
        else:
            vocal_pts += 5.0
            improvements.append("Tốc độ nói quá nhanh, dễ làm khách hàng bị choáng ngợp.")

        # Vocal enthusiasm (max 7 pts)
        if features.enthusiasm_ratio >= 0.20:
            vocal_pts += 7.0
            strengths.append("Âm sắc hào hứng, nhiệt huyết khi tư vấn.")
        elif features.enthusiasm_ratio >= 0.10:
            vocal_pts += 5.0
        else:
            vocal_pts += 3.0
            vocal_feedback.append("Cần truyền tải thêm năng lượng tích cực vào giọng nói.")

        # =========================================================================
        # 4. FACIAL WARMTH & ENGAGEMENT (25 pts)
        # =========================================================================
        # Metrics: Happy ratio (>= 0.25), Valence (> 0.15), low negative ratio (< 0.08)
        warmth_pts = 0.0
        warmth_feedback = []

        # Happy ratio (max 10 pts)
        if features.happy_ratio >= 0.25:
            warmth_pts += 10.0
            strengths.append(f"Khuôn mặt tươi cười thân thiện ({int(features.happy_ratio*100)}% thời lượng).")
        elif features.happy_ratio >= 0.12:
            warmth_pts += 7.0
        else:
            warmth_pts += 4.0
            improvements.append("Gương mặt còn nghiêm nghị, nên mỉm cười chào và cảm ơn khách hàng.")

        # Average Valence (max 10 pts)
        if features.average_valence >= 0.20:
            warmth_pts += 10.0
        elif features.average_valence >= 0.05:
            warmth_pts += 7.0
        else:
            warmth_pts += 4.0
            warmth_feedback.append("Chỉ số cảm xúc trung bình còn trầm, cần tươi tắn hơn.")

        # Negative ratio (max 5 pts)
        if features.negative_ratio <= 0.05:
            warmth_pts += 5.0
        elif features.negative_ratio <= 0.15:
            warmth_pts += 3.5
        else:
            warmth_pts += 1.5
            improvements.append("Xuất hiện biểu cảm căng thẳng hoặc cau mày khi gặp từ chối.")

        # =========================================================================
        # TOTAL & GRADE
        # =========================================================================
        total = round(conf_pts + listen_pts + vocal_pts + warmth_pts, 1)

        if total >= 85.0:
            grade = "S (Xuất Sắc)"
        elif total >= 70.0:
            grade = "A (Thành Thạo)"
        elif total >= 55.0:
            grade = "B (Đạt Yêu Cầu)"
        else:
            grade = "C (Cần Rèn Luyện Thêm)"

        return SaleAssessmentReport(
            session_id=features.session_id,
            total_score=total,
            grade=grade,
            confidence_score=round(conf_pts, 1),
            active_listening_score=round(listen_pts, 1),
            vocal_dynamism_score=round(vocal_pts, 1),
            facial_warmth_score=round(warmth_pts, 1),
            strengths=strengths,
            areas_for_improvement=improvements,
            criteria_details={
                "confidence": {"score": conf_pts, "max": 25, "feedback": conf_feedback},
                "active_listening": {"score": listen_pts, "max": 25, "feedback": listen_feedback},
                "vocal_dynamism": {"score": vocal_pts, "max": 25, "feedback": vocal_feedback},
                "facial_warmth": {"score": warmth_pts, "max": 25, "feedback": warmth_feedback},
            },
            raw_features=features.to_dict(),
            metadata=metadata or {},
            final_behavioral_metrics=final_behavioral_metrics or {},
            key_behavioral_events=key_behavioral_events or {},
            conversational_turns=conversational_turns or [],
            timeline_1s=timeline_1s or []
        )

    def generate_markdown_report(self, report: SaleAssessmentReport) -> str:
        """Generates clean, executive Markdown report with detailed focus & turn statistics."""
        md = []
        md.append(f"# Báo Cáo Đánh Giá Kỹ Năng Sale - Phiên {report.session_id}\n")
        md.append(f"**Tổng Điểm:** `{report.total_score} / 100` | **Xếp Loại:** `{report.grade}`\n")
        md.append("## 1. Bảng Điểm 4 Trụ Cột Năng Lực\n")
        md.append("| Trụ Cột Đánh Giá | Điểm Đạt Được | Điểm Tối Đa | Đánh Giá Tóm Tắt |")
        md.append("| :--- | :---: | :---: | :--- |")
        md.append(f"| **1. Sự Tự Tin & Hiện Diện** | **{report.confidence_score}** | 25 | Eye Contact: {int(report.raw_features.get('eye_contact_ratio', 0)*100)}% |")
        md.append(f"| **2. Lắng Nghe Tích Cực** | **{report.active_listening_score}** | 25 | Gật đầu: {report.raw_features.get('nodding_count', 0)} lần |")
        md.append(f"| **3. Năng Lượng & Ngữ Điệu Giọng** | **{report.vocal_dynamism_score}** | 25 | Tốc độ: {report.raw_features.get('speech_rate', 0)} từ/s |")
        md.append(f"| **4. Sự Thân Thiện & Biểu Cảm** | **{report.facial_warmth_score}** | 25 | Nụ cười: {int(report.raw_features.get('happy_ratio', 0)*100)}% |")
        md.append("\n---\n")

        # 2. Attention & Emotion Distribution
        metrics = report.final_behavioral_metrics
        if metrics:
            attn = metrics.get("attention_and_focus", {})
            emo = metrics.get("facial_emotions", {})
            longest_dist = attn.get("longest_distraction_episode") or report.key_behavioral_events.get("longest_distraction")

            md.append("## 2. Phân Tích Mức Độ Tập Trung & Cảm Xúc Khuôn Mặt\n")
            md.append(f"- **Tỷ Lệ Tập Trung**: `{int(attn.get('focus_ratio', 1.0)*100)}%` (Mất tập trung / lơ đễnh: `{int(attn.get('distracted_ratio', 0.0)*100)}%`)")
            md.append(f"- **Khoảng Tập Trung Liên Tục Dài Nhất**: `{attn.get('longest_focus_streak_seconds', 0.0)} giây`")
            md.append(f"- **Khoảng Lơ Đễnh Đi Chỗ Khác Dài Nhất**: `{attn.get('longest_distraction_seconds', 0.0)} giây`")
            if longest_dist and (longest_dist.get("duration", 0) > 0 or longest_dist.get("duration_seconds", 0) > 0):
                dur_val = longest_dist.get('duration', longest_dist.get('duration_seconds'))
                md.append(f"  > *Chi tiết đợt lơ đễnh nhất*: Từ giây `{longest_dist.get('start_time')}` đến `{longest_dist.get('end_time')}` ({dur_val}s) — {longest_dist.get('reason')}.")

            md.append("\n**Phân Bố Tỷ Lệ Cảm Xúc Suốt Cuộc Gọi:**\n")
            md.append(f"- Vui vẻ / Nụ cười: `{int(emo.get('happy_ratio', 0.0)*100)}%` | Điềm tĩnh / Trung tính: `{int(emo.get('neutral_ratio', 0.0)*100)}%`")
            md.append(f"- Buồn / Thất vọng: `{int(emo.get('sad_ratio', 0.0)*100)}%` | Căng thẳng / Cau có: `{int(emo.get('angry_ratio', 0.0)*100)}%` | Tiêu cực: `{int(emo.get('negative_ratio', 0.0)*100)}%`")
            md.append(f"- Điểm tâm lý Valence trung bình: `{emo.get('average_valence', 0.0):+.2f}` (thang -1.0 đến +1.0)\n")
            md.append("---\n")

        # 3. Conversational Turns
        turns = report.conversational_turns
        if turns:
            md.append("## 3. Thống Kê Các Lượt Nói Trong Hội Thoại (Conversational Turns)\n")
            md.append("| Turn | Loại Lượt | Thời Gian (s) | Thời Lượng | Nội Dung Trao Đổi (Transcript) | Đánh Giá Tập Trung / Cảm Xúc |")
            md.append("| :---: | :---: | :---: | :---: | :--- | :--- |")
            for t in turns:
                t_type = "Học viên Nói" if t["type"] == "speaking" else "Học viên Lắng Nghe"
                t_range = f"{t['start_time']}s - {t['end_time']}s"
                t_dur = f"{t['duration']}s"
                t_text = t.get("text", "").strip()
                if t_text:
                    t_text_display = f"\"{t_text}\""
                else:
                    t_text_display = "*(Im lặng / Lắng nghe)*" if t["type"] == "listening" else "*(Không phát hiện lời thoại)*"
                vis = t.get("vision", {})
                t_eval = f"Tập trung: {int(vis.get('focus_ratio', 1.0)*100)}%, Cảm xúc: {vis.get('dominant_emotion')}"
                md.append(f"| {t['turn_id']} | **{t_type}** | {t_range} | {t_dur} | {t_text_display} | {t_eval} |")
            md.append("\n---\n")

        md.append("## 4. Điểm Mạnh Nổi Bật (Strengths)\n")
        if report.strengths:
            for s in report.strengths:
                md.append(f"- :white_check_mark: {s}")
        else:
            md.append("- Cần rèn luyện thêm để tạo dấu ấn chuyên nghiệp.")
        md.append("\n---\n")

        md.append("## 5. Khuyến Nghị Cải Thiện (Actionable Feedback)\n")
        if report.areas_for_improvement:
            for imp in report.areas_for_improvement:
                md.append(f"- :point_right: {imp}")
        else:
            md.append("- Thể hiện hoàn hảo toàn bộ các tiêu chí!")
        md.append("\n")

        return "\n".join(md)

    def save_reports(self, report: SaleAssessmentReport, output_dir: str):
        """Saves both JSON and Markdown reports to disk."""
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{report.session_id}_report.json")
        md_path = os.path.join(output_dir, f"{report.session_id}_report.md")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.generate_markdown_report(report))

        print(f"[REPORT SAVED] JSON: {json_path}")
        print(f"[REPORT SAVED] Markdown: {md_path}")
        return json_path, md_path
