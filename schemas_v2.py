"""
Pydantic Schemas for Face-motion-gpu API v2.
Sensory Perception Engine data contracts.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TurnMarker(BaseModel):
    role: str = Field(..., description="'student' hoặc 'ai'")
    start_sec: float = Field(..., ge=0.0, description="Thời điểm bắt đầu (giây)")
    end_sec: float = Field(..., ge=0.0, description="Thời điểm kết thúc (giây)")
    text: Optional[str] = Field(None, description="Nội dung câu nói nếu có")


class AnalyzeRequestV2(BaseModel):
    session_id: Optional[str] = Field(None, description="Mã phiên thực hành (ví dụ: session_123)")
    video_path: str = Field(..., description="Đường dẫn tuyệt đối hoặc tương đối tới file video/audio trên server")
    media_type: str = Field("video", description="'video' hoặc 'audio'")
    turn_markers: Optional[List[TurnMarker]] = Field(default=[], description="Danh sách mốc thời gian từng lượt nói")
    step: int = Field(6, ge=1, le=30, description="Bước nhảy frame xử lý video (mặc định 6)")
    batch_size: int = Field(32, ge=1, le=128, description="Batch size inference trên GPU (mặc định 32)")
    include_timeline_1s: bool = Field(True, description="Có trả về timeline 1 giây không")


class AnalyzeAutoRequest(BaseModel):
    video_path: str = Field(..., description="Đường dẫn tuyệt đối hoặc tương đối tới file video/audio trên server")
    session_id: Optional[str] = Field(None, description="Mã phiên thực hành (ví dụ: session_123)")
    vad_threshold: float = Field(0.3, ge=0.05, le=0.95, description="Ngưỡng nhạy kích hoạt VAD (mặc định 0.3)")
    pause_threshold_sec: float = Field(0.3, ge=0.1, le=5.0, description="Khoảng lặng dừng nói để ngắt thành một turn riêng biệt (giây, mặc định 0.3)")
    step: int = Field(6, ge=1, le=30, description="Bước nhảy frame xử lý video (mặc định 6)")
    batch_size: int = Field(32, ge=1, le=128, description="Batch size inference trên GPU (mặc định 32)")
    include_timeline_1s: bool = Field(True, description="Có trả về timeline 1 giây không")



class AnomalyMoment(BaseModel):
    start_sec: float
    end_sec: float
    duration: float
    reason: str


class TurnEvidence(BaseModel):
    turn_index: int
    role: str
    start_sec: float
    end_sec: float
    duration_sec: float
    text: Optional[str] = None
    # Vision metrics (chỉ có khi media_type == 'video')
    eye_contact_ratio: Optional[float] = None
    smile_ratio: Optional[float] = None
    dominant_emotion: Optional[str] = None
    nodding_count: Optional[int] = None
    attentive_gaze_ratio: Optional[float] = None
    # Audio metrics
    speech_rate_wpm: Optional[float] = None
    hesitation_seconds: Optional[float] = None
    is_speaking: Optional[bool] = None
    energy_mean: Optional[float] = None
    vocal_tone: Optional[str] = None


class OverallBehavioralMetrics(BaseModel):
    eye_contact_ratio: Optional[float] = None
    distracted_ratio: Optional[float] = None
    nodding_count: Optional[int] = None
    smile_ratio: Optional[float] = None
    dominant_emotions: Optional[Dict[str, float]] = None
    speech_rate_wpm: Optional[float] = None
    pitch_variance: Optional[float] = None
    speech_ratio: Optional[float] = None
    pause_ratio: Optional[float] = None
    total_hesitations_count: Optional[int] = None
    energy_mean: Optional[float] = None
    vocal_enthusiasm_ratio: Optional[float] = None
    vocal_tone: Optional[str] = None


class BehavioralAnomalies(BaseModel):
    distraction_moments: List[AnomalyMoment] = Field(default=[])
    hesitation_moments: List[AnomalyMoment] = Field(default=[])
    nodding_moments: List[Dict[str, float]] = Field(default=[])


class BehavioralEvidenceResponse(BaseModel):
    session_id: str
    status: str = "completed"
    media_type: str
    duration_seconds: float
    performance: Dict[str, Any]
    overall_metrics: OverallBehavioralMetrics
    anomalies: BehavioralAnomalies
    turn_evidence: List[TurnEvidence]
    timeline_1s: Optional[List[Dict[str, Any]]] = None
