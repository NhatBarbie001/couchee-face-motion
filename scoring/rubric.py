"""
Sale Evaluation Rubric and Assessment Benchmarks.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class CriterionScore:
    name: str
    score: float
    max_score: float
    percentage: float
    feedback: List[str] = field(default_factory=list)


@dataclass
class SaleAssessmentReport:
    session_id: str
    total_score: float
    max_score: float = 100.0
    grade: str = "B"
    confidence_score: float = 0.0
    active_listening_score: float = 0.0
    vocal_dynamism_score: float = 0.0
    facial_warmth_score: float = 0.0
    strengths: List[str] = field(default_factory=list)
    areas_for_improvement: List[str] = field(default_factory=list)
    criteria_details: Dict[str, Any] = field(default_factory=dict)
    raw_features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    final_behavioral_metrics: Dict[str, Any] = field(default_factory=dict)
    key_behavioral_events: Dict[str, Any] = field(default_factory=dict)
    conversational_turns: List[Dict[str, Any]] = field(default_factory=list)
    timeline_1s: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "metadata": self.metadata,
            "overall_evaluation": {
                "total_score": round(self.total_score, 1),
                "max_score": self.max_score,
                "grade": self.grade,
                "breakdown": {
                    "confidence_score": round(self.confidence_score, 1),
                    "active_listening_score": round(self.active_listening_score, 1),
                    "vocal_dynamism_score": round(self.vocal_dynamism_score, 1),
                    "facial_warmth_score": round(self.facial_warmth_score, 1),
                },
                "strengths": self.strengths,
                "areas_for_improvement": self.areas_for_improvement,
                "criteria_details": self.criteria_details,
            },
            "final_behavioral_metrics": self.final_behavioral_metrics,
            "key_behavioral_events": self.key_behavioral_events,
            "conversational_turns": self.conversational_turns,
            "timeline_1s": self.timeline_1s
        }
