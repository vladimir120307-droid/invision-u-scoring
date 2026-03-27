from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class EducationLevel(str, Enum):
    SCHOOL = "school"
    BACHELOR = "bachelor"
    MASTER = "master"
    PHD = "phd"
    OTHER = "other"


class ApplicationStatus(str, Enum):
    NEW = "new"
    SCORED = "scored"
    SHORTLISTED = "shortlisted"
    REJECTED = "rejected"
    ACCEPTED = "accepted"
    WAITLISTED = "waitlisted"


@dataclass
class AchievementRecord:
    title: str
    category: str
    year: int
    description: str = ""
    verified: bool = False


@dataclass
class EssaySubmission:
    prompt: str
    text: str
    language: str = "ru"
    word_count: int = 0

    def __post_init__(self):
        if self.word_count == 0:
            self.word_count = len(self.text.split())


@dataclass
class ScoreBreakdown:
    dimension: str
    score: float
    weight: float
    weighted_score: float
    explanation: str
    evidence: list[str] = field(default_factory=list)
    sub_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class NLPMetrics:
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"
    complexity_score: float = 0.0
    vocabulary_richness: float = 0.0
    avg_sentence_length: float = 0.0
    readability_grade: float = 0.0
    ai_detection_score: float = 0.0
    ai_detection_label: str = "likely_human"
    authenticity_score: float = 0.0
    keyword_density: dict[str, float] = field(default_factory=dict)
    emotional_tone: dict[str, float] = field(default_factory=dict)


@dataclass
class CandidateProfile:
    id: str
    full_name: str
    email: str
    age: int
    city: str
    country: str = "Казахстан"
    education_level: str = "bachelor"
    university: str = ""
    gpa: float = 0.0
    work_experience_years: float = 0.0
    skills: list[str] = field(default_factory=list)
    achievements: list[dict] = field(default_factory=list)
    essays: list[dict] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    leadership_roles: list[str] = field(default_factory=list)
    volunteer_hours: int = 0
    projects: list[str] = field(default_factory=list)
    recommendation_count: int = 0
    application_status: str = "new"

    total_score: float = 0.0
    score_breakdown: list[dict] = field(default_factory=list)
    nlp_metrics: list[dict] = field(default_factory=list)
    rank: int = 0
    override_score: Optional[float] = None
    override_comment: str = ""
    reviewer_notes: str = ""


@dataclass
class ScoringConfig:
    motivation_weight: float = 0.25
    leadership_weight: float = 0.20
    growth_weight: float = 0.20
    skills_weight: float = 0.20
    experience_weight: float = 0.15

    essay_nlp_boost: float = 0.10
    ai_penalty_factor: float = 0.15

    min_score: float = 0.0
    max_score: float = 100.0

    shortlist_threshold: float = 70.0
    auto_reject_threshold: float = 30.0

    def get_weights(self) -> dict[str, float]:
        return {
            "motivation": self.motivation_weight,
            "leadership": self.leadership_weight,
            "growth": self.growth_weight,
            "skills": self.skills_weight,
            "experience": self.experience_weight,
        }

    def validate(self) -> bool:
        weights = self.get_weights()
        total = sum(weights.values())
        return abs(total - 1.0) < 0.01
