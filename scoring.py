import math
from dataclasses import asdict

from models import CandidateProfile, ScoringConfig, ScoreBreakdown, NLPMetrics
from nlp_analysis import (
    analyze_essay,
    LEADERSHIP_KEYWORDS,
    GROWTH_KEYWORDS,
    MOTIVATION_KEYWORDS,
    tokenize,
)
from utils import clamp, normalize, weighted_average


class ScoringEngine:

    def __init__(self, config: ScoringConfig | None = None):
        self.config = config or ScoringConfig()

    def score_candidate(self, candidate: CandidateProfile) -> CandidateProfile:
        essay_metrics = []
        for essay in candidate.essays:
            text = essay.get("text", "") if isinstance(essay, dict) else ""
            if text:
                metrics = analyze_essay(text)
                essay_metrics.append(asdict(metrics))

        candidate.nlp_metrics = essay_metrics

        breakdowns = []

        motivation = self._score_motivation(candidate, essay_metrics)
        breakdowns.append(motivation)

        leadership = self._score_leadership(candidate, essay_metrics)
        breakdowns.append(leadership)

        growth = self._score_growth(candidate, essay_metrics)
        breakdowns.append(growth)

        skills = self._score_skills(candidate)
        breakdowns.append(skills)

        experience = self._score_experience(candidate)
        breakdowns.append(experience)

        raw_total = sum(b["weighted_score"] for b in breakdowns)

        nlp_bonus = self._compute_nlp_bonus(essay_metrics)
        ai_penalty = self._compute_ai_penalty(essay_metrics)

        total_score = raw_total + nlp_bonus - ai_penalty
        total_score = clamp(total_score, self.config.min_score, self.config.max_score)

        candidate.score_breakdown = breakdowns
        candidate.total_score = round(total_score, 2)
        candidate.application_status = "scored"

        return candidate

    def _score_motivation(self, candidate: CandidateProfile, essay_metrics: list[dict]) -> dict:
        signals = []
        evidence = []

        essay_motivation = 0.0
        for i, essay in enumerate(candidate.essays):
            text = essay.get("text", "") if isinstance(essay, dict) else ""
            prompt = essay.get("prompt", "") if isinstance(essay, dict) else ""
            if not text:
                continue

            tokens = tokenize(text)
            motivation_count = sum(1 for t in tokens if t in MOTIVATION_KEYWORDS)
            density = motivation_count / max(len(tokens), 1)
            essay_score = min(1.0, density * 25)
            essay_motivation = max(essay_motivation, essay_score)

            if i < len(essay_metrics):
                sentiment = essay_metrics[i].get("sentiment_score", 0)
                if sentiment > 0.2:
                    essay_motivation = min(1.0, essay_motivation + 0.1)
                    evidence.append(f"Позитивный тон в эссе ({sentiment:.2f})")

            if motivation_count > 0:
                evidence.append(f"Ключевые слова мотивации в эссе: {motivation_count} шт.")

        signals.append((essay_motivation * 100, 0.45))

        vol_score = min(1.0, candidate.volunteer_hours / 200) * 100
        signals.append((vol_score, 0.20))
        if candidate.volunteer_hours > 0:
            evidence.append(f"Волонтёрский опыт: {candidate.volunteer_hours} часов")

        rec_score = min(1.0, candidate.recommendation_count / 3) * 100
        signals.append((rec_score, 0.15))
        if candidate.recommendation_count > 0:
            evidence.append(f"Рекомендации: {candidate.recommendation_count}")

        project_score = min(1.0, len(candidate.projects) / 4) * 100
        signals.append((project_score, 0.20))
        if candidate.projects:
            evidence.append(f"Проекты: {len(candidate.projects)}")

        raw_score = weighted_average(signals)
        weight = self.config.motivation_weight
        weighted_score = raw_score * weight

        explanation = self._build_explanation(
            "Мотивация", raw_score, evidence,
            "Оценивается на основе анализа эссе (тональность, ключевые слова), "
            "волонтёрского опыта, рекомендаций и проектной активности."
        )

        return {
            "dimension": "motivation",
            "score": round(raw_score, 2),
            "weight": weight,
            "weighted_score": round(weighted_score, 2),
            "explanation": explanation,
            "evidence": evidence,
            "sub_scores": {
                "essay_analysis": round(essay_motivation * 100, 1),
                "volunteer": round(vol_score, 1),
                "recommendations": round(rec_score, 1),
                "projects": round(project_score, 1),
            },
        }

    def _score_leadership(self, candidate: CandidateProfile, essay_metrics: list[dict]) -> dict:
        signals = []
        evidence = []

        role_score = min(1.0, len(candidate.leadership_roles) / 3) * 100
        signals.append((role_score, 0.35))
        if candidate.leadership_roles:
            evidence.append(f"Лидерские роли: {', '.join(candidate.leadership_roles[:3])}")

        essay_leadership = 0.0
        for essay in candidate.essays:
            text = essay.get("text", "") if isinstance(essay, dict) else ""
            if not text:
                continue
            tokens = tokenize(text)
            lead_count = sum(1 for t in tokens if t in LEADERSHIP_KEYWORDS)
            density = lead_count / max(len(tokens), 1)
            score = min(1.0, density * 30)
            essay_leadership = max(essay_leadership, score)
            if lead_count > 0:
                evidence.append(f"Упоминания лидерства в эссе: {lead_count}")

        signals.append((essay_leadership * 100, 0.30))

        team_achievements = 0
        for ach in candidate.achievements:
            cat = ach.get("category", "") if isinstance(ach, dict) else ""
            if cat in ("leadership", "team", "лидерство", "команда", "организация"):
                team_achievements += 1
        ach_score = min(1.0, team_achievements / 2) * 100
        signals.append((ach_score, 0.20))
        if team_achievements > 0:
            evidence.append(f"Достижения в лидерстве/команде: {team_achievements}")

        vol_leadership = min(1.0, candidate.volunteer_hours / 150) * 50
        signals.append((vol_leadership, 0.15))

        raw_score = weighted_average(signals)
        weight = self.config.leadership_weight
        weighted_score = raw_score * weight

        explanation = self._build_explanation(
            "Лидерский потенциал", raw_score, evidence,
            "Анализ лидерских ролей, упоминаний в эссе, командных достижений "
            "и организаторского опыта."
        )

        return {
            "dimension": "leadership",
            "score": round(raw_score, 2),
            "weight": weight,
            "weighted_score": round(weighted_score, 2),
            "explanation": explanation,
            "evidence": evidence,
            "sub_scores": {
                "roles": round(role_score, 1),
                "essay_analysis": round(essay_leadership * 100, 1),
                "achievements": round(ach_score, 1),
                "volunteer_leadership": round(vol_leadership, 1),
            },
        }

    def _score_growth(self, candidate: CandidateProfile, essay_metrics: list[dict]) -> dict:
        signals = []
        evidence = []

        gpa_score = 0.0
        if candidate.gpa > 0:
            gpa_score = normalize(candidate.gpa, 2.0, 4.0) * 100
            evidence.append(f"GPA: {candidate.gpa:.2f}")
        signals.append((gpa_score, 0.25))

        essay_growth = 0.0
        for essay in candidate.essays:
            text = essay.get("text", "") if isinstance(essay, dict) else ""
            if not text:
                continue
            tokens = tokenize(text)
            growth_count = sum(1 for t in tokens if t in GROWTH_KEYWORDS)
            density = growth_count / max(len(tokens), 1)
            score = min(1.0, density * 25)
            essay_growth = max(essay_growth, score)
            if growth_count > 0:
                evidence.append(f"Индикаторы роста в эссе: {growth_count}")

        signals.append((essay_growth * 100, 0.30))

        edu_scores = {
            "school": 30, "bachelor": 50, "master": 75, "phd": 90, "other": 40,
        }
        edu_score = edu_scores.get(candidate.education_level, 40)
        signals.append((edu_score, 0.20))
        evidence.append(f"Уровень образования: {candidate.education_level}")

        ach_count = len(candidate.achievements)
        ach_diversity = len(set(
            a.get("category", "") for a in candidate.achievements
            if isinstance(a, dict)
        ))
        ach_score = min(1.0, (ach_count / 5 + ach_diversity / 3) / 2) * 100
        signals.append((ach_score, 0.25))
        if ach_count > 0:
            evidence.append(f"Достижения: {ach_count} (в {ach_diversity} категориях)")

        raw_score = weighted_average(signals)
        weight = self.config.growth_weight
        weighted_score = raw_score * weight

        explanation = self._build_explanation(
            "Траектория роста", raw_score, evidence,
            "Оценка GPA, уровня образования, динамики достижений и "
            "индикаторов личностного роста в эссе."
        )

        return {
            "dimension": "growth",
            "score": round(raw_score, 2),
            "weight": weight,
            "weighted_score": round(weighted_score, 2),
            "explanation": explanation,
            "evidence": evidence,
            "sub_scores": {
                "gpa": round(gpa_score, 1),
                "essay_analysis": round(essay_growth * 100, 1),
                "education_level": round(edu_score, 1),
                "achievements": round(ach_score, 1),
            },
        }

    def _score_skills(self, candidate: CandidateProfile) -> dict:
        signals = []
        evidence = []

        HIGH_VALUE_SKILLS = {
            "python", "machine learning", "data science", "ai", "leadership",
            "project management", "analytics", "research", "design thinking",
            "critical thinking", "public speaking", "entrepreneurship",
            "product management", "ux design", "blockchain", "cloud computing",
            "программирование", "аналитика", "управление проектами",
            "машинное обучение", "дизайн-мышление", "предпринимательство",
        }

        skill_count = len(candidate.skills)
        skill_breadth = min(1.0, skill_count / 8) * 100
        signals.append((skill_breadth, 0.30))
        if skill_count > 0:
            evidence.append(f"Количество навыков: {skill_count}")

        high_value = [s for s in candidate.skills if s.lower() in HIGH_VALUE_SKILLS]
        hv_score = min(1.0, len(high_value) / 4) * 100
        signals.append((hv_score, 0.35))
        if high_value:
            evidence.append(f"Ключевые навыки: {', '.join(high_value[:4])}")

        lang_score = min(1.0, len(candidate.languages) / 3) * 100
        signals.append((lang_score, 0.15))
        if candidate.languages:
            evidence.append(f"Языки: {', '.join(candidate.languages)}")

        project_complexity = min(1.0, len(candidate.projects) / 3) * 100
        signals.append((project_complexity, 0.20))
        if candidate.projects:
            evidence.append(f"Проекты: {', '.join(candidate.projects[:3])}")

        raw_score = weighted_average(signals)
        weight = self.config.skills_weight
        weighted_score = raw_score * weight

        explanation = self._build_explanation(
            "Навыки и компетенции", raw_score, evidence,
            "Анализ количества и качества навыков, владения языками, "
            "проектного опыта и наличия востребованных компетенций."
        )

        return {
            "dimension": "skills",
            "score": round(raw_score, 2),
            "weight": weight,
            "weighted_score": round(weighted_score, 2),
            "explanation": explanation,
            "evidence": evidence,
            "sub_scores": {
                "breadth": round(skill_breadth, 1),
                "high_value": round(hv_score, 1),
                "languages": round(lang_score, 1),
                "projects": round(project_complexity, 1),
            },
        }

    def _score_experience(self, candidate: CandidateProfile) -> dict:
        signals = []
        evidence = []

        work_score = min(1.0, candidate.work_experience_years / 5) * 100
        signals.append((work_score, 0.35))
        if candidate.work_experience_years > 0:
            evidence.append(f"Опыт работы: {candidate.work_experience_years} лет")

        ach_score = min(1.0, len(candidate.achievements) / 5) * 100
        signals.append((ach_score, 0.25))
        if candidate.achievements:
            categories = set(
                a.get("category", "") for a in candidate.achievements
                if isinstance(a, dict)
            )
            evidence.append(f"Достижения: {len(candidate.achievements)} в категориях: {', '.join(categories)}")

        vol_score = min(1.0, candidate.volunteer_hours / 100) * 100
        signals.append((vol_score, 0.20))

        project_score = min(1.0, len(candidate.projects) / 4) * 100
        signals.append((project_score, 0.20))

        raw_score = weighted_average(signals)
        weight = self.config.experience_weight
        weighted_score = raw_score * weight

        explanation = self._build_explanation(
            "Опыт", raw_score, evidence,
            "Учитывается профессиональный стаж, достижения, "
            "волонтёрская деятельность и проектный опыт."
        )

        return {
            "dimension": "experience",
            "score": round(raw_score, 2),
            "weight": weight,
            "weighted_score": round(weighted_score, 2),
            "explanation": explanation,
            "evidence": evidence,
            "sub_scores": {
                "work": round(work_score, 1),
                "achievements": round(ach_score, 1),
                "volunteer": round(vol_score, 1),
                "projects": round(project_score, 1),
            },
        }

    def _compute_nlp_bonus(self, essay_metrics: list[dict]) -> float:
        if not essay_metrics:
            return 0.0

        avg_complexity = sum(m.get("complexity_score", 0) for m in essay_metrics) / len(essay_metrics)
        avg_authenticity = sum(m.get("authenticity_score", 0) for m in essay_metrics) / len(essay_metrics)
        avg_richness = sum(m.get("vocabulary_richness", 0) for m in essay_metrics) / len(essay_metrics)

        bonus = (avg_complexity * 0.3 + avg_authenticity * 0.4 + avg_richness * 0.3)
        return bonus * self.config.essay_nlp_boost * 100

    def _compute_ai_penalty(self, essay_metrics: list[dict]) -> float:
        if not essay_metrics:
            return 0.0

        max_ai_score = max(m.get("ai_detection_score", 0) for m in essay_metrics)

        if max_ai_score > 0.65:
            penalty = max_ai_score * self.config.ai_penalty_factor * 100
        elif max_ai_score > 0.40:
            penalty = max_ai_score * self.config.ai_penalty_factor * 50
        else:
            penalty = 0.0

        return penalty

    def _build_explanation(
        self, dimension_name: str, score: float,
        evidence: list[str], methodology: str
    ) -> str:
        if score >= 80:
            level = "Отлично"
        elif score >= 60:
            level = "Хорошо"
        elif score >= 40:
            level = "Удовлетворительно"
        elif score >= 20:
            level = "Ниже среднего"
        else:
            level = "Низкий уровень"

        parts = [f"{dimension_name}: {score:.1f}/100 ({level})"]
        parts.append(f"Методология: {methodology}")

        if evidence:
            parts.append("Основания:")
            for e in evidence:
                parts.append(f"  - {e}")

        return "\n".join(parts)

    def rank_candidates(self, candidates: list[CandidateProfile]) -> list[CandidateProfile]:
        for c in candidates:
            if c.application_status == "new":
                self.score_candidate(c)

        scored = sorted(
            candidates,
            key=lambda c: c.override_score if c.override_score is not None else c.total_score,
            reverse=True,
        )

        for i, c in enumerate(scored, 1):
            c.rank = i

        return scored

    def generate_shortlist(
        self, candidates: list[CandidateProfile],
        threshold: float | None = None,
        max_count: int | None = None,
    ) -> list[CandidateProfile]:
        threshold = threshold or self.config.shortlist_threshold

        ranked = self.rank_candidates(candidates)
        shortlisted = []

        for c in ranked:
            effective_score = c.override_score if c.override_score is not None else c.total_score
            if effective_score >= threshold:
                c.application_status = "shortlisted"
                shortlisted.append(c)
            elif effective_score < self.config.auto_reject_threshold:
                c.application_status = "rejected"

        if max_count and len(shortlisted) > max_count:
            shortlisted = shortlisted[:max_count]
            for c in ranked:
                if c not in shortlisted and c.application_status == "shortlisted":
                    c.application_status = "waitlisted"

        return shortlisted

    def get_dimension_stats(self, candidates: list[CandidateProfile]) -> dict:
        dimensions = ["motivation", "leadership", "growth", "skills", "experience"]
        stats = {}

        for dim in dimensions:
            scores = []
            for c in candidates:
                for b in c.score_breakdown:
                    if b["dimension"] == dim:
                        scores.append(b["score"])
                        break

            if scores:
                stats[dim] = {
                    "mean": round(sum(scores) / len(scores), 2),
                    "min": round(min(scores), 2),
                    "max": round(max(scores), 2),
                    "median": round(sorted(scores)[len(scores) // 2], 2),
                    "std": round(
                        math.sqrt(sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)),
                        2
                    ),
                    "count": len(scores),
                }
            else:
                stats[dim] = {"mean": 0, "min": 0, "max": 0, "median": 0, "std": 0, "count": 0}

        return stats
