import json
import csv
import io
import hashlib
from datetime import datetime
from typing import Any

import pandas as pd

from models import CandidateProfile


def generate_candidate_id(name: str, email: str) -> str:
    raw = f"{name}:{email}:{datetime.now().isoformat()}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def load_candidates_from_json(file_content: str) -> list[dict]:
    try:
        data = json.loads(file_content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "candidates" in data:
            return data["candidates"]
        raise ValueError("JSON должен содержать массив кандидатов или объект с полем 'candidates'")
    except json.JSONDecodeError as e:
        raise ValueError(f"Ошибка парсинга JSON: {str(e)}")


def load_candidates_from_csv(file_content: str) -> list[dict]:
    try:
        reader = csv.DictReader(io.StringIO(file_content))
        candidates = []
        for row in reader:
            candidate = {}
            for key, value in row.items():
                if key is None:
                    continue
                key = key.strip()
                if value and value.startswith("["):
                    try:
                        candidate[key] = json.loads(value)
                    except json.JSONDecodeError:
                        candidate[key] = [v.strip() for v in value.strip("[]").split(",")]
                elif value and value.replace(".", "").replace("-", "").isdigit():
                    try:
                        candidate[key] = float(value) if "." in value else int(value)
                    except ValueError:
                        candidate[key] = value
                else:
                    candidate[key] = value if value else ""
            candidates.append(candidate)
        return candidates
    except Exception as e:
        raise ValueError(f"Ошибка парсинга CSV: {str(e)}")


def dict_to_candidate(data: dict) -> CandidateProfile:
    field_mapping = {
        "id": "id",
        "full_name": "full_name",
        "name": "full_name",
        "email": "email",
        "age": "age",
        "city": "city",
        "country": "country",
        "education_level": "education_level",
        "university": "university",
        "gpa": "gpa",
        "work_experience_years": "work_experience_years",
        "experience_years": "work_experience_years",
        "skills": "skills",
        "achievements": "achievements",
        "essays": "essays",
        "languages": "languages",
        "leadership_roles": "leadership_roles",
        "volunteer_hours": "volunteer_hours",
        "projects": "projects",
        "recommendation_count": "recommendation_count",
    }

    mapped = {}
    for src_key, dst_key in field_mapping.items():
        if src_key in data:
            mapped[dst_key] = data[src_key]

    if "id" not in mapped:
        mapped["id"] = generate_candidate_id(
            mapped.get("full_name", "unknown"),
            mapped.get("email", "unknown"),
        )

    if "age" in mapped:
        mapped["age"] = int(mapped["age"])
    if "gpa" in mapped:
        mapped["gpa"] = float(mapped["gpa"])
    if "work_experience_years" in mapped:
        mapped["work_experience_years"] = float(mapped["work_experience_years"])
    if "volunteer_hours" in mapped:
        mapped["volunteer_hours"] = int(mapped["volunteer_hours"])
    if "recommendation_count" in mapped:
        mapped["recommendation_count"] = int(mapped["recommendation_count"])

    if isinstance(mapped.get("skills"), str):
        mapped["skills"] = [s.strip() for s in mapped["skills"].split(",")]
    if isinstance(mapped.get("languages"), str):
        mapped["languages"] = [s.strip() for s in mapped["languages"].split(",")]
    if isinstance(mapped.get("leadership_roles"), str):
        mapped["leadership_roles"] = [s.strip() for s in mapped["leadership_roles"].split(",")]
    if isinstance(mapped.get("projects"), str):
        mapped["projects"] = [s.strip() for s in mapped["projects"].split(",")]

    return CandidateProfile(**mapped)


def candidates_to_dataframe(candidates: list[CandidateProfile]) -> pd.DataFrame:
    records = []
    for c in candidates:
        records.append({
            "ID": c.id,
            "Имя": c.full_name,
            "Возраст": c.age,
            "Город": c.city,
            "Образование": c.education_level,
            "Университет": c.university,
            "GPA": c.gpa,
            "Опыт (лет)": c.work_experience_years,
            "Навыки": ", ".join(c.skills[:5]) if c.skills else "",
            "Языки": ", ".join(c.languages) if c.languages else "",
            "Волонтёрство (ч.)": c.volunteer_hours,
            "Рекомендации": c.recommendation_count,
            "Общий балл": round(c.total_score, 1),
            "Статус": translate_status(c.application_status),
            "Ранг": c.rank,
        })
    return pd.DataFrame(records)


def translate_status(status: str) -> str:
    status_map = {
        "new": "Новый",
        "scored": "Оценён",
        "shortlisted": "В шорт-листе",
        "rejected": "Отклонён",
        "accepted": "Принят",
        "waitlisted": "Лист ожидания",
    }
    return status_map.get(status, status)


def format_score(score: float, precision: int = 1) -> str:
    return f"{score:.{precision}f}"


def score_to_grade(score: float) -> str:
    if score >= 90:
        return "A+"
    elif score >= 85:
        return "A"
    elif score >= 80:
        return "A-"
    elif score >= 75:
        return "B+"
    elif score >= 70:
        return "B"
    elif score >= 65:
        return "B-"
    elif score >= 60:
        return "C+"
    elif score >= 55:
        return "C"
    elif score >= 50:
        return "C-"
    elif score >= 40:
        return "D"
    else:
        return "F"


def score_to_color(score: float) -> str:
    if score >= 80:
        return "#22c55e"
    elif score >= 60:
        return "#eab308"
    elif score >= 40:
        return "#f97316"
    else:
        return "#ef4444"


def export_results_to_json(candidates: list[CandidateProfile]) -> str:
    results = []
    for c in candidates:
        results.append({
            "id": c.id,
            "full_name": c.full_name,
            "total_score": round(c.total_score, 2),
            "rank": c.rank,
            "status": c.application_status,
            "score_breakdown": c.score_breakdown,
            "override_score": c.override_score,
            "override_comment": c.override_comment,
        })
    return json.dumps(results, ensure_ascii=False, indent=2)


def export_results_to_csv(candidates: list[CandidateProfile]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID", "Имя", "Общий балл", "Ранг", "Статус",
        "Мотивация", "Лидерство", "Рост", "Навыки", "Опыт",
        "Корректировка", "Комментарий",
    ])
    for c in candidates:
        breakdown = {s["dimension"]: s["weighted_score"] for s in c.score_breakdown} if c.score_breakdown else {}
        writer.writerow([
            c.id, c.full_name, round(c.total_score, 2), c.rank, c.application_status,
            round(breakdown.get("motivation", 0), 2),
            round(breakdown.get("leadership", 0), 2),
            round(breakdown.get("growth", 0), 2),
            round(breakdown.get("skills", 0), 2),
            round(breakdown.get("experience", 0), 2),
            c.override_score or "",
            c.override_comment,
        ])
    return output.getvalue()


def clamp(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    return max(min_val, min(max_val, value))


def normalize(value: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


def weighted_average(values: list[tuple[float, float]]) -> float:
    total_weight = sum(w for _, w in values)
    if total_weight == 0:
        return 0.0
    return sum(v * w for v, w in values) / total_weight
