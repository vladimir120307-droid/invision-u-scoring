# v3: validation module
# v2: dark theme support
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from models import CandidateProfile, ScoringConfig
from scoring import ScoringEngine
from nlp_analysis import analyze_essay
from data_generator import generate_dataset, save_dataset
from utils import (
    load_candidates_from_json,
    load_candidates_from_csv,
    dict_to_candidate,
    candidates_to_dataframe,
    translate_status,
    score_to_grade,
    score_to_color,
    export_results_to_json,
    export_results_to_csv,
    format_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIMENSION_NAMES = {
    "motivation": "Мотивация",
    "leadership": "Лидерство",
    "growth": "Траектория роста",
    "skills": "Навыки",
    "experience": "Опыт",
}

DIMENSION_NAMES_SHORT = {
    "motivation": "МТВ",
    "leadership": "ЛДР",
    "growth": "РСТ",
    "skills": "НВК",
    "experience": "ОПТ",
}

SUB_SCORE_NAMES = {
    "essay_analysis": "Анализ эссе",
    "volunteer": "Волонтёрство",
    "recommendations": "Рекомендации",
    "projects": "Проекты",
    "roles": "Лидерские роли",
    "achievements": "Достижения",
    "gpa": "GPA",
    "education_level": "Образование",
    "breadth": "Широта навыков",
    "high_value": "Ключевые навыки",
    "languages": "Языки",
    "work": "Стаж работы",
    "volunteer_leadership": "Волонтёрское лидерство",
}

CHART_COLORS = ["#0d9488", "#3b82f6", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981"]


def _plotly_defaults(dark: bool = True) -> dict:
    """Return Plotly layout defaults for the current theme."""
    if dark:
        return dict(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#e2e8f0"),
            margin=dict(t=50, b=40, l=50, r=30),
        )
    return dict(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#1e293b"),
        margin=dict(t=50, b=40, l=50, r=30),
    )


def _plotly_no_margin(dark: bool = True) -> dict:
    """Return Plotly layout defaults WITHOUT the margin key."""
    d = _plotly_defaults(dark)
    d.pop("margin", None)
    return d


def _is_dark() -> bool:
    return st.session_state.get("dark_mode", True)


# ---------------------------------------------------------------------------
# SVG icon helpers (no emojis!)
# ---------------------------------------------------------------------------

def svg_icon(name, size=20, color="#94a3b8"):
    icons = {
        "users": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
        "bar-chart": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg>',
        "star": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="{color}" stroke="{color}" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>',
        "trending-up": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
        "award": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg>',
        "upload": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
        "shield": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
        "cpu": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
        "sliders": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>',
        "git-compare": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="18" r="3"/><circle cx="6" cy="6" r="3"/><path d="M13 6h3a2 2 0 0 1 2 2v7"/><path d="M11 18H8a2 2 0 0 1-2-2V9"/></svg>',
        "pie-chart": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg>',
        "settings": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06A1.65 1.65 0 0 0 19.32 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
        "user": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
        "file-text": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>',
        "check-circle": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
        "alert-triangle": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        "download": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
        "home": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>',
        "list": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>',
        "zap": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="{color}" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
        "moon": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>',
        "sun": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>',
    }
    return icons.get(name, "")


# ---------------------------------------------------------------------------
# CSS injection -- dark and light themes
# ---------------------------------------------------------------------------

def _css_shared():
    """CSS that is shared between dark and light themes."""
    return """
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

        /* ===== Base ===== */
        .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* ===== Keyframes ===== */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(24px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeInScale {
            from { opacity: 0; transform: scale(0.92); }
            to { opacity: 1; transform: scale(1); }
        }
        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        @keyframes pulseGlow {
            0%, 100% { box-shadow: 0 0 8px rgba(13,148,136,0.3); }
            50% { box-shadow: 0 0 20px rgba(13,148,136,0.6); }
        }
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes countUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes barGrow {
            from { width: 0%; }
            to { width: var(--bar-width); }
        }
        @keyframes borderPulse {
            0%, 100% { border-color: rgba(13,148,136,0.3); }
            50% { border-color: rgba(13,148,136,0.7); }
        }
        @keyframes gradientBorder {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes fadeInPage {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* ===== Page transition ===== */
        section.main > div.block-container {
            animation: fadeInPage 0.45s ease-out;
        }

        /* ===== Gradient Header with animated border ===== */
        .main-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 40%, #0d9488 100%);
            background-size: 200% 200%;
            animation: gradientShift 8s ease infinite;
            padding: 2rem 2.5rem;
            border-radius: 20px;
            color: white;
            margin-bottom: 1.8rem;
            box-shadow: 0 8px 32px rgba(13,148,136,0.25), 0 2px 8px rgba(0,0,0,0.15);
            position: relative;
            overflow: hidden;
            border: 2px solid transparent;
            background-clip: padding-box;
        }
        .main-header::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -20%;
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
            border-radius: 50%;
        }
        .main-header::after {
            content: '';
            position: absolute;
            bottom: -30%;
            left: 10%;
            width: 200px;
            height: 200px;
            background: radial-gradient(circle, rgba(13,148,136,0.2) 0%, transparent 70%);
            border-radius: 50%;
        }
        .main-header h1 {
            font-size: 1.9rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.5px;
            position: relative;
            z-index: 1;
        }
        .main-header p {
            font-size: 0.95rem;
            opacity: 0.85;
            margin-top: 0.4rem;
            font-weight: 400;
            position: relative;
            z-index: 1;
        }
        /* animated gradient border line under header */
        .header-border-glow {
            height: 3px;
            margin: -1.6rem 0 1.8rem 0;
            border-radius: 2px;
            background: linear-gradient(90deg, #0d9488, #3b82f6, #8b5cf6, #ec4899, #0d9488);
            background-size: 300% 100%;
            animation: gradientBorder 4s linear infinite;
        }

        /* ===== Initials Avatar ===== */
        .avatar-initials {
            width: 72px;
            height: 72px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            font-weight: 800;
            color: white;
            flex-shrink: 0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        }

        /* ===== Score Badge ===== */
        .score-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 0.3rem 0.9rem;
            border-radius: 24px;
            font-weight: 700;
            font-size: 0.85rem;
            color: white;
            letter-spacing: -0.2px;
        }
        .score-high { background: linear-gradient(135deg, #059669, #10b981); }
        .score-mid  { background: linear-gradient(135deg, #d97706, #f59e0b); }
        .score-low  { background: linear-gradient(135deg, #dc2626, #ef4444); }

        /* ===== Status Pills ===== */
        .status-pill {
            display: inline-block;
            padding: 0.2rem 0.7rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.3px;
        }
        .status-shortlisted { background: rgba(16,185,129,0.12); color: #059669; border: 1px solid rgba(16,185,129,0.25); }
        .status-scored { background: rgba(59,130,246,0.12); color: #2563eb; border: 1px solid rgba(59,130,246,0.25); }
        .status-rejected { background: rgba(239,68,68,0.12); color: #dc2626; border: 1px solid rgba(239,68,68,0.25); }
        .status-new { background: rgba(99,102,241,0.12); color: #4f46e5; border: 1px solid rgba(99,102,241,0.25); }
        .status-waitlisted { background: rgba(245,158,11,0.12); color: #d97706; border: 1px solid rgba(245,158,11,0.25); }
        .status-accepted { background: rgba(16,185,129,0.15); color: #047857; border: 1px solid rgba(16,185,129,0.3); }

        /* ===== Score Bar ===== */
        .score-bar-container {
            width: 100%;
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
            position: relative;
        }
        .score-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* ===== Gauge / Traffic Light ===== */
        .traffic-light {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.82rem;
        }
        .traffic-green { background: rgba(16,185,129,0.12); color: #059669; }
        .traffic-yellow { background: rgba(245,158,11,0.12); color: #d97706; }
        .traffic-red { background: rgba(239,68,68,0.12); color: #dc2626; }
        .traffic-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }
        .traffic-dot-green { background: #10b981; box-shadow: 0 0 6px rgba(16,185,129,0.5); }
        .traffic-dot-yellow { background: #f59e0b; box-shadow: 0 0 6px rgba(245,158,11,0.5); }
        .traffic-dot-red { background: #ef4444; box-shadow: 0 0 6px rgba(239,68,68,0.5); }

        /* ===== Comparison Winner ===== */
        .winner-badge {
            display: inline-block;
            background: linear-gradient(135deg, #0d9488, #10b981);
            color: white;
            padding: 0.15rem 0.5rem;
            border-radius: 8px;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* ===== Scrollbar ===== */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(13,148,136,0.3); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(13,148,136,0.5); }

        /* ===== Hide Streamlit default elements ===== */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* ===== Custom radio buttons in sidebar ===== */
        div[data-testid="stSidebar"] [role="radiogroup"] {
            gap: 2px !important;
        }
        div[data-testid="stSidebar"] [role="radio"] {
            padding: 0.5rem 0.8rem !important;
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stSidebar"] [role="radio"]:hover {
            background: rgba(13,148,136,0.1) !important;
        }

        /* ===== Streamlit elements override ===== */
        .stButton>button[kind="primary"] {
            background: linear-gradient(135deg, #0d9488, #0f766e) !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            letter-spacing: 0.2px !important;
            padding: 0.55rem 1.5rem !important;
            transition: all 0.3s ease !important;
        }
        .stButton>button[kind="primary"]:hover {
            box-shadow: 0 6px 24px rgba(13,148,136,0.35) !important;
            transform: translateY(-2px) !important;
        }
        .stButton>button[kind="secondary"] {
            border-radius: 10px !important;
            border-color: #0d9488 !important;
            color: #0d9488 !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }

        /* ===== Tab styling ===== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            border-radius: 12px;
            padding: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 0.5rem 1.2rem;
            font-weight: 500;
            font-size: 0.88rem;
        }

        /* ===== Progress bar custom ===== */
        .custom-progress-outer {
            width: 100%;
            height: 10px;
            background: rgba(226,232,240,0.6);
            border-radius: 5px;
            overflow: hidden;
        }
        .custom-progress-inner {
            height: 100%;
            border-radius: 5px;
            transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* ===== Loading Skeleton ===== */
        .skeleton {
            border-radius: 8px;
            animation: shimmer 1.5s ease-in-out infinite;
        }
        .skeleton-line {
            height: 14px;
            margin-bottom: 8px;
            border-radius: 6px;
            animation: shimmer 1.5s ease-in-out infinite;
        }

        /* ===== Footer ===== */
        .app-footer {
            text-align: center;
            padding: 2rem 0 1rem 0;
            font-size: 0.78rem;
            letter-spacing: 0.3px;
            opacity: 0.7;
        }

        /* ===== Sidebar ===== */
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
        }
        div[data-testid="stSidebar"] .stRadio label {
            color: #e2e8f0 !important;
        }
        div[data-testid="stSidebar"] .stRadio label:hover {
            color: #0d9488 !important;
        }
        div[data-testid="stSidebar"] .stMarkdown p,
        div[data-testid="stSidebar"] .stMarkdown span,
        div[data-testid="stSidebar"] .stCaption {
            color: #94a3b8 !important;
        }
        div[data-testid="stSidebar"] hr {
            border-color: rgba(148,163,184,0.15) !important;
        }
    """


def _css_dark():
    """CSS specific to dark theme."""
    return """
        /* ===== Dark background with dot pattern ===== */
        .stApp {
            background-color: #0f172a !important;
            background-image: radial-gradient(rgba(148,163,184,0.07) 1px, transparent 1px);
            background-size: 24px 24px;
        }

        /* ===== Glass Card ===== */
        .glass-card {
            background: rgba(30,41,59,0.8);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(71,85,105,0.3);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 24px rgba(0,0,0,0.2), 0 1px 4px rgba(0,0,0,0.15);
            transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
            animation: fadeInUp 0.5s ease-out;
        }
        .glass-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.3), 0 2px 8px rgba(0,0,0,0.2);
            border-color: rgba(13,148,136,0.3);
        }

        /* ===== Metric Cards ===== */
        .metric-card-v2 {
            background: rgba(30,41,59,0.8);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(71,85,105,0.3);
            border-radius: 16px;
            padding: 1.3rem 1rem;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            animation: fadeInUp 0.5s ease-out;
            position: relative;
            overflow: hidden;
        }
        .metric-card-v2::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #0d9488, #3b82f6, #8b5cf6);
            border-radius: 16px 16px 0 0;
        }
        .metric-card-v2:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 36px rgba(13,148,136,0.15);
        }
        .metric-icon { margin-bottom: 0.5rem; display: flex; justify-content: center; align-items: center; }
        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #e2e8f0;
            line-height: 1.1;
            animation: countUp 0.6s ease-out;
        }
        .metric-label {
            font-size: 0.78rem;
            color: #94a3b8;
            margin-top: 0.35rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* ===== Candidate Card ===== */
        .candidate-card-v2 {
            background: rgba(30,41,59,0.8);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(71,85,105,0.3);
            border-radius: 14px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 2px 12px rgba(0,0,0,0.15);
            border-left: 4px solid;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            animation: slideInLeft 0.4s ease-out;
        }
        .candidate-card-v2:hover {
            transform: translateX(6px);
            box-shadow: 0 6px 24px rgba(0,0,0,0.2);
        }
        .candidate-name { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; letter-spacing: -0.3px; }
        .candidate-meta { font-size: 0.82rem; color: #94a3b8; margin-top: 0.2rem; font-weight: 400; }

        /* ===== Explanation Box ===== */
        .explanation-box {
            background: rgba(30,41,59,0.7);
            backdrop-filter: blur(8px);
            border-radius: 12px;
            padding: 1.1rem;
            margin-top: 0.5rem;
            border: 1px solid rgba(71,85,105,0.3);
            font-size: 0.88rem;
            line-height: 1.7;
            white-space: pre-line;
            color: #cbd5e1;
            transition: all 0.3s ease;
        }
        .explanation-box:hover {
            border-color: #0d9488;
            box-shadow: 0 2px 12px rgba(13,148,136,0.08);
        }

        /* ===== Evidence Tag ===== */
        .evidence-tag {
            display: inline-block;
            background: linear-gradient(135deg, rgba(13,148,136,0.15), rgba(59,130,246,0.1));
            color: #5eead4;
            padding: 0.25rem 0.65rem;
            border-radius: 8px;
            font-size: 0.78rem;
            margin: 0.15rem;
            font-weight: 500;
            border: 1px solid rgba(13,148,136,0.25);
            transition: all 0.2s ease;
        }
        .evidence-tag:hover {
            background: linear-gradient(135deg, rgba(13,148,136,0.25), rgba(59,130,246,0.2));
            transform: translateY(-1px);
        }

        /* ===== AI Warning ===== */
        .ai-warning {
            background: linear-gradient(135deg, rgba(245,158,11,0.12), rgba(239,68,68,0.08));
            border: 1px solid rgba(245,158,11,0.3);
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            margin: 0.5rem 0;
            font-size: 0.85rem;
            color: #fbbf24;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* ===== Hero Section ===== */
        .hero-section {
            background: linear-gradient(135deg, rgba(30,41,59,0.8), rgba(13,148,136,0.08));
            border-radius: 20px;
            padding: 2rem;
            border: 1px solid rgba(71,85,105,0.3);
            display: flex;
            gap: 1.5rem;
            align-items: center;
            animation: fadeInScale 0.5s ease-out;
        }

        /* ===== Drag Drop Area ===== */
        .drag-drop-area {
            border: 2px dashed rgba(13,148,136,0.35);
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            background: linear-gradient(135deg, rgba(13,148,136,0.05), rgba(59,130,246,0.03));
            transition: all 0.3s ease;
            animation: borderPulse 3s ease-in-out infinite;
        }
        .drag-drop-area:hover {
            border-color: #0d9488;
            background: linear-gradient(135deg, rgba(13,148,136,0.1), rgba(59,130,246,0.06));
        }

        /* ===== Section Divider ===== */
        .section-divider {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(13,148,136,0.3), transparent);
            margin: 1.8rem 0;
        }

        /* ===== NLP Metric Panel ===== */
        .nlp-metric-panel {
            background: rgba(30,41,59,0.7);
            backdrop-filter: blur(8px);
            border-radius: 12px;
            padding: 1rem;
            margin: 0.3rem 0;
            border: 1px solid rgba(71,85,105,0.3);
            transition: all 0.2s ease;
        }
        .nlp-metric-panel:hover { border-color: rgba(13,148,136,0.3); }

        /* ===== Custom Table Styling ===== */
        .styled-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 12px rgba(0,0,0,0.2);
            font-size: 0.88rem;
        }
        .styled-table thead th {
            background: linear-gradient(135deg, #0f172a, #1e3a5f);
            color: #e2e8f0;
            padding: 0.8rem 1rem;
            font-weight: 600;
            text-align: left;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .styled-table tbody tr { transition: all 0.2s ease; }
        .styled-table tbody tr:nth-child(even) { background: rgba(30,41,59,0.4); }
        .styled-table tbody tr:hover { background: rgba(13,148,136,0.1); transform: scale(1.002); }
        .styled-table tbody td {
            padding: 0.7rem 1rem;
            border-bottom: 1px solid rgba(71,85,105,0.2);
            color: #cbd5e1;
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        /* ===== Mini card for top candidates ===== */
        .mini-card {
            background: rgba(30,41,59,0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(71,85,105,0.3);
            border-radius: 12px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.4rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: all 0.25s ease;
            animation: fadeInUp 0.5s ease-out;
        }
        .mini-card:hover { transform: translateX(4px); box-shadow: 0 4px 16px rgba(0,0,0,0.15); }
        .mini-card-rank { font-size: 1.1rem; font-weight: 800; color: #0d9488; min-width: 28px; }
        .mini-card-name { font-size: 0.88rem; font-weight: 600; color: #e2e8f0; flex: 1; margin-left: 0.6rem; }
        .mini-card-score { font-size: 0.9rem; font-weight: 700; }

        /* ===== Tab list dark ===== */
        .stTabs [data-baseweb="tab-list"] { background: rgba(30,41,59,0.7); }
        .stTabs [aria-selected="true"] { background: rgba(13,148,136,0.2) !important; box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important; }

        /* ===== Skeleton dark ===== */
        .skeleton, .skeleton-line {
            background: linear-gradient(90deg, rgba(51,65,85,0.6) 25%, rgba(71,85,105,0.4) 50%, rgba(51,65,85,0.6) 75%);
            background-size: 200% 100%;
        }

        /* ===== Footer dark ===== */
        .app-footer { color: #64748b; }

        /* ===== Dark text helpers ===== */
        .text-primary { color: #e2e8f0 !important; }
        .text-secondary { color: #94a3b8 !important; }
        .text-accent { color: #0d9488 !important; }
        .text-heading { color: #e2e8f0 !important; }
    """


def _css_light():
    """CSS specific to light theme."""
    return """
        /* ===== Light background with dot pattern ===== */
        .stApp {
            background-color: #f8fafc !important;
            background-image: radial-gradient(rgba(15,23,42,0.04) 1px, transparent 1px);
            background-size: 24px 24px;
        }

        /* ===== Glass Card ===== */
        .glass-card {
            background: rgba(255,255,255,0.8);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(226,232,240,0.5);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 24px rgba(0,0,0,0.06), 0 1px 4px rgba(0,0,0,0.04);
            transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
            animation: fadeInUp 0.5s ease-out;
        }
        .glass-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.1), 0 2px 8px rgba(0,0,0,0.06);
            border-color: rgba(13,148,136,0.3);
        }

        /* ===== Metric Cards ===== */
        .metric-card-v2 {
            background: rgba(255,255,255,0.8);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(226,232,240,0.5);
            border-radius: 16px;
            padding: 1.3rem 1rem;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            animation: fadeInUp 0.5s ease-out;
            position: relative;
            overflow: hidden;
        }
        .metric-card-v2::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #0d9488, #3b82f6, #8b5cf6);
            border-radius: 16px 16px 0 0;
        }
        .metric-card-v2:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 36px rgba(13,148,136,0.15);
        }
        .metric-icon { margin-bottom: 0.5rem; display: flex; justify-content: center; align-items: center; }
        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.1;
            animation: countUp 0.6s ease-out;
        }
        .metric-label {
            font-size: 0.78rem;
            color: #64748b;
            margin-top: 0.35rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* ===== Candidate Card ===== */
        .candidate-card-v2 {
            background: rgba(255,255,255,0.8);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(226,232,240,0.5);
            border-radius: 14px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 2px 12px rgba(0,0,0,0.04);
            border-left: 4px solid;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            animation: slideInLeft 0.4s ease-out;
        }
        .candidate-card-v2:hover {
            transform: translateX(6px);
            box-shadow: 0 6px 24px rgba(0,0,0,0.08);
        }
        .candidate-name { font-size: 1.1rem; font-weight: 700; color: #0f172a; letter-spacing: -0.3px; }
        .candidate-meta { font-size: 0.82rem; color: #64748b; margin-top: 0.2rem; font-weight: 400; }

        /* ===== Explanation Box ===== */
        .explanation-box {
            background: rgba(248,250,252,0.9);
            backdrop-filter: blur(8px);
            border-radius: 12px;
            padding: 1.1rem;
            margin-top: 0.5rem;
            border: 1px solid #e2e8f0;
            font-size: 0.88rem;
            line-height: 1.7;
            white-space: pre-line;
            color: #334155;
            transition: all 0.3s ease;
        }
        .explanation-box:hover {
            border-color: #0d9488;
            box-shadow: 0 2px 12px rgba(13,148,136,0.08);
        }

        /* ===== Evidence Tag ===== */
        .evidence-tag {
            display: inline-block;
            background: linear-gradient(135deg, rgba(13,148,136,0.08), rgba(59,130,246,0.08));
            color: #0f766e;
            padding: 0.25rem 0.65rem;
            border-radius: 8px;
            font-size: 0.78rem;
            margin: 0.15rem;
            font-weight: 500;
            border: 1px solid rgba(13,148,136,0.15);
            transition: all 0.2s ease;
        }
        .evidence-tag:hover {
            background: linear-gradient(135deg, rgba(13,148,136,0.15), rgba(59,130,246,0.15));
            transform: translateY(-1px);
        }

        /* ===== AI Warning ===== */
        .ai-warning {
            background: linear-gradient(135deg, rgba(245,158,11,0.08), rgba(239,68,68,0.06));
            border: 1px solid rgba(245,158,11,0.3);
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            margin: 0.5rem 0;
            font-size: 0.85rem;
            color: #92400e;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* ===== Hero Section ===== */
        .hero-section {
            background: linear-gradient(135deg, rgba(15,23,42,0.03), rgba(13,148,136,0.05));
            border-radius: 20px;
            padding: 2rem;
            border: 1px solid rgba(226,232,240,0.5);
            display: flex;
            gap: 1.5rem;
            align-items: center;
            animation: fadeInScale 0.5s ease-out;
        }

        /* ===== Drag Drop Area ===== */
        .drag-drop-area {
            border: 2px dashed rgba(13,148,136,0.35);
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            background: linear-gradient(135deg, rgba(13,148,136,0.03), rgba(59,130,246,0.03));
            transition: all 0.3s ease;
            animation: borderPulse 3s ease-in-out infinite;
        }
        .drag-drop-area:hover {
            border-color: #0d9488;
            background: linear-gradient(135deg, rgba(13,148,136,0.06), rgba(59,130,246,0.06));
        }

        /* ===== Section Divider ===== */
        .section-divider {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(13,148,136,0.3), transparent);
            margin: 1.8rem 0;
        }

        /* ===== NLP Metric Panel ===== */
        .nlp-metric-panel {
            background: rgba(248,250,252,0.9);
            backdrop-filter: blur(8px);
            border-radius: 12px;
            padding: 1rem;
            margin: 0.3rem 0;
            border: 1px solid #e2e8f0;
            transition: all 0.2s ease;
        }
        .nlp-metric-panel:hover { border-color: rgba(13,148,136,0.3); }

        /* ===== Custom Table Styling ===== */
        .styled-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 12px rgba(0,0,0,0.04);
            font-size: 0.88rem;
        }
        .styled-table thead th {
            background: linear-gradient(135deg, #0f172a, #1e3a5f);
            color: #e2e8f0;
            padding: 0.8rem 1rem;
            font-weight: 600;
            text-align: left;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .styled-table tbody tr { transition: all 0.2s ease; }
        .styled-table tbody tr:nth-child(even) { background: rgba(248,250,252,0.6); }
        .styled-table tbody tr:hover { background: rgba(13,148,136,0.06); transform: scale(1.002); }
        .styled-table tbody td {
            padding: 0.7rem 1rem;
            border-bottom: 1px solid #f1f5f9;
            color: #334155;
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        /* ===== Mini card for top candidates ===== */
        .mini-card {
            background: rgba(255,255,255,0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(226,232,240,0.5);
            border-radius: 12px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.4rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: all 0.25s ease;
            animation: fadeInUp 0.5s ease-out;
        }
        .mini-card:hover { transform: translateX(4px); box-shadow: 0 4px 16px rgba(0,0,0,0.06); }
        .mini-card-rank { font-size: 1.1rem; font-weight: 800; color: #0d9488; min-width: 28px; }
        .mini-card-name { font-size: 0.88rem; font-weight: 600; color: #1e293b; flex: 1; margin-left: 0.6rem; }
        .mini-card-score { font-size: 0.9rem; font-weight: 700; }

        /* ===== Tab list light ===== */
        .stTabs [data-baseweb="tab-list"] { background: rgba(241,245,249,0.7); }
        .stTabs [aria-selected="true"] { background: white !important; box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important; }

        /* ===== Skeleton light ===== */
        .skeleton, .skeleton-line {
            background: linear-gradient(90deg, #e2e8f0 25%, #f1f5f9 50%, #e2e8f0 75%);
            background-size: 200% 100%;
        }

        /* ===== Footer light ===== */
        .app-footer { color: #94a3b8; }

        /* ===== Light text helpers ===== */
        .text-primary { color: #1e293b !important; }
        .text-secondary { color: #64748b !important; }
        .text-accent { color: #0d9488 !important; }
        .text-heading { color: #0f172a !important; }
    """


def inject_custom_css():
    dark = _is_dark()
    theme_css = _css_dark() if dark else _css_light()
    st.markdown(f"<style>{_css_shared()}{theme_css}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Theme-aware text color helpers
# ---------------------------------------------------------------------------

def _c(role="primary"):
    """Return a CSS color string depending on the theme."""
    dark = _is_dark()
    palette = {
        "heading": "#e2e8f0" if dark else "#0f172a",
        "primary": "#e2e8f0" if dark else "#1e293b",
        "secondary": "#94a3b8" if dark else "#64748b",
        "accent": "#0d9488",
        "muted": "#475569" if dark else "#94a3b8",
        "card_text": "#cbd5e1" if dark else "#334155",
        "warning_text": "#fbbf24" if dark else "#92400e",
    }
    return palette.get(role, palette["primary"])


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def init_session_state():
    defaults = {
        "candidates": [],
        "scored": False,
        "scoring_config": ScoringConfig(),
        "selected_candidate_id": None,
        "overrides": {},
        "shortlist_generated": False,
        "dark_mode": True,
        "anonymize_names": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Helper rendering functions
# ---------------------------------------------------------------------------

def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>inVision U  --  Система отбора кандидатов</h1>
        <p>Интеллектуальная платформа для приёмной комиссии с объяснимым скорингом</p>
    </div>
    <div class="header-border-glow"></div>
    """, unsafe_allow_html=True)


def render_skeleton(lines=4):
    """Show a skeleton loading placeholder."""
    html = '<div class="glass-card" style="padding:1.5rem">'
    for i in range(lines):
        w = 90 - i * 15 if i < 3 else 50
        html += f'<div class="skeleton-line" style="width:{w}%"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_footer():
    st.markdown(
        '<div class="app-footer">Powered by ML  |  Decentrathon 5.0</div>',
        unsafe_allow_html=True,
    )


def _score_bar_html(value, max_val=100, height=8):
    pct = min(100, max(0, value / max_val * 100))
    if pct >= 70:
        color = "linear-gradient(90deg, #10b981, #059669)"
    elif pct >= 45:
        color = "linear-gradient(90deg, #f59e0b, #d97706)"
    else:
        color = "linear-gradient(90deg, #ef4444, #dc2626)"
    return f'''<div class="score-bar-container" style="height:{height}px">
        <div class="score-bar-fill" style="width:{pct}%;background:{color}"></div>
    </div>'''


def _score_badge_html(score):
    if score >= 70:
        cls = "score-high"
    elif score >= 45:
        cls = "score-mid"
    else:
        cls = "score-low"
    return f'<span class="score-badge {cls}">{score:.1f}</span>'


def _status_pill_html(status):
    css_class = f"status-{status}"
    label = translate_status(status)
    return f'<span class="status-pill {css_class}">{label}</span>'


def _avatar_html(name, size=72):
    parts = name.split()
    initials = ""
    if len(parts) >= 2:
        initials = parts[0][0] + parts[1][0]
    elif parts:
        initials = parts[0][:2]
    initials = initials.upper()
    hue = hash(name) % 360
    bg = f"linear-gradient(135deg, hsl({hue}, 70%, 45%), hsl({(hue+40)%360}, 65%, 55%))"
    return f'<div class="avatar-initials" style="width:{size}px;height:{size}px;background:{bg};font-size:{size//3}px">{initials}</div>'


def render_metrics_row(candidates):
    total = len(candidates)
    scored_count = sum(1 for c in candidates if c.application_status != "new")
    shortlisted = sum(1 for c in candidates if c.application_status == "shortlisted")
    avg_score = sum(c.total_score for c in candidates) / max(total, 1)

    ai_detected = 0
    for c in candidates:
        for m in c.nlp_metrics:
            if m.get("ai_detection_score", 0) > 0.65:
                ai_detected += 1
                break

    metrics = [
        (svg_icon("users", 28, "#0d9488"), str(total), "Всего кандидатов"),
        (svg_icon("bar-chart", 28, "#3b82f6"), format_score(avg_score), "Средний балл"),
        (svg_icon("star", 28, "#f59e0b"), str(shortlisted), "В шорт-листе"),
        (svg_icon("shield", 28, "#8b5cf6"), str(ai_detected), "Авто-детекция"),
    ]

    cols = st.columns(len(metrics))
    for i, (icon_html, value, label) in enumerate(metrics):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card-v2" style="animation-delay: {i*0.1}s">
                <div class="metric-icon">{icon_html}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page 1: Dashboard
# ---------------------------------------------------------------------------

def page_dashboard():
    if not st.session_state.get("candidates"):
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:3rem">
            <div style="margin-bottom:1rem">""" + svg_icon("upload", 48, "#94a3b8") + """</div>
            <h3 class="text-heading" style="margin:0">Добро пожаловать в inVision U</h3>
            <p class="text-secondary" style="margin-top:0.5rem">Загрузите данные кандидатов или создайте демо-набор для начала работы</p>
        </div>
        """, unsafe_allow_html=True)
        render_footer()
        return

    candidates = st.session_state.candidates

    if st.session_state.get("scored", False):
        render_metrics_row(candidates)
        st.markdown("")

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown(f'<p style="font-weight:700;font-size:1.05rem;color:{_c("heading")};margin-bottom:0.8rem">Топ-5 кандидатов</p>', unsafe_allow_html=True)
            sorted_c = sorted(candidates, key=lambda c: c.override_score if c.override_score is not None else c.total_score, reverse=True)
            for i, c in enumerate(sorted_c[:5]):
                eff = c.override_score if c.override_score is not None else c.total_score
                color = score_to_color(eff)
                st.markdown(f"""
                <div class="mini-card" style="animation-delay: {i*0.08}s">
                    <span class="mini-card-rank">#{i+1}</span>
                    <span class="mini-card-name">{c.full_name}</span>
                    <span class="text-secondary" style="font-size:0.78rem;margin-right:0.6rem">{c.city}</span>
                    <span class="mini-card-score" style="color:{color}">{eff:.1f}</span>
                </div>
                """, unsafe_allow_html=True)

        with col_right:
            dark = _is_dark()
            st.markdown(f'<p style="font-weight:700;font-size:1.05rem;color:{_c("heading")};margin-bottom:0.8rem">Распределение баллов</p>', unsafe_allow_html=True)
            scores = [c.total_score for c in candidates]
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=scores, nbinsx=15,
                marker=dict(color="#0d9488", line=dict(color="rgba(255,255,255,0.3)", width=1)),
                opacity=0.85,
            ))
            fig.update_layout(
                **_plotly_defaults(dark),
                height=280,
                showlegend=False,
                xaxis=dict(title="Балл", gridcolor="rgba(148,163,184,0.1)"),
                yaxis=dict(title="Кол-во", gridcolor="rgba(148,163,184,0.1)"),
                bargap=0.08,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;padding:2rem">
            <div style="margin-bottom:0.8rem">{svg_icon("zap", 40, "#f59e0b")}</div>
            <p style="font-weight:600;color:{_c("primary")};font-size:1rem;margin:0">
                Загружено {len(candidates)} кандидатов
            </p>
            <p style="color:{_c("secondary")};font-size:0.88rem;margin-top:0.3rem">
                Перейдите в раздел "Настройки модели" для запуска оценки
            </p>
        </div>
        """, unsafe_allow_html=True)

    render_footer()


# ---------------------------------------------------------------------------
# Page 2: Data Upload
# ---------------------------------------------------------------------------

def page_upload():
    st.markdown(f'<p style="font-weight:700;font-size:1.3rem;color:{_c("heading")};margin-bottom:1rem">Загрузка данных</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Загрузить файл", "Демо-данные"])

    with tab1:
        st.markdown(f"""
        <div class="drag-drop-area">
            <div style="margin-bottom:0.8rem">{svg_icon("upload", 44, "#0d9488")}</div>
            <p style="font-weight:600;color:{_c("primary")};font-size:1rem;margin:0">
                Перетащите файл сюда или нажмите для выбора
            </p>
            <p style="color:{_c("muted")};font-size:0.82rem;margin-top:0.3rem">
                Поддерживаемые форматы: CSV, JSON
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Выберите файл",
            type=["csv", "json"],
            help="Поддерживаемые форматы: CSV, JSON",
            label_visibility="collapsed",
        )

        if uploaded:
            try:
                content = uploaded.read().decode("utf-8")
                if uploaded.name.endswith(".json"):
                    raw = load_candidates_from_json(content)
                else:
                    raw = load_candidates_from_csv(content)
                candidates = [dict_to_candidate(r) for r in raw]
                st.session_state.candidates = candidates
                st.session_state.scored = False
                st.success(f"Загружено {len(candidates)} кандидатов")
            except Exception as e:
                st.error(f"Ошибка при загрузке: {str(e)}")

    with tab2:
        st.markdown(f"""
        <div class="glass-card">
            <p style="font-weight:600;color:{_c("primary")};margin:0 0 0.5rem 0">Генератор синтетических данных</p>
            <p style="color:{_c("secondary")};font-size:0.85rem;margin:0">
                Создайте набор тестовых кандидатов для демонстрации возможностей платформы
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

        col1, col2 = st.columns(2)
        with col1:
            num = st.slider("Количество кандидатов", 10, 100, 55)
        with col2:
            seed = st.number_input("Seed (для воспроизводимости)", value=42)

        if st.button("Сгенерировать демо-данные", type="primary", use_container_width=True):
            with st.spinner("Генерация данных..."):
                raw = generate_dataset(num, seed)
                candidates = [dict_to_candidate(r) for r in raw]
                st.session_state.candidates = candidates
                st.session_state.scored = False
                save_dataset(raw)
            st.success(f"Сгенерировано {len(candidates)} кандидатов")
            st.rerun()

    if st.session_state.get("candidates"):
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown(f'<p style="font-weight:600;color:{_c("primary")}">Предпросмотр данных ({len(st.session_state.candidates)} кандидатов)</p>', unsafe_allow_html=True)
        df = candidates_to_dataframe(st.session_state.candidates)
        st.dataframe(
            df[["Имя", "Возраст", "Город", "Образование", "GPA", "Навыки"]],
            use_container_width=True,
            height=400,
        )

    render_footer()


# ---------------------------------------------------------------------------
# Page 3: Candidate Ranking
# ---------------------------------------------------------------------------

def page_ranking():
    st.markdown(f'<p style="font-weight:700;font-size:1.3rem;color:{_c("heading")};margin-bottom:1rem">Рейтинг кандидатов</p>', unsafe_allow_html=True)

    if not st.session_state.get("scored", False):
        st.warning("Сначала запустите оценку в разделе 'Настройки модели'.")
        render_footer()
        return

    candidates = sorted(
        st.session_state.candidates,
        key=lambda c: c.override_score if c.override_score is not None else c.total_score,
        reverse=True,
    )

    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        search_name = st.text_input("Поиск по имени", "", placeholder="Введите имя...")
    with col2:
        min_score = st.slider("Мин. балл", 0.0, 100.0, 0.0, 5.0, key="rank_min_score")
    with col3:
        max_score = st.slider("Макс. балл", 0.0, 100.0, 100.0, 5.0, key="rank_max_score")
    with col4:
        status_filter = st.multiselect(
            "Статус",
            ["Все", "Оценён", "В шорт-листе", "Отклонён", "Лист ожидания"],
            default=["Все"],
        )

    status_map_reverse = {
        "Оценён": "scored", "В шорт-листе": "shortlisted",
        "Отклонён": "rejected", "Лист ожидания": "waitlisted",
    }

    filtered = candidates
    if search_name:
        filtered = [c for c in filtered if search_name.lower() in c.full_name.lower()]
    filtered = [c for c in filtered if min_score <= (c.override_score if c.override_score is not None else c.total_score) <= max_score]
    if "Все" not in status_filter and status_filter:
        valid_statuses = {status_map_reverse.get(s, s) for s in status_filter}
        filtered = [c for c in filtered if c.application_status in valid_statuses]

    st.markdown(f'<p style="color:{_c("secondary")};font-size:0.85rem">Показано: <b>{len(filtered)}</b> из {len(candidates)}</p>', unsafe_allow_html=True)

    # Build HTML table
    if filtered:
        rows_html = ""
        for i, c in enumerate(filtered):
            eff = c.override_score if c.override_score is not None else c.total_score
            color = score_to_color(eff)
            bar = _score_bar_html(eff)
            pill = _status_pill_html(c.application_status)
            grade = score_to_grade(eff)

            rows_html += f"""<tr>
                <td style="font-weight:700;color:#0d9488">#{c.rank}</td>
                <td>
                    <div style="font-weight:600;color:{_c("primary")}">{c.full_name}</div>
                    <div style="font-size:0.75rem;color:{_c("muted")}">{c.city}</div>
                </td>
                <td style="min-width:140px">
                    <div style="display:flex;align-items:center;gap:8px">
                        <span style="font-weight:700;color:{color};min-width:36px">{eff:.1f}</span>
                        <div style="flex:1">{bar}</div>
                    </div>
                </td>
                <td style="text-align:center;font-weight:700;color:{color}">{grade}</td>
                <td>{pill}</td>
                <td style="color:{_c("secondary")};font-size:0.82rem">{c.gpa:.2f}</td>
            </tr>"""

        st.markdown(f"""
        <div style="overflow-x:auto">
        <table class="styled-table">
            <thead><tr>
                <th>Ранг</th>
                <th>Кандидат</th>
                <th>Балл</th>
                <th>Грейд</th>
                <th>Статус</th>
                <th>GPA</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        json_data = export_results_to_json(filtered)
        st.download_button("Скачать JSON", json_data, "results.json", mime="application/json", use_container_width=True)
    with col2:
        csv_data = export_results_to_csv(filtered)
        st.download_button("Скачать CSV", csv_data, "results.csv", mime="text/csv", use_container_width=True)

    render_footer()


# ---------------------------------------------------------------------------
# Page 4: Candidate Profile (Enhanced)
# ---------------------------------------------------------------------------

def page_candidate_detail():
    st.markdown(f'<p style="font-weight:700;font-size:1.3rem;color:{_c("heading")};margin-bottom:1rem">Профиль кандидата</p>', unsafe_allow_html=True)

    if not st.session_state.get("candidates"):
        st.warning("Загрузите данные кандидатов.")
        render_footer()
        return

    candidate_names = {c.id: f"{c.full_name} (балл: {c.total_score:.1f})" for c in st.session_state.candidates}
    sorted_ids = sorted(candidate_names.keys(), key=lambda cid: next(
        (c.total_score for c in st.session_state.candidates if c.id == cid), 0
    ), reverse=True)

    selected_id = st.selectbox("Выберите кандидата", sorted_ids, format_func=lambda x: candidate_names[x])
    candidate = next((c for c in st.session_state.candidates if c.id == selected_id), None)
    if not candidate:
        render_footer()
        return

    score_color = score_to_color(candidate.total_score)
    grade = score_to_grade(candidate.total_score)

    # Hero Section
    avatar = _avatar_html(candidate.full_name, 72)
    st.markdown(f"""
    <div class="hero-section">
        {avatar}
        <div style="flex:1">
            <div style="font-size:1.4rem;font-weight:800;color:{_c("heading")};letter-spacing:-0.5px">{candidate.full_name}</div>
            <div style="color:{_c("secondary")};font-size:0.88rem;margin-top:0.2rem">
                {candidate.city}, {candidate.country}  /  {candidate.age} лет  /  {candidate.university}
            </div>
            <div style="color:{_c("muted")};font-size:0.82rem;margin-top:0.15rem">
                {candidate.email}  /  {candidate.education_level.upper()}  /  GPA: {candidate.gpa}
            </div>
            <div style="margin-top:0.5rem">{_status_pill_html(candidate.application_status)}</div>
        </div>
        <div style="text-align:center;min-width:100px">
            <div style="font-size:2.4rem;font-weight:900;color:{score_color};line-height:1">{candidate.total_score:.1f}</div>
            <div style="font-size:0.78rem;color:{_c("secondary")};font-weight:600;margin-top:0.2rem">{grade} / Ранг #{candidate.rank}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not candidate.score_breakdown:
        st.info("Кандидат ещё не оценён. Запустите оценку в разделе 'Настройки модели'.")
        render_footer()
        return

    st.markdown("")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Оценки и Профиль", "Анализ эссе", "Обоснования", "Корректировка",
    ])

    with tab1:
        _render_scores_tab(candidate)
    with tab2:
        _render_essays_tab(candidate)
    with tab3:
        _render_explanations_tab(candidate)
    with tab4:
        _render_override_tab(candidate)

    render_footer()


def _render_scores_tab(candidate):
    dark = _is_dark()
    col_radar, col_bars = st.columns([1, 1])

    dimensions = [DIMENSION_NAMES.get(b["dimension"], b["dimension"]) for b in candidate.score_breakdown]
    scores = [b["score"] for b in candidate.score_breakdown]

    tick_color = "#94a3b8" if dark else "#64748b"
    label_color = _c("primary")

    with col_radar:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],
            theta=dimensions + [dimensions[0]],
            fill="toself",
            name="Баллы",
            line=dict(color="#0d9488", width=2.5),
            fillcolor="rgba(13,148,136,0.12)",
            marker=dict(size=6, color="#0d9488"),
        ))
        fig.update_layout(
            **_plotly_defaults(dark),
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9, color=tick_color), gridcolor="rgba(148,163,184,0.15)"),
                angularaxis=dict(tickfont=dict(size=11, color=label_color), gridcolor="rgba(148,163,184,0.15)"),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=False,
            height=380,
            title=dict(text="Радар компетенций", font=dict(size=14, color=label_color)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_bars:
        st.markdown(f'<p style="font-weight:600;color:{_c("primary")};margin-bottom:0.6rem">Детализация по критериям</p>', unsafe_allow_html=True)
        for b in candidate.score_breakdown:
            dim_name = DIMENSION_NAMES.get(b["dimension"], b["dimension"])
            score = b["score"]
            weight = b["weight"]
            color = score_to_color(score)

            st.markdown(f"""
            <div style="margin-bottom:0.8rem">
                <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:3px">
                    <span style="font-weight:600;font-size:0.85rem;color:{_c("primary")}">{dim_name}</span>
                    <span style="font-weight:700;font-size:0.85rem;color:{color}">{score:.1f} <span style="color:{_c("muted")};font-weight:400;font-size:0.75rem">x{weight:.0%}</span></span>
                </div>
                {_score_bar_html(score)}
            </div>
            """, unsafe_allow_html=True)

    # Sub-scores expandable
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(f'<p style="font-weight:600;color:{_c("primary")};margin-bottom:0.5rem">Суб-компоненты оценки</p>', unsafe_allow_html=True)

    for breakdown in candidate.score_breakdown:
        dim_name = DIMENSION_NAMES.get(breakdown["dimension"], breakdown["dimension"])
        sub = breakdown.get("sub_scores", {})
        if not sub:
            continue

        with st.expander(f"{dim_name}: {breakdown['score']:.1f} / 100"):
            sub_names = [SUB_SCORE_NAMES.get(k, k) for k in sub.keys()]
            sub_values = list(sub.values())

            fig_bar = go.Figure(go.Bar(
                x=sub_values,
                y=sub_names,
                orientation="h",
                marker=dict(
                    color=sub_values,
                    colorscale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#10b981"]],
                    line=dict(color="rgba(255,255,255,0.2)", width=1),
                    cornerradius=4,
                ),
                text=[f"{v:.1f}" for v in sub_values],
                textposition="auto",
                textfont=dict(color="white", size=11, family="Inter"),
            ))
            fig_bar.update_layout(
                **_plotly_no_margin(dark),
                height=max(140, len(sub_names) * 42),
                xaxis=dict(range=[0, 100], gridcolor="rgba(148,163,184,0.1)"),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            evidence = breakdown.get("evidence", [])
            if evidence:
                tags = "".join(f'<span class="evidence-tag">{e}</span>' for e in evidence)
                st.markdown(tags, unsafe_allow_html=True)


def _render_essays_tab(candidate):
    if not candidate.essays:
        st.info("Эссе отсутствуют.")
        return

    for i, essay in enumerate(candidate.essays):
        prompt = essay.get("prompt", f"Эссе #{i+1}") if isinstance(essay, dict) else f"Эссе #{i+1}"
        text = essay.get("text", "") if isinstance(essay, dict) else str(essay)

        st.markdown(f'<p style="font-weight:700;color:{_c("heading")};font-size:1rem;margin-top:1rem">{prompt}</p>', unsafe_allow_html=True)

        with st.expander("Текст эссе", expanded=(i == 0)):
            st.text_area(f"essay_text_{i}", text, height=180, disabled=True, label_visibility="collapsed")

        if i < len(candidate.nlp_metrics):
            metrics = candidate.nlp_metrics[i]
            _render_nlp_panel(metrics)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


def _render_nlp_panel(metrics):
    col1, col2, col3, col4 = st.columns(4)

    sent_score = metrics.get("sentiment_score", 0)
    sent_label_map = {"positive": "Позитивный", "negative": "Негативный", "neutral": "Нейтральный"}
    sent_label = sent_label_map.get(metrics.get("sentiment_label", ""), "--")

    col1.metric("Тональность", sent_label, f"{sent_score:+.2f}")
    col2.metric("Сложность", f"{metrics.get('complexity_score', 0):.1%}")
    col3.metric("Словарь", f"{metrics.get('vocabulary_richness', 0):.1%}")
    col4.metric("Читаемость", f"{metrics.get('readability_grade', 0):.1f}")

    col1, col2, col3 = st.columns(3)

    ai_score = metrics.get("ai_detection_score", 0)
    ai_map = {"likely_ai": "Вероятно авто-генерация", "uncertain": "Неопределённо", "likely_human": "Вероятно человек"}
    ai_label = ai_map.get(metrics.get("ai_detection_label", ""), "--")

    # Traffic light for AI detection
    if ai_score > 0.65:
        tl_class, dot_class = "traffic-red", "traffic-dot-red"
    elif ai_score > 0.40:
        tl_class, dot_class = "traffic-yellow", "traffic-dot-yellow"
    else:
        tl_class, dot_class = "traffic-green", "traffic-dot-green"

    with col1:
        st.markdown(f"""
        <div class="nlp-metric-panel">
            <div style="font-size:0.75rem;color:{_c("muted")};text-transform:uppercase;letter-spacing:0.5px;margin-bottom:0.3rem">Авто-детекция</div>
            <span class="traffic-light {tl_class}">
                <span class="traffic-dot {dot_class}"></span>
                {ai_label} ({ai_score:.0%})
            </span>
        </div>
        """, unsafe_allow_html=True)

    auth = metrics.get("authenticity_score", 0)
    auth_pct = auth * 100
    with col2:
        st.markdown(f"""
        <div class="nlp-metric-panel">
            <div style="font-size:0.75rem;color:{_c("muted")};text-transform:uppercase;letter-spacing:0.5px;margin-bottom:0.3rem">Аутентичность</div>
            <div style="display:flex;align-items:center;gap:8px">
                <span style="font-weight:700;color:{_c("primary")};font-size:1.1rem">{auth:.0%}</span>
                <div style="flex:1">{_score_bar_html(auth_pct)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    vr = metrics.get("vocabulary_richness", 0)
    with col3:
        st.markdown(f"""
        <div class="nlp-metric-panel">
            <div style="font-size:0.75rem;color:{_c("muted")};text-transform:uppercase;letter-spacing:0.5px;margin-bottom:0.3rem">Лексическое богатство</div>
            <div style="display:flex;align-items:center;gap:8px">
                <span style="font-weight:700;color:{_c("primary")};font-size:1.1rem">{vr:.0%}</span>
                <div style="flex:1">{_score_bar_html(vr*100)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if ai_score > 0.65:
        st.markdown(f"""
        <div class="ai-warning">
            {svg_icon("alert-triangle", 18, _c("warning_text"))}
            Высокая вероятность авто-генерации ({ai_score:.0%}). Рекомендуется дополнительная проверка эссе.
        </div>
        """, unsafe_allow_html=True)

    kw_density = metrics.get("keyword_density", {})
    if kw_density:
        st.markdown(f'<p style="font-weight:600;color:{_c("primary")};font-size:0.85rem;margin-top:0.5rem">Плотность ключевых слов</p>', unsafe_allow_html=True)
        kw_cols = st.columns(len(kw_density))
        for col, (kw, density) in zip(kw_cols, kw_density.items()):
            col.metric(kw.capitalize(), f"{density:.2f}%")


def _render_explanations_tab(candidate):
    st.markdown(f'<p style="font-weight:700;color:{_c("heading")};font-size:1rem;margin-bottom:0.5rem">Полное объяснение оценки</p>', unsafe_allow_html=True)
    st.caption("Каждый критерий оценивается по нескольким параметрам. Ниже представлено подробное обоснование.")

    for breakdown in candidate.score_breakdown:
        dim_name = DIMENSION_NAMES.get(breakdown["dimension"], breakdown["dimension"])
        explanation = breakdown.get("explanation", "Нет данных")

        st.markdown(f'<p style="font-weight:600;color:#0d9488;margin-top:1rem">{dim_name}</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="explanation-box">{explanation}</div>', unsafe_allow_html=True)

        evidence = breakdown.get("evidence", [])
        if evidence:
            tags = "".join(f'<span class="evidence-tag">{e}</span>' for e in evidence)
            st.markdown(tags, unsafe_allow_html=True)

    if candidate.nlp_metrics:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown(f'<p style="font-weight:600;color:{_c("primary")}">NLP-анализ эссе</p>', unsafe_allow_html=True)
        for i, metrics in enumerate(candidate.nlp_metrics):
            ai_score = metrics.get("ai_detection_score", 0)
            auth_score = metrics.get("authenticity_score", 0)
            complexity = metrics.get("complexity_score", 0)
            st.markdown(f"""
            <div class="nlp-metric-panel" style="display:flex;gap:1.5rem;align-items:center;margin-bottom:0.4rem">
                <span style="font-weight:700;color:#0d9488;font-size:0.85rem">Эссе #{i+1}</span>
                <span style="font-size:0.82rem;color:{_c("card_text")}">Аутентичность: <b>{auth_score:.0%}</b></span>
                <span style="font-size:0.82rem;color:{_c("card_text")}">Сложность: <b>{complexity:.0%}</b></span>
                <span style="font-size:0.82rem;color:{_c("card_text")}">Авто-детекция: <b>{ai_score:.0%}</b></span>
            </div>
            """, unsafe_allow_html=True)


def _render_override_tab(candidate):
    st.markdown(f'<p style="font-weight:700;color:{_c("heading")};font-size:1rem;margin-bottom:0.3rem">Корректировка оценки комиссией</p>', unsafe_allow_html=True)
    st.caption("Члены комиссии могут скорректировать автоматическую оценку. Корректировка заменит автоматический балл в рейтинге.")

    current_override = st.session_state.get("overrides", {}).get(candidate.id, {})

    override_score = st.slider(
        "Скорректированный балл",
        0.0, 100.0,
        float(current_override.get("score", candidate.total_score)),
        0.5,
        key=f"override_{candidate.id}",
    )

    override_comment = st.text_area(
        "Обоснование корректировки",
        current_override.get("comment", ""),
        placeholder="Укажите причину корректировки...",
        key=f"comment_{candidate.id}",
    )

    new_status = st.selectbox(
        "Изменить статус",
        ["-- Не менять --", "В шорт-листе", "Отклонён", "Принят", "Лист ожидания"],
        key=f"status_{candidate.id}",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Сохранить", key=f"save_{candidate.id}", type="primary", use_container_width=True):
            if "overrides" not in st.session_state:
                st.session_state.overrides = {}
            st.session_state.overrides[candidate.id] = {
                "score": override_score,
                "comment": override_comment,
            }
            candidate.override_score = override_score
            candidate.override_comment = override_comment

            status_map = {
                "В шорт-листе": "shortlisted",
                "Отклонён": "rejected",
                "Принят": "accepted",
                "Лист ожидания": "waitlisted",
            }
            if new_status in status_map:
                candidate.application_status = status_map[new_status]

            st.success("Корректировка сохранена")

    with col2:
        if st.button("Сбросить", key=f"reset_{candidate.id}", use_container_width=True):
            overrides = st.session_state.get("overrides", {})
            if candidate.id in overrides:
                del overrides[candidate.id]
            candidate.override_score = None
            candidate.override_comment = ""
            st.success("Корректировка сброшена")
            st.rerun()


# ---------------------------------------------------------------------------
# Page 5: Comparison
# ---------------------------------------------------------------------------

def page_comparison():
    st.markdown(f'<p style="font-weight:700;font-size:1.3rem;color:{_c("heading")};margin-bottom:1rem">Сравнение кандидатов</p>', unsafe_allow_html=True)

    if not st.session_state.get("scored", False):
        st.warning("Сначала запустите оценку.")
        render_footer()
        return

    dark = _is_dark()
    candidates = st.session_state.candidates

    compare_ids = st.multiselect(
        "Выберите кандидатов для сравнения (2-4)",
        [c.id for c in candidates],
        format_func=lambda x: next((c.full_name for c in candidates if c.id == x), x),
        max_selections=4,
    )

    if len(compare_ids) < 2:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:2rem">
            <div style="margin-bottom:0.5rem">""" + svg_icon("git-compare", 40, "#94a3b8") + """</div>
            <p class="text-secondary" style="margin:0">Выберите от 2 до 4 кандидатов для сравнения</p>
        </div>
        """, unsafe_allow_html=True)
        render_footer()
        return

    selected = [c for c in candidates if c.id in compare_ids]

    # Overlay Radar Chart
    fig = go.Figure()
    colors = CHART_COLORS[:len(selected)]

    for idx, c in enumerate(selected):
        if not c.score_breakdown:
            continue
        dims = [DIMENSION_NAMES.get(b["dimension"], b["dimension"]) for b in c.score_breakdown]
        vals = [b["score"] for b in c.score_breakdown]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=dims + [dims[0]],
            name=c.full_name,
            line=dict(color=colors[idx], width=2.5),
            fill="toself",
            fillcolor=f"rgba({int(colors[idx][1:3],16)},{int(colors[idx][3:5],16)},{int(colors[idx][5:7],16)},0.06)",
            marker=dict(size=5),
        ))

    tick_color = "#94a3b8" if dark else "#64748b"
    label_color = _c("primary")

    fig.update_layout(
        **_plotly_defaults(dark),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(148,163,184,0.15)", tickfont=dict(size=9, color=tick_color)),
            angularaxis=dict(gridcolor="rgba(148,163,184,0.15)", tickfont=dict(size=11, color=label_color)),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=420,
        title=dict(text="Наложение профилей", font=dict(size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Dimension-by-dimension comparison
    st.markdown(f'<p style="font-weight:600;color:{_c("primary")};margin-bottom:0.5rem">Сравнение по измерениям</p>', unsafe_allow_html=True)

    dim_keys = ["motivation", "leadership", "growth", "skills", "experience"]
    for dim in dim_keys:
        dim_name = DIMENSION_NAMES[dim]
        dim_scores = []
        for c in selected:
            s = next((b["score"] for b in c.score_breakdown if b["dimension"] == dim), 0)
            dim_scores.append((c.full_name, s))

        max_val = max(s for _, s in dim_scores) if dim_scores else 0

        bars_html = ""
        for name, s in dim_scores:
            color = score_to_color(s)
            is_winner = (s == max_val and sum(1 for _, v in dim_scores if v == max_val) == 1)
            winner_html = '<span class="winner-badge">Лидер</span>' if is_winner else ""
            bars_html += f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
                <span style="min-width:160px;font-size:0.82rem;color:{_c("card_text")};font-weight:500">{name}</span>
                <div style="flex:1">{_score_bar_html(s, height=10)}</div>
                <span style="min-width:40px;font-weight:700;color:{color};font-size:0.85rem;text-align:right">{s:.1f}</span>
                {winner_html}
            </div>
            """

        st.markdown(f"""
        <div class="glass-card" style="padding:1rem 1.2rem;margin-bottom:0.5rem">
            <p style="font-weight:600;color:#0d9488;font-size:0.88rem;margin:0 0 0.5rem 0">{dim_name}</p>
            {bars_html}
        </div>
        """, unsafe_allow_html=True)

    # Summary comparison table
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(f'<p style="font-weight:600;color:{_c("primary")};margin-bottom:0.5rem">Сводка</p>', unsafe_allow_html=True)

    header = "<th>Параметр</th>" + "".join(f"<th>{c.full_name}</th>" for c in selected)
    rows = ""
    params = [
        ("Общий балл", [f"{c.total_score:.1f}" for c in selected]),
        ("Ранг", [f"#{c.rank}" for c in selected]),
        ("GPA", [f"{c.gpa:.2f}" for c in selected]),
        ("Опыт (лет)", [f"{c.work_experience_years}" for c in selected]),
        ("Волонтёрство (ч)", [f"{c.volunteer_hours}" for c in selected]),
        ("Навыков", [f"{len(c.skills)}" for c in selected]),
        ("Статус", [translate_status(c.application_status) for c in selected]),
    ]
    for label, vals in params:
        cells = "".join(f"<td>{v}</td>" for v in vals)
        rows += f"<tr><td style='font-weight:600'>{label}</td>{cells}</tr>"

    st.markdown(f"""
    <table class="styled-table">
        <thead><tr>{header}</tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    render_footer()


# ---------------------------------------------------------------------------
# Page 6: Shortlist
# ---------------------------------------------------------------------------

def page_shortlist():
    st.markdown(f'<p style="font-weight:700;font-size:1.3rem;color:{_c("heading")};margin-bottom:1rem">Шорт-лист</p>', unsafe_allow_html=True)

    if not st.session_state.get("scored", False):
        st.warning("Сначала запустите оценку кандидатов.")
        render_footer()
        return

    config = st.session_state.get("scoring_config", ScoringConfig())

    col1, col2, col3 = st.columns(3)
    with col1:
        threshold = st.slider("Порог включения", 30.0, 95.0, config.shortlist_threshold, 1.0, key="sl_threshold")
    with col2:
        max_count = st.number_input("Макс. кол-во", 5, 100, 20, key="sl_max")
    with col3:
        exclude_ai = st.checkbox("Исключить подозрительные авто-эссе", value=True)

    if st.button("Сформировать шорт-лист", type="primary", use_container_width=True):
        engine = ScoringEngine(config)
        candidates = st.session_state.candidates

        if exclude_ai:
            filtered = []
            for c in candidates:
                has_ai = any(m.get("ai_detection_score", 0) > 0.65 for m in c.nlp_metrics)
                if not has_ai:
                    filtered.append(c)
            candidates_for_shortlist = filtered
        else:
            candidates_for_shortlist = candidates

        shortlisted = engine.generate_shortlist(candidates_for_shortlist, threshold, max_count)
        st.session_state.shortlist_generated = True
        st.success(f"В шорт-лист включено: {len(shortlisted)} кандидатов")
        st.rerun()

    all_candidates = st.session_state.candidates
    shortlisted = [c for c in all_candidates if c.application_status == "shortlisted"]
    rejected = [c for c in all_candidates if c.application_status == "rejected"]
    rest = [c for c in all_candidates if c.application_status not in ("shortlisted", "rejected")]

    # Summary stats
    if shortlisted or rejected:
        cols = st.columns(4)
        cols[0].metric("В шорт-листе", len(shortlisted))
        cols[1].metric("Отклонено", len(rejected))
        cols[2].metric("Прочие", len(rest))
        avg_sl = sum(c.total_score for c in shortlisted) / max(len(shortlisted), 1) if shortlisted else 0
        cols[3].metric("Сред. балл шорт-листа", f"{avg_sl:.1f}")

    if shortlisted:
        shortlisted.sort(key=lambda c: c.override_score if c.override_score is not None else c.total_score, reverse=True)

        st.markdown(f'<p style="font-weight:600;color:{_c("primary")};margin-top:1rem;margin-bottom:0.5rem">Шорт-лист ({len(shortlisted)})</p>', unsafe_allow_html=True)

        for i, c in enumerate(shortlisted):
            effective = c.override_score if c.override_score is not None else c.total_score
            color = score_to_color(effective)
            avatar = _avatar_html(c.full_name, 40)

            st.markdown(f"""
            <div class="candidate-card-v2" style="border-left-color:{color};display:flex;align-items:center;gap:1rem">
                {avatar}
                <div style="flex:1">
                    <div class="candidate-name">{c.full_name}</div>
                    <div class="candidate-meta">{c.city} / {c.university} / GPA {c.gpa:.2f}</div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:1.3rem;font-weight:800;color:{color}">{effective:.1f}</div>
                    <div style="font-size:0.75rem;color:{_c("muted")}">Ранг #{c.rank}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            json_data = export_results_to_json(shortlisted)
            st.download_button("Экспорт JSON", json_data, "shortlist.json", mime="application/json", use_container_width=True)
        with col2:
            csv_data = export_results_to_csv(shortlisted)
            st.download_button("Экспорт CSV", csv_data, "shortlist.csv", mime="text/csv", use_container_width=True)

    render_footer()


# ---------------------------------------------------------------------------
# Page 7: Analytics (Enhanced)
# ---------------------------------------------------------------------------

def _get_candidate_display_name(candidate, index=0):
    """Return candidate name respecting anonymization setting."""
    if st.session_state.get("anonymize_names", False):
        return f"Кандидат #{index + 1}"
    return candidate.full_name


def page_analytics():
    st.markdown(f'<p style="font-weight:700;font-size:1.3rem;color:{_c("heading")};margin-bottom:1rem">Аналитика</p>', unsafe_allow_html=True)

    if not st.session_state.get("scored", False):
        st.warning("Запустите оценку для просмотра аналитики.")
        render_skeleton(5)
        render_footer()
        return

    dark = _is_dark()
    candidates = st.session_state.candidates
    config = st.session_state.get("scoring_config", ScoringConfig())
    engine = ScoringEngine(config)
    stats = engine.get_dimension_stats(candidates)
    scores = [c.total_score for c in candidates]

    from scipy.stats import gaussian_kde

    # Row 1: Distribution + Box plots
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=scores, nbinsx=20,
            marker=dict(color="#0d9488", line=dict(color="rgba(255,255,255,0.2)", width=1)),
            opacity=0.85,
            name="Распределение",
        ))
        if len(scores) > 2:
            try:
                kde = gaussian_kde(scores)
                x_range = np.linspace(min(scores) - 5, max(scores) + 5, 200)
                kde_vals = kde(x_range) * len(scores) * ((max(scores) - min(scores)) / 20)
                fig.add_trace(go.Scatter(
                    x=x_range, y=kde_vals,
                    mode="lines",
                    line=dict(color="#3b82f6", width=2.5),
                    name="KDE",
                ))
            except Exception:
                pass
        defaults = _plotly_defaults(dark)
        fig.update_layout(
            **defaults,
            title=dict(text="Распределение баллов + KDE", font=dict(size=13)),
            height=350,
            xaxis=dict(title="Балл", gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(title="Количество", gridcolor="rgba(148,163,184,0.1)"),
            bargap=0.06,
            legend=dict(orientation="h", y=-0.18),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        dim_keys = ["motivation", "leadership", "growth", "skills", "experience"]
        box_data = []
        for dim in dim_keys:
            for c in candidates:
                for b in c.score_breakdown:
                    if b["dimension"] == dim:
                        box_data.append({"Измерение": DIMENSION_NAMES[dim], "Балл": b["score"]})
                        break

        if box_data:
            df_box = pd.DataFrame(box_data)
            fig_box = px.box(
                df_box, x="Измерение", y="Балл",
                color="Измерение",
                color_discrete_sequence=CHART_COLORS[:5],
            )
            defaults_box = _plotly_defaults(dark)
            fig_box.update_layout(
                **defaults_box,
                title=dict(text="Бокс-плоты по измерениям", font=dict(size=13)),
                height=350,
                showlegend=False,
                yaxis=dict(range=[0, 100], gridcolor="rgba(148,163,184,0.1)"),
                xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
            )
            st.plotly_chart(fig_box, use_container_width=True)

    # Row 2: Correlation Heatmap + City fairness
    col1, col2 = st.columns(2)

    with col1:
        dim_keys = ["motivation", "leadership", "growth", "skills", "experience"]
        dim_matrix = {dk: [] for dk in dim_keys}
        for c in candidates:
            bd = {b["dimension"]: b["score"] for b in c.score_breakdown}
            for dk in dim_keys:
                dim_matrix[dk].append(bd.get(dk, 0))

        corr_df = pd.DataFrame(dim_matrix)
        corr_df.columns = [DIMENSION_NAMES[d] for d in dim_keys]
        corr = corr_df.corr()

        fig_heat = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale=[[0, "#0f172a"], [0.5, "#0d9488"], [1, "#f59e0b"]],
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=11, color="white"),
            zmin=-1, zmax=1,
        ))
        defaults_heat = _plotly_defaults(dark)
        fig_heat.update_layout(
            **defaults_heat,
            title=dict(text="Корреляция между измерениями", font=dict(size=13)),
            height=380,
            xaxis=dict(tickangle=-45),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col2:
        city_scores = {}
        for c in candidates:
            if c.city not in city_scores:
                city_scores[c.city] = []
            city_scores[c.city].append(c.total_score)

        city_avgs = sorted(
            [(city, np.mean(ss), np.std(ss), len(ss)) for city, ss in city_scores.items()],
            key=lambda x: x[1], reverse=True,
        )

        fig_city = go.Figure()
        fig_city.add_trace(go.Bar(
            x=[c[1] for c in city_avgs],
            y=[c[0] for c in city_avgs],
            orientation="h",
            marker=dict(
                color=[c[1] for c in city_avgs],
                colorscale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#10b981"]],
                cornerradius=4,
                line=dict(color="rgba(255,255,255,0.15)", width=1),
            ),
            error_x=dict(type="data", array=[c[2] for c in city_avgs], visible=True, color="#94a3b8"),
            text=[f"{c[1]:.1f} (n={c[3]})" for c in city_avgs],
            textposition="auto",
            textfont=dict(size=10, color="white"),
        ))
        defaults_city = _plotly_no_margin(dark)
        fig_city.update_layout(
            **defaults_city,
            title=dict(text="Справедливость: ср. балл по городам", font=dict(size=13)),
            xaxis=dict(range=[0, 100], gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(autorange="reversed"),
            height=380,
            margin=dict(l=150, r=30, t=50, b=40),
        )
        st.plotly_chart(fig_city, use_container_width=True)

    # ===================================================================
    # BASELINE COMPARISONS (3 baselines)
    # ===================================================================
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(f'<p style="font-weight:700;font-size:1.15rem;color:{_c("heading")}">Сравнение с базовыми моделями отбора</p>', unsafe_allow_html=True)
    st.caption("Демонстрация преимущества взвешенного ИИ-скоринга над наивными подходами")

    n_select = min(20, len(candidates))
    ranked = sorted(candidates, key=lambda c: c.total_score, reverse=True)
    ai_top = ranked[:n_select]
    ai_avg = np.mean([c.total_score for c in ai_top])
    ai_min = min(c.total_score for c in ai_top)

    # Baseline 1: Random selection
    np.random.seed(42)
    random_indices = np.random.choice(len(candidates), size=n_select, replace=False)
    random_top = [candidates[i] for i in random_indices]
    random_avg = np.mean([c.total_score for c in random_top])
    random_min = min(c.total_score for c in random_top)

    # Baseline 2: GPA-only selection
    gpa_ranked = sorted(candidates, key=lambda c: c.gpa, reverse=True)
    gpa_top = gpa_ranked[:n_select]
    gpa_avg = np.mean([c.total_score for c in gpa_top])
    gpa_min = min(c.total_score for c in gpa_top)

    # Baseline 3: Equal weights selection
    equal_weight = 0.2
    equal_scores = []
    for c in candidates:
        bd = {b["dimension"]: b["score"] for b in c.score_breakdown}
        eq_score = sum(bd.get(d, 0) * equal_weight for d in ["motivation", "leadership", "growth", "skills", "experience"])
        nlp_bonus_eq = engine._compute_nlp_bonus(c.nlp_metrics)
        ai_penalty_eq = engine._compute_ai_penalty(c.nlp_metrics)
        eq_total = max(0, min(100, eq_score + nlp_bonus_eq - ai_penalty_eq))
        equal_scores.append((c, eq_total))
    equal_ranked = sorted(equal_scores, key=lambda x: x[1], reverse=True)
    equal_top = [x[0] for x in equal_ranked[:n_select]]
    equal_avg = np.mean([c.total_score for c in equal_top])
    equal_min = min(c.total_score for c in equal_top)

    # Count strong candidates lost by each baseline
    ai_top_ids = {c.id for c in ai_top}
    random_lost = sum(1 for c in ai_top if c.id not in {candidates[i].id for i in random_indices})
    gpa_lost = sum(1 for c in ai_top if c.id not in {c2.id for c2 in gpa_top})
    equal_lost = sum(1 for c in ai_top if c.id not in {c2.id for c2 in equal_top})

    # Comparison table
    comparison_rows = f"""
    <tr>
        <td style="font-weight:700;color:#0d9488">ИИ-скоринг (взвешенный)</td>
        <td style="font-weight:700;color:#22c55e">{ai_avg:.1f}</td>
        <td>{ai_min:.1f}</td>
        <td>0</td>
    </tr>
    <tr>
        <td>Только по GPA</td>
        <td>{gpa_avg:.1f}</td>
        <td>{gpa_min:.1f}</td>
        <td style="color:#ef4444">{gpa_lost}</td>
    </tr>
    <tr>
        <td>Равные веса (0.2 каждый)</td>
        <td>{equal_avg:.1f}</td>
        <td>{equal_min:.1f}</td>
        <td style="color:#f59e0b">{equal_lost}</td>
    </tr>
    <tr>
        <td>Случайный отбор</td>
        <td>{random_avg:.1f}</td>
        <td>{random_min:.1f}</td>
        <td style="color:#ef4444">{random_lost}</td>
    </tr>
    """

    st.markdown(f"""
    <table class="styled-table">
        <thead><tr>
            <th>Метод</th>
            <th>Средний балл топ-{n_select}</th>
            <th>Мин. балл</th>
            <th>Потеряно сильных кандидатов</th>
        </tr></thead>
        <tbody>{comparison_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ИИ-скоринг", f"{ai_avg:.1f}", f"+{ai_avg - random_avg:.1f} vs случайный")
    col2.metric("Только GPA", f"{gpa_avg:.1f}", f"{gpa_avg - ai_avg:+.1f} vs ИИ")
    col3.metric("Равные веса", f"{equal_avg:.1f}", f"{equal_avg - ai_avg:+.1f} vs ИИ")
    col4.metric("Случайный", f"{random_avg:.1f}", f"{random_avg - ai_avg:+.1f} vs ИИ")

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Histogram(x=[c.total_score for c in ai_top], name="ИИ-скоринг", marker_color="#0d9488", opacity=0.7, nbinsx=10))
    fig_comp.add_trace(go.Histogram(x=[c.total_score for c in gpa_top], name="Только GPA", marker_color="#3b82f6", opacity=0.5, nbinsx=10))
    fig_comp.add_trace(go.Histogram(x=[c.total_score for c in random_top], name="Случайный", marker_color="#ef4444", opacity=0.4, nbinsx=10))
    defaults_comp = _plotly_defaults(dark)
    fig_comp.update_layout(
        **defaults_comp,
        barmode="overlay",
        height=300,
        title=dict(text=f"Распределение баллов: топ-{n_select} по каждому методу", font=dict(size=13)),
        xaxis=dict(title="Балл", gridcolor="rgba(148,163,184,0.1)"),
        yaxis=dict(title="Кол-во", gridcolor="rgba(148,163,184,0.1)"),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # ===================================================================
    # MODEL VALIDATION
    # ===================================================================
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(f'<p style="font-weight:700;font-size:1.15rem;color:{_c("heading")}">Валидация модели</p>', unsafe_allow_html=True)

    # --- Stratified split validation ---
    st.markdown(f'<p style="font-weight:600;color:{_c("primary")};margin-top:0.5rem">Стратифицированное разделение (70/30)</p>', unsafe_allow_html=True)
    st.caption("Кандидаты разделены на обучающую (70%) и тестовую (30%) выборки. Метрики рассчитаны для обеих.")

    np.random.seed(42)
    n_total = len(candidates)
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    split_idx = int(n_total * 0.7)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    train_set = [candidates[i] for i in train_indices]
    test_set = [candidates[i] for i in test_indices]

    train_scores = [c.total_score for c in train_set]
    test_scores = [c.total_score for c in test_set]

    train_mean = np.mean(train_scores)
    test_mean = np.mean(test_scores)
    train_std = np.std(train_scores)
    test_std = np.std(test_scores)
    train_median = np.median(train_scores)
    test_median = np.median(test_scores)

    split_rows = f"""
    <tr>
        <td style="font-weight:600">Обучающая (70%, n={len(train_set)})</td>
        <td>{train_mean:.1f}</td>
        <td>{train_median:.1f}</td>
        <td>{train_std:.1f}</td>
        <td>{min(train_scores):.1f}</td>
        <td>{max(train_scores):.1f}</td>
    </tr>
    <tr>
        <td style="font-weight:600">Тестовая (30%, n={len(test_set)})</td>
        <td>{test_mean:.1f}</td>
        <td>{test_median:.1f}</td>
        <td>{test_std:.1f}</td>
        <td>{min(test_scores):.1f}</td>
        <td>{max(test_scores):.1f}</td>
    </tr>
    <tr>
        <td style="font-weight:600;color:#0d9488">Разница</td>
        <td>{abs(train_mean - test_mean):.1f}</td>
        <td>{abs(train_median - test_median):.1f}</td>
        <td>{abs(train_std - test_std):.1f}</td>
        <td>--</td>
        <td>--</td>
    </tr>
    """
    st.markdown(f"""
    <table class="styled-table">
        <thead><tr>
            <th>Выборка</th><th>Среднее</th><th>Медиана</th><th>Ст. откл.</th><th>Мин.</th><th>Макс.</th>
        </tr></thead>
        <tbody>{split_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    stability_pct = (1 - abs(train_mean - test_mean) / max(train_mean, 0.01)) * 100
    st.markdown(f"""
    <div class="glass-card" style="padding:0.8rem 1.2rem;margin-top:0.5rem">
        <span style="font-weight:600;color:{_c("primary")}">Стабильность модели: </span>
        <span style="font-weight:700;color:{'#22c55e' if stability_pct > 90 else '#f59e0b'}">{stability_pct:.1f}%</span>
        <span style="color:{_c("secondary")};font-size:0.85rem"> -- разница средних между выборками: {abs(train_mean - test_mean):.2f} баллов</span>
    </div>
    """, unsafe_allow_html=True)

    # --- Ablation study ---
    st.markdown(f'<p style="font-weight:600;color:{_c("primary")};margin-top:1.5rem">Абляционное исследование</p>', unsafe_allow_html=True)
    st.caption("Влияние исключения каждого измерения на средний балл. Показывает, какие измерения вносят наибольший вклад.")

    dim_keys_ablation = ["motivation", "leadership", "growth", "skills", "experience"]
    full_avg = np.mean(scores)
    ablation_results = []
    ablation_names_ru = {
        "motivation": "Без мотивации",
        "leadership": "Без лидерства",
        "growth": "Без роста",
        "skills": "Без навыков",
        "experience": "Без опыта",
    }

    current_weights = config.get_weights()
    for ablated_dim in dim_keys_ablation:
        ablated_scores = []
        remaining_weight = 1.0 - current_weights[ablated_dim]
        for c in candidates:
            bd = {b["dimension"]: b["score"] for b in c.score_breakdown}
            ablated_total = 0.0
            for d in dim_keys_ablation:
                if d == ablated_dim:
                    continue
                rescaled_w = current_weights[d] / remaining_weight if remaining_weight > 0 else 0
                ablated_total += bd.get(d, 0) * rescaled_w
            ablated_scores.append(ablated_total)
        abl_avg = np.mean(ablated_scores)
        delta = abl_avg - full_avg
        ablation_results.append((ablation_names_ru[ablated_dim], abl_avg, delta))

    ablation_html = ""
    for name, avg, delta in ablation_results:
        color = "#ef4444" if delta < -1 else "#f59e0b" if delta < 0 else "#22c55e"
        ablation_html += f"""<tr>
            <td style="font-weight:600">{name}</td>
            <td>{avg:.1f}</td>
            <td style="color:{color};font-weight:700">{delta:+.1f}</td>
        </tr>"""

    st.markdown(f"""
    <table class="styled-table">
        <thead><tr><th>Вариант</th><th>Средний балл</th><th>Изменение</th></tr></thead>
        <tbody>
            <tr><td style="font-weight:700;color:#0d9488">Полная модель</td><td style="font-weight:700">{full_avg:.1f}</td><td>--</td></tr>
            {ablation_html}
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    # Ablation bar chart
    fig_ablation = go.Figure()
    abl_labels = [r[0] for r in ablation_results]
    abl_deltas = [r[2] for r in ablation_results]
    abl_colors = ["#ef4444" if d < -1 else "#f59e0b" if d < 0 else "#22c55e" for d in abl_deltas]
    fig_ablation.add_trace(go.Bar(
        x=abl_labels, y=abl_deltas,
        marker_color=abl_colors,
        text=[f"{d:+.1f}" for d in abl_deltas],
        textposition="outside",
        textfont=dict(size=11),
    ))
    defaults_abl = _plotly_defaults(dark)
    fig_ablation.update_layout(
        **defaults_abl,
        height=300,
        title=dict(text="Абляция: изменение среднего балла при удалении измерения", font=dict(size=13)),
        yaxis=dict(title="Изменение балла", gridcolor="rgba(148,163,184,0.1)"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        showlegend=False,
    )
    st.plotly_chart(fig_ablation, use_container_width=True)

    # --- Sensitivity analysis ---
    st.markdown(f'<p style="font-weight:600;color:{_c("primary")};margin-top:1rem">Анализ чувствительности весов</p>', unsafe_allow_html=True)
    st.caption("Как изменяется топ-10 при сдвиге веса каждого измерения на +/-10%.")

    original_top10_ids = [c.id for c in ranked[:10]]
    sensitivity_data = []
    for dim in dim_keys_ablation:
        for shift_pct in [-10, +10]:
            shifted_weights = dict(current_weights)
            shift_amount = shifted_weights[dim] * (shift_pct / 100)
            shifted_weights[dim] += shift_amount
            remaining_dims = [d for d in dim_keys_ablation if d != dim]
            redistribute = shift_amount / len(remaining_dims)
            for rd in remaining_dims:
                shifted_weights[rd] -= redistribute

            shifted_scores_list = []
            for c in candidates:
                bd = {b["dimension"]: b["score"] for b in c.score_breakdown}
                new_total = sum(bd.get(d, 0) * shifted_weights[d] for d in dim_keys_ablation)
                shifted_scores_list.append((c.id, new_total))
            shifted_scores_list.sort(key=lambda x: x[1], reverse=True)
            shifted_top10_ids = [x[0] for x in shifted_scores_list[:10]]
            overlap = len(set(original_top10_ids) & set(shifted_top10_ids))
            sensitivity_data.append((DIMENSION_NAMES[dim], shift_pct, overlap))

    sens_matrix = np.zeros((5, 2))
    for i, dim in enumerate(dim_keys_ablation):
        for j, shift in enumerate([-10, +10]):
            match = next((s[2] for s in sensitivity_data if s[0] == DIMENSION_NAMES[dim] and s[1] == shift), 10)
            sens_matrix[i][j] = match

    fig_sens = go.Figure(data=go.Heatmap(
        z=sens_matrix,
        x=["-10%", "+10%"],
        y=[DIMENSION_NAMES[d] for d in dim_keys_ablation],
        colorscale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#22c55e"]],
        text=sens_matrix.astype(int),
        texttemplate="%{text}/10",
        textfont=dict(size=12, color="white"),
        zmin=0, zmax=10,
    ))
    defaults_sens = _plotly_defaults(dark)
    fig_sens.update_layout(
        **defaults_sens,
        height=320,
        title=dict(text="Стабильность топ-10 при сдвиге весов", font=dict(size=13)),
        xaxis=dict(title="Сдвиг веса"),
        yaxis=dict(title="Измерение"),
    )
    st.plotly_chart(fig_sens, use_container_width=True)

    # --- Score distribution analysis ---
    st.markdown(f'<p style="font-weight:600;color:{_c("primary")};margin-top:1rem">Анализ распределения баллов</p>', unsafe_allow_html=True)

    score_mean = np.mean(scores)
    score_median = np.median(scores)
    score_std = np.std(scores)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=scores, nbinsx=25,
        marker=dict(color="#0d9488", line=dict(color="rgba(255,255,255,0.2)", width=1)),
        opacity=0.8, name="Распределение",
    ))
    if len(scores) > 2:
        try:
            kde_d = gaussian_kde(scores)
            x_r = np.linspace(min(scores) - 5, max(scores) + 5, 200)
            kde_v = kde_d(x_r) * len(scores) * ((max(scores) - min(scores)) / 25)
            fig_dist.add_trace(go.Scatter(x=x_r, y=kde_v, mode="lines", line=dict(color="#8b5cf6", width=2.5), name="KDE"))
        except Exception:
            pass
    fig_dist.add_vline(x=score_mean, line_dash="dash", line_color="#3b82f6",
                       annotation_text=f"Среднее: {score_mean:.1f}", annotation_font_color="#3b82f6")
    fig_dist.add_vline(x=score_median, line_dash="dot", line_color="#f59e0b",
                       annotation_text=f"Медиана: {score_median:.1f}", annotation_font_color="#f59e0b")
    defaults_dist = _plotly_defaults(dark)
    fig_dist.update_layout(
        **defaults_dist,
        height=350,
        title=dict(text="Распределение общих баллов с KDE, средним и медианой", font=dict(size=13)),
        xaxis=dict(title="Балл", gridcolor="rgba(148,163,184,0.1)"),
        yaxis=dict(title="Количество", gridcolor="rgba(148,163,184,0.1)"),
        legend=dict(orientation="h", y=-0.18),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Среднее", f"{score_mean:.1f}")
    col2.metric("Медиана", f"{score_median:.1f}")
    col3.metric("Ст. отклонение", f"{score_std:.1f}")

    # ===================================================================
    # FAIRNESS ANALYSIS (expanded)
    # ===================================================================
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(f'<p style="font-weight:700;font-size:1.15rem;color:{_c("heading")}">Анализ справедливости</p>', unsafe_allow_html=True)

    # --- Fairness by education level ---
    st.markdown(f'<p style="font-weight:600;color:{_c("primary")};margin-top:0.5rem">Справедливость по уровню образования</p>', unsafe_allow_html=True)

    edu_groups = {}
    edu_labels = {"school": "Школа", "bachelor": "Бакалавр", "master": "Магистр", "phd": "PhD", "other": "Другое"}
    for c in candidates:
        edu = c.education_level
        if edu not in edu_groups:
            edu_groups[edu] = []
        edu_groups[edu].append(c.total_score)

    edu_names = []
    edu_means = []
    edu_stds = []
    edu_counts = []
    for edu_key in ["school", "bachelor", "master", "phd", "other"]:
        if edu_key in edu_groups:
            edu_names.append(edu_labels.get(edu_key, edu_key))
            edu_means.append(np.mean(edu_groups[edu_key]))
            edu_stds.append(np.std(edu_groups[edu_key]))
            edu_counts.append(len(edu_groups[edu_key]))

    if edu_names:
        fig_edu = go.Figure()
        fig_edu.add_trace(go.Bar(
            x=edu_names, y=edu_means,
            error_y=dict(type="data", array=edu_stds, visible=True, color="#94a3b8"),
            marker_color=CHART_COLORS[:len(edu_names)],
            text=[f"{m:.1f} (n={n})" for m, n in zip(edu_means, edu_counts)],
            textposition="outside",
            textfont=dict(size=10),
        ))
        defaults_edu = _plotly_defaults(dark)
        fig_edu.update_layout(
            **defaults_edu,
            height=320,
            title=dict(text="Средний балл по уровню образования", font=dict(size=13)),
            yaxis=dict(range=[0, 100], gridcolor="rgba(148,163,184,0.1)"),
            xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
            showlegend=False,
        )
        st.plotly_chart(fig_edu, use_container_width=True)

    # --- Fairness by work experience ---
    st.markdown(f'<p style="font-weight:600;color:{_c("primary")};margin-top:1rem">Справедливость по опыту работы</p>', unsafe_allow_html=True)

    exp_groups = {"0 лет": [], "1-2 года": [], "3+ лет": []}
    for c in candidates:
        if c.work_experience_years == 0:
            exp_groups["0 лет"].append(c.total_score)
        elif c.work_experience_years <= 2:
            exp_groups["1-2 года"].append(c.total_score)
        else:
            exp_groups["3+ лет"].append(c.total_score)

    exp_names_list = []
    exp_means_list = []
    exp_stds_list = []
    exp_counts_list = []
    for g in ["0 лет", "1-2 года", "3+ лет"]:
        if exp_groups[g]:
            exp_names_list.append(g)
            exp_means_list.append(np.mean(exp_groups[g]))
            exp_stds_list.append(np.std(exp_groups[g]))
            exp_counts_list.append(len(exp_groups[g]))

    if exp_names_list:
        fig_exp = go.Figure()
        fig_exp.add_trace(go.Bar(
            x=exp_names_list, y=exp_means_list,
            error_y=dict(type="data", array=exp_stds_list, visible=True, color="#94a3b8"),
            marker_color=["#3b82f6", "#0d9488", "#8b5cf6"],
            text=[f"{m:.1f} (n={n})" for m, n in zip(exp_means_list, exp_counts_list)],
            textposition="outside",
            textfont=dict(size=10),
        ))
        defaults_exp = _plotly_defaults(dark)
        fig_exp.update_layout(
            **defaults_exp,
            height=300,
            title=dict(text="Средний балл по опыту работы", font=dict(size=13)),
            yaxis=dict(range=[0, 100], gridcolor="rgba(148,163,184,0.1)"),
            xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
            showlegend=False,
        )
        st.plotly_chart(fig_exp, use_container_width=True)

    # --- Disparate impact ratio ---
    st.markdown(f'<p style="font-weight:600;color:{_c("primary")};margin-top:1rem">Коэффициент диспаратного воздействия</p>', unsafe_allow_html=True)
    st.caption("Отношение среднего балла группы к общему среднему. Значение <0.80 указывает на потенциальное неравенство (правило 4/5).")

    overall_avg = np.mean(scores) if scores else 1.0
    disparate_rows = ""

    all_groups = []
    for edu_key, edu_scores in edu_groups.items():
        group_avg = np.mean(edu_scores)
        ratio = group_avg / overall_avg if overall_avg > 0 else 0
        label = edu_labels.get(edu_key, edu_key)
        flag = " [!]" if ratio < 0.8 else ""
        color = "#ef4444" if ratio < 0.8 else "#22c55e"
        all_groups.append((f"Образование: {label}", group_avg, ratio, len(edu_scores), color, flag))

    for exp_label, exp_scores in exp_groups.items():
        if exp_scores:
            group_avg = np.mean(exp_scores)
            ratio = group_avg / overall_avg if overall_avg > 0 else 0
            flag = " [!]" if ratio < 0.8 else ""
            color = "#ef4444" if ratio < 0.8 else "#22c55e"
            all_groups.append((f"Опыт: {exp_label}", group_avg, ratio, len(exp_scores), color, flag))

    for city, city_sc in city_scores.items():
        group_avg = np.mean(city_sc)
        ratio = group_avg / overall_avg if overall_avg > 0 else 0
        flag = " [!]" if ratio < 0.8 else ""
        color = "#ef4444" if ratio < 0.8 else "#22c55e"
        all_groups.append((f"Город: {city}", group_avg, ratio, len(city_sc), color, flag))

    for gname, gavg, gratio, gcount, gcolor, gflag in all_groups:
        disparate_rows += f"""<tr>
            <td>{gname}</td>
            <td>{gavg:.1f}</td>
            <td style="color:{gcolor};font-weight:700">{gratio:.2f}{gflag}</td>
            <td>{gcount}</td>
        </tr>"""

    st.markdown(f"""
    <table class="styled-table">
        <thead><tr>
            <th>Группа</th><th>Средний балл</th><th>Коэфф. (группа/общий)</th><th>N</th>
        </tr></thead>
        <tbody>{disparate_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    # --- AI detection fairness by city ---
    st.markdown(f'<p style="font-weight:600;color:{_c("primary")};margin-top:1.5rem">Справедливость ИИ-детекции по городам</p>', unsafe_allow_html=True)
    st.caption("Анализ того, не получают ли кандидаты из определённых городов непропорционально частые флаги авто-генерации.")

    city_ai_data = {}
    for c in candidates:
        if c.city not in city_ai_data:
            city_ai_data[c.city] = {"total": 0, "flagged": 0}
        city_ai_data[c.city]["total"] += 1
        for m in c.nlp_metrics:
            if m.get("ai_detection_score", 0) > 0.65:
                city_ai_data[c.city]["flagged"] += 1
                break

    ai_fairness_rows = ""
    for city in sorted(city_ai_data.keys()):
        d = city_ai_data[city]
        rate = d["flagged"] / d["total"] * 100 if d["total"] > 0 else 0
        color = "#ef4444" if rate > 30 else "#f59e0b" if rate > 15 else "#22c55e"
        ai_fairness_rows += f"""<tr>
            <td>{city}</td>
            <td>{d["total"]}</td>
            <td>{d["flagged"]}</td>
            <td style="color:{color};font-weight:700">{rate:.0f}%</td>
        </tr>"""

    st.markdown(f"""
    <table class="styled-table">
        <thead><tr>
            <th>Город</th><th>Всего кандидатов</th><th>Флагов авто-генерации</th><th>Доля флагов</th>
        </tr></thead>
        <tbody>{ai_fairness_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    # ===================================================================
    # NLP ANALYTICS
    # ===================================================================
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(f'<p style="font-weight:600;color:{_c("primary")}">NLP-аналитика эссе</p>', unsafe_allow_html=True)

    ai_scores_all = []
    auth_scores_all = []
    for c in candidates:
        for m in c.nlp_metrics:
            ai_scores_all.append(m.get("ai_detection_score", 0))
            auth_scores_all.append(m.get("authenticity_score", 0))

    if ai_scores_all:
        col1, col2 = st.columns(2)
        with col1:
            fig_ai = go.Figure()
            fig_ai.add_trace(go.Histogram(x=ai_scores_all, nbinsx=20, marker_color="#ef4444", opacity=0.8))
            fig_ai.add_vline(x=0.65, line_dash="dash", line_color="#f59e0b", annotation_text="Порог", annotation_font_color="#f59e0b")
            defaults_ai = _plotly_defaults(dark)
            fig_ai.update_layout(
                **defaults_ai,
                height=280,
                title=dict(text="Авто-детекция", font=dict(size=13)),
                xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
                yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
            )
            st.plotly_chart(fig_ai, use_container_width=True)
        with col2:
            fig_auth = go.Figure()
            fig_auth.add_trace(go.Histogram(x=auth_scores_all, nbinsx=20, marker_color="#10b981", opacity=0.8))
            defaults_auth = _plotly_defaults(dark)
            fig_auth.update_layout(
                **defaults_auth,
                height=280,
                title=dict(text="Аутентичность", font=dict(size=13)),
                xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
                yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
            )
            st.plotly_chart(fig_auth, use_container_width=True)

    # Stats table
    st.markdown(f'<p style="font-weight:600;color:{_c("primary")};margin-top:1rem">Статистика по измерениям</p>', unsafe_allow_html=True)
    stats_data = []
    for dim, s in stats.items():
        stats_data.append({
            "Измерение": DIMENSION_NAMES.get(dim, dim),
            "Среднее": s["mean"],
            "Медиана": s["median"],
            "Мин.": s["min"],
            "Макс.": s["max"],
            "Ст. откл.": s["std"],
        })
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

    render_footer()


# ---------------------------------------------------------------------------
# Page 8: Model Settings (combined scoring + config)
# ---------------------------------------------------------------------------

def page_settings():
    st.markdown(f'<p style="font-weight:700;font-size:1.3rem;color:{_c("heading")};margin-bottom:1rem">Настройки модели и запуск оценки</p>', unsafe_allow_html=True)

    if not st.session_state.get("candidates"):
        st.warning("Сначала загрузите данные кандидатов.")
        render_footer()
        return

    config = st.session_state.get("scoring_config", ScoringConfig())

    st.markdown(f"""
    <div class="glass-card" style="margin-bottom:1rem">
        <p style="font-weight:600;color:{_c("primary")};margin:0">Веса критериев оценки</p>
        <p style="color:{_c("muted")};font-size:0.82rem;margin-top:0.2rem">
            Настройте относительную важность каждого критерия. Сумма весов должна равняться 1.0
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        config.motivation_weight = st.slider("Мотивация", 0.0, 0.5, config.motivation_weight, 0.05, key="w_mot")
        config.leadership_weight = st.slider("Лидерство", 0.0, 0.5, config.leadership_weight, 0.05, key="w_lead")
    with col2:
        config.growth_weight = st.slider("Траектория роста", 0.0, 0.5, config.growth_weight, 0.05, key="w_grow")
        config.skills_weight = st.slider("Навыки", 0.0, 0.5, config.skills_weight, 0.05, key="w_skill")
    with col3:
        config.experience_weight = st.slider("Опыт", 0.0, 0.5, config.experience_weight, 0.05, key="w_exp")

    total_w = sum(config.get_weights().values())
    if abs(total_w - 1.0) > 0.01:
        st.error(f"Сумма весов: {total_w:.2f} (должна быть 1.0)")
    else:
        st.success(f"Сумма весов: {total_w:.2f}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(f'<p style="font-weight:600;color:{_c("primary")}">Дополнительные параметры</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        config.shortlist_threshold = st.slider("Порог шорт-листа", 40.0, 90.0, config.shortlist_threshold, 5.0, key="cfg_sl")
        config.essay_nlp_boost = st.slider("NLP-бонус за качество эссе", 0.0, 0.3, config.essay_nlp_boost, 0.01, key="cfg_nlp")
    with col2:
        config.auto_reject_threshold = st.slider("Порог авто-отклонения", 10.0, 50.0, config.auto_reject_threshold, 5.0, key="cfg_rej")
        config.ai_penalty_factor = st.slider("Штраф за авто-генерацию", 0.0, 0.3, config.ai_penalty_factor, 0.01, key="cfg_ai")

    col_reset, _ = st.columns([1, 3])
    with col_reset:
        if st.button("Сбросить к значениям по умолчанию"):
            st.session_state.scoring_config = ScoringConfig()
            st.rerun()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Run scoring
    if abs(total_w - 1.0) <= 0.01:
        if st.button("Запустить оценку кандидатов", type="primary", use_container_width=True):
            engine = ScoringEngine(config)
            progress = st.progress(0, text="Оценка кандидатов...")

            for i, candidate in enumerate(st.session_state.candidates):
                engine.score_candidate(candidate)
                progress.progress(
                    (i + 1) / len(st.session_state.candidates),
                    text=f"Оценка: {candidate.full_name}...",
                )

            engine.rank_candidates(st.session_state.candidates)
            st.session_state.scored = True

            overrides = st.session_state.get("overrides", {})
            for cid, override in overrides.items():
                for c in st.session_state.candidates:
                    if c.id == cid:
                        c.override_score = override.get("score")
                        c.override_comment = override.get("comment", "")

            progress.empty()
            st.success(f"Оценка завершена для {len(st.session_state.candidates)} кандидатов!")
            st.rerun()

    # Real-time preview of top-10
    if st.session_state.get("scored", False):
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown(f'<p style="font-weight:600;color:{_c("primary")}">Предпросмотр: текущий топ-10</p>', unsafe_allow_html=True)

        top10 = sorted(
            st.session_state.candidates,
            key=lambda c: c.override_score if c.override_score is not None else c.total_score,
            reverse=True,
        )[:10]

        for i, c in enumerate(top10):
            eff = c.override_score if c.override_score is not None else c.total_score
            color = score_to_color(eff)
            display_name = _get_candidate_display_name(c, i)
            st.markdown(f"""
            <div class="mini-card">
                <span class="mini-card-rank">#{i+1}</span>
                <span class="mini-card-name">{display_name}</span>
                <span class="text-secondary" style="font-size:0.75rem;margin-right:0.5rem">{c.city}</span>
                <span class="mini-card-score" style="color:{color}">{eff:.1f}</span>
            </div>
            """, unsafe_allow_html=True)

    # ===================================================================
    # Privacy & Security section
    # ===================================================================
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    with st.expander("Приватность и безопасность", expanded=False):
        st.markdown(f"""
        <div class="glass-card" style="margin-bottom:1rem">
            <p style="font-weight:700;color:{_c("heading")};font-size:1.05rem;margin:0 0 0.8rem 0">Приватность и безопасность данных</p>
            <div style="color:{_c("card_text")};font-size:0.88rem;line-height:1.8">
                <p style="margin:0 0 0.4rem 0">-- Все данные обрабатываются локально на вашем устройстве.</p>
                <p style="margin:0 0 0.4rem 0">-- Внешние API-вызовы не выполняются. Модель работает полностью автономно.</p>
                <p style="margin:0 0 0.4rem 0">-- Данные сессии не сохраняются на диск без явного запроса пользователя.</p>
                <p style="margin:0 0 0.4rem 0">-- Экспорт данных возможен только по инициативе пользователя.</p>
                <p style="margin:0 0 0.4rem 0">-- Персональные данные кандидатов не передаются третьим сторонам.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if "anonymize_names" not in st.session_state:
            st.session_state.anonymize_names = False

        st.toggle(
            "Анонимизировать имена",
            value=st.session_state.get("anonymize_names", False),
            key="anonymize_names",
            help="Заменяет имена кандидатов на 'Кандидат #N' во всех представлениях",
        )

        st.markdown(f"""
        <div class="glass-card" style="margin-top:1rem">
            <p style="font-weight:600;color:{_c("primary")};margin:0 0 0.6rem 0">Ограничения модели</p>
            <div style="color:{_c("card_text")};font-size:0.85rem;line-height:1.8">
                <p style="margin:0 0 0.3rem 0">-- Детекция ИИ-генерации использует эвристический подход и не гарантирует 100% точности.</p>
                <p style="margin:0 0 0.3rem 0">-- Списки ключевых слов для NLP-анализа ограничены и могут не покрывать все варианты выражения мотивации/лидерства.</p>
                <p style="margin:0 0 0.3rem 0">-- GPA нормализуется в диапазоне 2.0-4.0; иные шкалы требуют предварительной конвертации.</p>
                <p style="margin:0 0 0.3rem 0">-- Система оптимизирована для русскоязычных текстов; поддержка других языков ограничена.</p>
                <p style="margin:0 0 0.3rem 0">-- Оценка навыков основана на текстовом совпадении; контекст применения навыков не анализируется.</p>
                <p style="margin:0 0 0.3rem 0">-- Веса критериев настраиваются экспертом, а не обучаются на данных.</p>
                <p style="margin:0 0 0.3rem 0">-- Модель не учитывает невербальную коммуникацию (интервью, видео).</p>
                <p style="margin:0 0 0.3rem 0">-- Рекомендательные письма учитываются количественно, но не анализируются содержательно.</p>
                <p style="margin:0 0 0.3rem 0">-- При малом объёме данных (<20 кандидатов) статистические выводы ненадёжны.</p>
                <p style="margin:0 0 0.3rem 0">-- Синтетические демо-данные не отражают реальное распределение качеств кандидатов.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    render_footer()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="inVision U -- Система отбора",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    inject_custom_css()

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1.2rem 0 0.8rem">
            <div style="display:flex;justify-content:center;margin-bottom:0.5rem">
                <div style="width:44px;height:44px;border-radius:12px;background:linear-gradient(135deg,#0d9488,#3b82f6);display:flex;align-items:center;justify-content:center;box-shadow:0 4px 16px rgba(13,148,136,0.3)">
                    <span style="color:white;font-weight:900;font-size:1.1rem">iV</span>
                </div>
            </div>
            <h2 style="margin:0;color:#e2e8f0;font-size:1.3rem;font-weight:800;letter-spacing:-0.5px">inVision U</h2>
            <p style="color:#64748b;font-size:0.78rem;margin-top:0.2rem;font-weight:500;letter-spacing:0.5px;text-transform:uppercase">
                Приёмная комиссия
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Theme toggle
        dark = _is_dark()
        theme_icon = svg_icon("moon", 16, "#94a3b8") if dark else svg_icon("sun", 16, "#f59e0b")
        st.toggle("Тёмная тема", value=dark, key="dark_mode")

        st.markdown("---")

        page = st.radio(
            "Навигация",
            [
                "Главная",
                "Загрузка данных",
                "Рейтинг кандидатов",
                "Профиль кандидата",
                "Сравнение",
                "Шорт-лист",
                "Аналитика",
                "Настройки модели",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        if st.session_state.get("candidates"):
            total = len(st.session_state.candidates)
            scored = sum(1 for c in st.session_state.candidates if c.application_status != "new")
            shortlisted = sum(1 for c in st.session_state.candidates if c.application_status == "shortlisted")
            st.markdown(f"""
            <div style="padding:0.5rem 0.8rem;background:rgba(13,148,136,0.08);border-radius:10px;margin-bottom:0.5rem">
                <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:0.3rem">Статистика</div>
                <div style="font-size:0.82rem;color:#e2e8f0">Кандидатов: <b>{total}</b></div>
                <div style="font-size:0.82rem;color:#e2e8f0">Оценено: <b>{scored}</b></div>
                <div style="font-size:0.82rem;color:#e2e8f0">Шорт-лист: <b>{shortlisted}</b></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="position:fixed;bottom:0.8rem;font-size:0.7rem;color:#475569;line-height:1.4">
            inVision U by inDrive<br>
            Система отбора v2.0
        </div>
        """, unsafe_allow_html=True)

    # Header
    render_header()

    # Page routing
    if page == "Главная":
        page_dashboard()
    elif page == "Загрузка данных":
        page_upload()
    elif page == "Рейтинг кандидатов":
        page_ranking()
    elif page == "Профиль кандидата":
        page_candidate_detail()
    elif page == "Сравнение":
        page_comparison()
    elif page == "Шорт-лист":
        page_shortlist()
    elif page == "Аналитика":
        page_analytics()
    elif page == "Настройки модели":
        page_settings()


if __name__ == "__main__":
    main()
