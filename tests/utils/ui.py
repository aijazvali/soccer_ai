from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import streamlit as st


def inject_base_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --ink: #e2e8f0;
            --muted: #94a3b8;
            --card: rgba(15, 23, 42, 0.88);
            --border: rgba(148, 163, 184, 0.2);
            --accent: #22d3ee;
            --accent-2: #38bdf8;
            --warn: #f59e0b;
            --bg-1: #0b1220;
            --bg-2: #0f172a;
            --bg-3: #111827;

            /* Override Streamlit theme variables for readability */
            --primary-color: var(--accent);
            --background-color: var(--bg-1);
            --secondary-background-color: #0f172a;
            --text-color: var(--ink);
            color-scheme: dark;
        }

        html, body, [class*="css"]  {
            font-family: "Manrope", "Segoe UI", sans-serif;
            color: var(--ink);
        }

        [data-testid="stAppViewContainer"] {
            color: var(--ink);
        }

        [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
            border-right: 1px solid var(--border);
            box-shadow: 6px 0 24px rgba(0, 0, 0, 0.3);
        }

        [data-testid="stSidebar"] * {
            color: var(--ink);
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(34, 211, 238, 0.18), transparent 45%),
                radial-gradient(circle at 85% 15%, rgba(56, 189, 248, 0.16), transparent 40%),
                linear-gradient(180deg, var(--bg-1), var(--bg-3));
        }

        .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 24px;
            padding: 20px 24px;
            background: linear-gradient(120deg, #0b1220, #0f172a 45%, #0c4a6e);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 22px;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
            margin-bottom: 24px;
        }

        .app-title {
            font-family: "Space Grotesk", "Manrope", sans-serif;
            font-size: 30px;
            font-weight: 600;
            margin-bottom: 6px;
            color: #f8fafc;
        }

        .app-subtitle {
            color: rgba(248, 250, 252, 0.75);
            font-size: 14px;
        }

        .badge-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .badge {
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            background: rgba(148, 163, 184, 0.18);
            color: #e2e8f0;
            border: 1px solid rgba(148, 163, 184, 0.3);
        }

        .badge.good {
            background: rgba(34, 211, 238, 0.2);
            color: #a5f3fc;
            border-color: rgba(34, 211, 238, 0.4);
        }

        .badge.warn {
            background: rgba(251, 146, 60, 0.2);
            color: #fed7aa;
            border-color: rgba(251, 146, 60, 0.35);
        }

        .badge.neutral {
            background: rgba(148, 163, 184, 0.16);
            color: #e2e8f0;
            border-color: rgba(148, 163, 184, 0.3);
        }

        div[data-testid="stVerticalBlock"]:has(.card-marker) {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 20px 22px;
            margin-bottom: 20px;
            box-shadow: 0 18px 36px rgba(15, 23, 42, 0.12);
            position: relative;
            overflow: hidden;
        }

        div[data-testid="stVerticalBlock"]:has(.card-marker)::before {
            content: "";
            position: absolute;
            left: 0;
            top: 16px;
            bottom: 16px;
            width: 4px;
            background: linear-gradient(180deg, var(--accent), var(--accent-2));
            border-radius: 999px;
        }

        .card-marker {
            display: none;
        }

        .section-title {
            font-family: "Space Grotesk", "Manrope", sans-serif;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 6px;
            margin-left: 6px;
        }

        .section-subtitle {
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 12px;
            margin-left: 6px;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }

        .pill {
            padding: 5px 12px;
            border-radius: 999px;
            background: rgba(56, 189, 248, 0.18);
            color: #e0f2fe;
            font-size: 12px;
            font-weight: 600;
        }

        .muted {
            color: var(--muted);
        }

        div.stButton > button {
            border-radius: 12px;
            padding: 0.7rem 1.4rem;
            border: 1px solid rgba(15, 23, 42, 0.14);
            font-weight: 600;
            background: linear-gradient(120deg, #0ea5a4, #38bdf8);
            color: #0b1220;
        }

        div.stButton > button:disabled {
            background: #1f2937;
            color: #64748b;
            border-color: transparent;
        }

        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.75);
            padding: 12px 14px;
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            box-shadow: inset 0 1px 0 rgba(148, 163, 184, 0.15);
        }

        div[data-testid="stMetric"] label {
            color: var(--muted);
        }

        div[data-testid="stFileUploader"] {
            border-radius: 14px;
            border: 1px dashed rgba(148, 163, 184, 0.45);
            padding: 8px;
            background: rgba(15, 23, 42, 0.6);
        }

        div[data-testid="stAlert"] {
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.25);
            background: rgba(15, 23, 42, 0.7);
            box-shadow: 0 8px 18px rgba(0, 0, 0, 0.3);
        }

        .tooltip-row {
            margin-top: 10px;
        }

        .tooltip-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(56, 189, 248, 0.2);
            color: #e2e8f0;
            font-size: 12px;
            font-weight: 600;
            border: 1px solid rgba(56, 189, 248, 0.4);
            cursor: help;
        }

        details > summary {
            font-family: "Space Grotesk", "Manrope", sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _badge(label: str, tone: str) -> str:
    return f"<span class=\"badge {tone}\">{label}</span>"


def render_header(title: str, subtitle: str, selected_test: str, has_video: bool, ready: bool) -> None:
    badges = [
        _badge(f"Test: {selected_test}", "neutral"),
        _badge("Video Loaded" if has_video else "No Video", "good" if has_video else "warn"),
        _badge("Ready to Analyze" if ready else "Not Ready", "good" if ready else "warn"),
    ]
    badge_html = "".join(badges)

    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <div class="app-title">{title}</div>
                <div class="app-subtitle">{subtitle}</div>
            </div>
            <div class="badge-row">{badge_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def card(title: str, subtitle: str | None = None):
    container = st.container()
    with container:
        st.markdown('<div class="card-marker"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
        if subtitle:
            st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)
        yield


def render_pills(labels: Iterable[str]) -> None:
    items = "".join(f"<span class=\"pill\">{label}</span>" for label in labels)
    st.markdown(f'<div class="pill-row">{items}</div>', unsafe_allow_html=True)
