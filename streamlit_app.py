import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import pandas as pd
from PIL import Image
import streamlit as st

import soccer_ai.config as cfg
from soccer_ai.calibration import build_calibration, save_calibration
from soccer_ai.options import TouchOptions
from soccer_ai.pipelines import run_touch_detection


st.set_page_config(
    page_title="Soccer AI - Performance Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Design System CSS
st.markdown(
    """
    <style>
    /* ===== GOOGLE FONTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* ===== ROOT VARIABLES ===== */
    :root {
        --bg-primary: #0a0f1a;
        --bg-secondary: #0d1321;
        --bg-card: rgba(255, 255, 255, 0.03);
        --bg-card-hover: rgba(255, 255, 255, 0.06);
        --border-subtle: rgba(255, 255, 255, 0.06);
        --border-accent: rgba(34, 211, 238, 0.3);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-cyan: #22d3ee;
        --accent-indigo: #818cf8;
        --accent-purple: #a78bfa;
        --gradient-primary: linear-gradient(135deg, #22d3ee 0%, #6366f1 50%, #a78bfa 100%);
        --gradient-button: linear-gradient(135deg, #22d3ee 0%, #6366f1 100%);
        --shadow-glow: 0 0 40px rgba(34, 211, 238, 0.15);
        --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.3);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
    }
    
    /* ===== GLOBAL STYLES ===== */
    .stApp {
        background: linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .block-container {
        padding: 2rem 3rem 3rem 3rem;
        max-width: 1400px;
    }
    
    /* ===== HEADER BRANDING ===== */
    .header-container {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-xl);
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-primary);
    }
    
    .header-title {
        font-size: 2rem;
        font-weight: 800;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        display: inline-flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .header-subtitle {
        color: var(--text-secondary);
        font-size: 1rem;
        font-weight: 400;
        margin-top: 0.25rem;
    }
    
    .version-badge {
        display: inline-block;
        background: var(--bg-card-hover);
        color: var(--accent-cyan);
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        border: 1px solid var(--border-accent);
        margin-left: 1rem;
    }
    
    /* ===== SECTION CONTAINERS ===== */
    .section-container {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .section-container:hover {
        border-color: var(--border-accent);
        box-shadow: var(--shadow-glow);
    }
    
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border-subtle);
    }
    
    .section-icon {
        font-size: 1.25rem;
    }
    
    /* ===== METRIC CARDS GRID ===== */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 1.25rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
        background: var(--gradient-primary);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: var(--border-accent);
        box-shadow: 0 12px 40px rgba(34, 211, 238, 0.15);
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
    }
    
    .metric-value-highlight {
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-subvalue {
        font-size: 0.85rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: var(--gradient-button) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: var(--radius-md) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.35) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px rgba(99, 102, 241, 0.45) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* ===== DOWNLOAD BUTTON ===== */
    .stDownloadButton > button {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-accent) !important;
        color: var(--accent-cyan) !important;
        transition: all 0.3s ease !important;
    }
    
    .stDownloadButton > button:hover {
        background: rgba(34, 211, 238, 0.1) !important;
        box-shadow: 0 4px 16px rgba(34, 211, 238, 0.2) !important;
    }
    
    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div > div {
        background: var(--gradient-primary) !important;
        border-radius: 999px !important;
    }
    
    .stProgress > div > div {
        background: var(--bg-card) !important;
        border-radius: 999px !important;
    }
    
    /* ===== FILE UPLOADER ===== */
    .stFileUploader > div {
        background: var(--bg-card) !important;
        border: 2px dashed var(--border-subtle) !important;
        border-radius: var(--radius-lg) !important;
        padding: 2rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--accent-cyan) !important;
        background: rgba(34, 211, 238, 0.03) !important;
    }
    
    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1525 0%, #0a1018 100%) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    
    [data-testid="stSidebar"] .block-container {
        padding: 1.5rem 1rem !important;
    }
    
    .sidebar-section {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .sidebar-section-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--accent-cyan);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
    }
    
    /* ===== SELECTBOX & INPUTS ===== */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
    }
    
    .stSelectbox > div > div:hover,
    .stNumberInput > div > div > input:hover,
    .stTextInput > div > div > input:hover {
        border-color: var(--accent-cyan) !important;
    }
    
    /* ===== CHECKBOX ===== */
    .stCheckbox > label > span {
        color: var(--text-primary) !important;
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card) !important;
        border-radius: var(--radius-md) !important;
        padding: 0.25rem !important;
        gap: 0.25rem !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--gradient-button) !important;
        color: white !important;
    }
    
    /* ===== DATAFRAME ===== */
    .stDataFrame {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
    }
    
    /* ===== SUCCESS/ERROR/WARNING ===== */
    .stSuccess {
        background: rgba(34, 197, 94, 0.1) !important;
        border: 1px solid rgba(34, 197, 94, 0.3) !important;
        border-radius: var(--radius-md) !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: var(--radius-md) !important;
    }
    
    .stWarning {
        background: rgba(251, 191, 36, 0.1) !important;
        border: 1px solid rgba(251, 191, 36, 0.3) !important;
        border-radius: var(--radius-md) !important;
    }
    
    /* ===== SLIDER ===== */
    .stSlider > div > div > div {
        background: var(--gradient-primary) !important;
    }
    
    /* ===== SPINNER ===== */
    .stSpinner > div {
        border-top-color: var(--accent-cyan) !important;
    }
    
    /* ===== CHARTS ===== */
    .stPlotlyChart, .stLineChart, .stAreaChart {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-lg) !important;
        padding: 1rem !important;
    }
    
    /* ===== VIDEO PREVIEW ===== */
    .video-preview-container {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 1rem;
        position: relative;
    }
    
    .video-preview-container img {
        border-radius: var(--radius-md);
    }
    
    /* ===== STATUS INDICATOR ===== */
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 0.5rem;
        animation: pulse 2s infinite;
    }
    
    .status-dot.active {
        background: #22c55e;
        box-shadow: 0 0 8px rgba(34, 197, 94, 0.5);
    }
    
    .status-dot.processing {
        background: var(--accent-cyan);
        box-shadow: 0 0 8px rgba(34, 211, 238, 0.5);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* ===== HIDE STREAMLIT BRANDING ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ===== CUSTOM SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-subtle);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _save_upload(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[-1] or ".mp4"
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def _video_frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    return total


def _video_fps(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else 30.0


def _read_video_frame(path: str, frame_idx: int):
    cap = cv2.VideoCapture(path)
    if frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return frame


def _prepare_canvas_frame(frame_bgr, max_width: int = 960):
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_width / max(1, w))
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame_bgr, (new_w, new_h)) if scale < 1.0 else frame_bgr
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb), scale


def _canvas_initial_drawing(points, radius=6):
    objects = []
    for x, y in points:
        objects.append(
            {
                "type": "circle",
                "left": float(x - radius),
                "top": float(y - radius),
                "radius": float(radius),
                "fill": "rgba(255, 196, 0, 0.6)",
                "stroke": "rgba(255, 196, 0, 0.9)",
                "strokeWidth": 2,
            }
        )
    return {"version": "4.4.0", "objects": objects}


def _extract_canvas_points(canvas_json):
    points = []
    if not canvas_json:
        return points
    for obj in canvas_json.get("objects", []):
        x = obj.get("x")
        y = obj.get("y")
        if x is not None and y is not None:
            points.append((float(x), float(y)))
            continue
        left = float(obj.get("left", 0.0))
        top = float(obj.get("top", 0.0))
        radius = float(obj.get("radius", 0.0))
        scale_x = float(obj.get("scaleX", 1.0))
        scale_y = float(obj.get("scaleY", 1.0))
        points.append((left + radius * scale_x, top + radius * scale_y))
    return points


def _ensure_canvas_compat():
    try:
        import streamlit.elements.image as st_image
        if hasattr(st_image, "image_to_url"):
            return
        from streamlit.elements.lib import image_utils, layout_utils

        def _image_to_url_compat(image, width, clamp, channels, output_format, image_id):
            layout_config = layout_utils.LayoutConfig(width=width)
            return image_utils.image_to_url(
                image, layout_config, clamp, channels, output_format, image_id
            )

        st_image.image_to_url = _image_to_url_compat
    except Exception:
        return


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "--"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    if minutes < 60:
        return f"{minutes:d}m {rem:04.1f}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours:d}h {minutes:02d}m"


def _format_distance(meters: Optional[float], use_metric: bool = True) -> str:
    if meters is None or meters < 0:
        return "--"
    if use_metric:
        # Display in meters
        if meters >= 1000:
            return f"{meters:.0f} m"
        return f"{meters:.1f} m"
    else:
        # Display in km
        if meters >= 1000:
            return f"{meters / 1000:.2f} km"
        return f"{meters:.0f} m"


def _format_speed(speed_kmh: Optional[float], use_metric: bool = True) -> str:
    """Format speed with unit toggle. Input is always in km/h."""
    if speed_kmh is None:
        return "--"
    if use_metric:
        # Convert to m/s
        speed_mps = speed_kmh / 3.6
        return f"{speed_mps:.1f} m/s"
    else:
        return f"{speed_kmh:.1f} km/h"


def _format_speed_mps(speed_mps: Optional[float], use_metric: bool = True) -> str:
    """Format speed with unit toggle. Input is in m/s."""
    if speed_mps is None:
        return "--"
    if use_metric:
        return f"{speed_mps:.1f} m/s"
    else:
        speed_kmh = speed_mps * 3.6
        return f"{speed_kmh:.1f} km/h"


def _format_accel(accel: Optional[float]) -> str:
    if accel is None:
        return "--"
    return f"{accel:.2f} m/s²"


def _draw_stats_overlay(
    frame_bgr,
    lines: List[str],
    header: Optional[str] = None,
    anchor: tuple[int, int] = (16, 16),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    header_scale = 0.7
    thickness = 2
    header_thickness = 2
    padding = 10
    line_gap = 6

    items = []
    if header:
        items.append((header, header_scale, header_thickness))
    for line in lines:
        items.append((line, font_scale, thickness))

    max_width = 0
    total_height = padding
    metrics = []
    for text, scale, thick in items:
        (w, h), base = cv2.getTextSize(text, font, scale, thick)
        max_width = max(max_width, w)
        total_height += h + base + line_gap
        metrics.append((text, scale, thick, h, base))
    total_height += padding - line_gap
    box_width = max_width + padding * 2
    box_height = max(0, total_height)
    x, y = anchor
    y = max(0, y)
    x = max(0, x)

    cv2.rectangle(
        frame_bgr,
        (x, y),
        (x + box_width, y + box_height),
        (0, 0, 0),
        -1,
    )

    cursor_y = y + padding
    text_x = x + padding
    for text, scale, thick, height, base in metrics:
        cursor_y += height
        cv2.putText(
            frame_bgr,
            text,
            (text_x, cursor_y),
            font,
            scale,
            (255, 255, 255),
            thick,
            cv2.LINE_AA,
        )
        cursor_y += base + line_gap


def _draw_event_overlay(frame_bgr, event: Optional[Dict[str, Any]]) -> None:
    if not event:
        return
    label = str(event.get("type") or "event").upper()
    power_val = event.get("power")
    if power_val is not None:
        label = f"{label} {power_val:.0f}"
    pos = event.get("pos")
    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
        pos = (40.0, 80.0)
    x = int(pos[0])
    y = max(24, int(pos[1]) - 10)
    color_map = {
        "SHOT": (0, 0, 255),
        "PASS": (0, 255, 0),
        "DRIBBLE": (255, 215, 0),
    }
    cv2.putText(
        frame_bgr,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color_map.get(label.split()[0], (0, 255, 255)),
        3,
        cv2.LINE_AA,
    )


def _draw_player_overlays(
    frame_bgr,
    players: List[Dict[str, Any]],
    show_ids: bool = True,
    show_feet: bool = True,
    show_speed: bool = True,
    use_metric_display: bool = True,
) -> None:
    if not players:
        return
    h = frame_bgr.shape[0]
    for player in players:
        bbox = player.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if show_ids:
            pid = player.get("id", "?")
            cv2.putText(
                frame_bgr,
                f"ID {pid}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        if show_feet:
            left = player.get("left")
            right = player.get("right")
            if left is not None:
                cv2.circle(
                    frame_bgr,
                    (int(left[0]), int(left[1])),
                    5,
                    (255, 0, 0),
                    -1,
                )
            if right is not None:
                cv2.circle(
                    frame_bgr,
                    (int(right[0]), int(right[1])),
                    5,
                    (0, 0, 255),
                    -1,
                )
        if show_speed:
            speed_kmh = player.get("speed_kmh")
            if speed_kmh is not None:
                speed_text = _format_speed(speed_kmh, use_metric_display)
                speed_y = min(h - 8, y2 + 24)
                cv2.putText(
                    frame_bgr,
                    speed_text,
                    (x1, speed_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )


def _draw_ball_overlay(
    frame_bgr,
    ball: Optional[Dict[str, Any]],
    show_vector: bool,
    show_speed: bool,
    use_homography: bool,
    vector_scale: float,
) -> None:
    if not ball:
        return
    center = ball.get("center")
    radius = ball.get("radius")
    if center is None or radius is None:
        return
    start_pt = (int(center[0]), int(center[1]))
    cv2.circle(
        frame_bgr,
        start_pt,
        int(radius),
        (0, 165, 255),
        2,
    )
    vel_draw = ball.get("vel_draw")
    if show_vector and vel_draw is not None:
        end_pt = (
            int(center[0] + vel_draw[0] * vector_scale),
            int(center[1] + vel_draw[1] * vector_scale),
        )
        cv2.arrowedLine(
            frame_bgr,
            start_pt,
            end_pt,
            (0, 255, 255),
            2,
            tipLength=0.3,
        )
        if show_speed:
            display_speed = ball.get("speed_mps") if use_homography else ball.get("speed_draw")
            speed_unit = "m/s" if use_homography else "px/s"
            if display_speed is not None:
                cv2.putText(
                    frame_bgr,
                    f"{display_speed:.1f}{speed_unit}",
                    (start_pt[0] + 6, start_pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )


def _draw_ball_trail_overlay(
    frame_bgr,
    trail_points: List[Tuple[float, float]],
    color: Tuple[int, int, int] = (0, 165, 255),
    max_thickness: int = 6,
) -> None:
    if len(trail_points) < 2:
        return
    max_thickness = max(1, int(max_thickness))
    denom = max(1, len(trail_points) - 1)
    for idx in range(1, len(trail_points)):
        pt1 = trail_points[idx - 1]
        pt2 = trail_points[idx]
        t = idx / denom
        intensity = 0.2 + 0.8 * t
        thickness = max(1, int(round(1 + (max_thickness - 1) * t)))
        color_scaled = (
            int(color[0] * intensity),
            int(color[1] * intensity),
            int(color[2] * intensity),
        )
        cv2.line(
            frame_bgr,
            (int(pt1[0]), int(pt1[1])),
            (int(pt2[0]), int(pt2[1])),
            color_scaled,
            thickness,
            lineType=cv2.LINE_AA,
        )


def _collect_ball_trail(
    frame_lookup: Dict[int, Dict[str, Any]],
    frame_idx: int,
    max_len: int,
    max_gap_frames: int,
) -> List[Tuple[float, float]]:
    if max_len <= 1:
        return []
    keys = [k for k in frame_lookup.keys() if k <= frame_idx]
    keys.sort(reverse=True)
    points = []
    last_frame = None
    for k in keys:
        meta = frame_lookup[k].get("meta") if frame_lookup.get(k) else None
        ball = meta.get("ball") if meta else None
        center = ball.get("center") if ball else None
        if center is None:
            continue
        if last_frame is not None and last_frame - k > max_gap_frames:
            break
        points.append((float(center[0]), float(center[1])))
        last_frame = k
        if len(points) >= max_len:
            break
    return list(reversed(points))


def _init_video_writer(path: str, frame_bgr, fps: float) -> Optional[cv2.VideoWriter]:
    height, width = frame_bgr.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        return None
    return writer


def _list_model_options():
    """Collect available model files from the models directory."""
    det_files = []
    pose_files = []
    if cfg.MODELS_DIR.exists():
        for path in cfg.MODELS_DIR.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".pt", ".pth"}:
                continue
            stem = path.stem.lower()
            if "pose" in stem:
                pose_files.append(path.name)
            else:
                det_files.append(path.name)

    default_det = Path(cfg.DETECTOR_WEIGHTS).name
    default_pose = Path(cfg.POSE_WEIGHTS).name
    if default_det not in det_files:
        det_files.append(default_det)
    if default_pose not in pose_files:
        pose_files.append(default_pose)

    det_files = sorted(set(det_files))
    pose_files = sorted(set(pose_files))
    return det_files, pose_files


def _default_index(options, default_name: str) -> int:
    return options.index(default_name) if default_name in options else 0


REPORT_SECTIONS = [
    "Processing info",
    "Touches",
    "Speed",
    "Jumps",
    "Shots",
    "Time & Distance",
    "Acceleration",
    "Shot log",
    "Speed chart",
    "Snapshots",
]
MAX_REPORT_TOKENS = 100000
TOKEN_CHAR_RATIO = 4
MAX_REPORT_CHARS = MAX_REPORT_TOKENS * TOKEN_CHAR_RATIO
CHAT_HISTORY_LIMIT = 8
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_TEMPERATURE = 0.2
DEFAULT_GEMINI_MAX_OUTPUT_TOKENS = 1024
SNAPSHOT_MAX = 8
SNAPSHOT_WIDTH = 640
SNAPSHOT_JPEG_QUALITY = 90
COACH_SYSTEM_PROMPT = (
    "You are a soccer performance analyst and coach. "
    "Use only the provided report JSON to answer. "
    "If a detail is missing, say you do not know. "
    "Be concise, practical, and specific."
)


def _report_section_list(report_sections) -> List[str]:
    return [section for section in REPORT_SECTIONS if section in report_sections]


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + TOKEN_CHAR_RATIO - 1) // TOKEN_CHAR_RATIO)


def _downsample_list(items: List[dict], max_items: int) -> List[dict]:
    if max_items <= 0:
        return []
    if len(items) <= max_items:
        return list(items)
    step = max(1, len(items) // max_items)
    sampled = items[::step]
    if sampled and sampled[-1] is not items[-1]:
        sampled[-1] = items[-1]
    return sampled[:max_items]


def _trim_report_for_context(report_data: dict, max_chars: int) -> tuple[dict, Optional[dict]]:
    data = dict(report_data)
    truncation: dict = {}

    def _dump(current: dict) -> str:
        return json.dumps(current, indent=2, ensure_ascii=True)

    def _size(current: dict) -> int:
        return len(_dump(current))

    if _size(data) <= max_chars:
        return data, None

    speed_points = data.get("speed_points")
    if isinstance(speed_points, list) and speed_points:
        original_count = len(speed_points)
        max_items = min(original_count, 2000)
        while max_items > 0:
            data["speed_points"] = _downsample_list(speed_points, max_items)
            if _size(data) <= max_chars:
                truncation["speed_points"] = {
                    "original_count": original_count,
                    "kept_count": len(data["speed_points"]),
                    "method": "downsampled",
                }
                break
            max_items //= 2
        if _size(data) > max_chars:
            data["speed_points"] = []
            truncation["speed_points"] = {
                "original_count": original_count,
                "kept_count": 0,
                "method": "removed",
            }

    shot_log = data.get("shot_log")
    if _size(data) > max_chars and isinstance(shot_log, list) and shot_log:
        original_count = len(shot_log)
        max_items = min(original_count, 200)
        while max_items > 0:
            data["shot_log"] = shot_log[-max_items:]
            if _size(data) <= max_chars:
                truncation["shot_log"] = {
                    "original_count": original_count,
                    "kept_count": len(data["shot_log"]),
                    "method": "tail",
                }
                break
            max_items //= 2
        if _size(data) > max_chars:
            data["shot_log"] = []
            truncation["shot_log"] = {
                "original_count": original_count,
                "kept_count": 0,
                "method": "removed",
            }

    if _size(data) > max_chars:
        dropped = []
        for key in (
            "speed_points",
            "snapshots",
            "shot_log",
            "processing",
            "speed",
            "jumps",
            "shots",
            "time_distance",
            "acceleration",
            "touches",
        ):
            if key in data:
                data.pop(key, None)
                dropped.append(key)
                if _size(data) <= max_chars:
                    break
        if dropped:
            truncation["dropped_sections"] = dropped

    if truncation:
        data["truncation"] = {
            "max_tokens": MAX_REPORT_TOKENS,
            "approx_tokens": _estimate_tokens(_dump(data)),
            "details": truncation,
            "note": "Approximate token estimate at 4 chars/token.",
        }
        if _size(data) > max_chars:
            data["truncation"] = {
                "max_tokens": MAX_REPORT_TOKENS,
                "approx_tokens": _estimate_tokens(_dump(data)),
                "note": "Truncation metadata reduced to fit size.",
            }

    return data, truncation or None


def _save_snapshot(image_bgr, path: Path) -> tuple[int, int]:
    if image_bgr is None:
        return 0, 0
    image_to_save = image_bgr
    if SNAPSHOT_WIDTH and image_bgr.shape[1] > SNAPSHOT_WIDTH:
        scale = SNAPSHOT_WIDTH / image_bgr.shape[1]
        new_w = SNAPSHOT_WIDTH
        new_h = max(1, int(image_bgr.shape[0] * scale))
        image_to_save = cv2.resize(image_bgr, (new_w, new_h))
    cv2.imwrite(
        str(path),
        image_to_save,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(SNAPSHOT_JPEG_QUALITY)],
    )
    return image_to_save.shape[1], image_to_save.shape[0]


def _get_gemini_api_key() -> Optional[str]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return api_key
    secrets = getattr(st, "secrets", None)
    if not secrets:
        return None
    try:
        api_key = secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None
    if api_key:
        return api_key
    try:
        return secrets["GEMINI_API_KEY"]
    except Exception:
        return None


def _gemini_prereq_error() -> Optional[str]:
    if not _get_gemini_api_key():
        return "Set GEMINI_API_KEY in your environment or Streamlit secrets to enable chat."
    try:
        from google import genai  # noqa: F401
    except Exception:
        return "Install the google-genai package to enable chat."
    return None


def _build_chat_prompt(report_json: str, history: List[dict], user_message: str) -> str:
    conversation_lines = []
    for msg in history:
        role = msg.get("role")
        prefix = "Assistant" if role == "assistant" else "User"
        content = msg.get("content", "")
        if content:
            conversation_lines.append(f"{prefix}: {content}")
    conversation_block = "\n".join(conversation_lines)
    parts = ["Report JSON:", report_json]
    if conversation_block:
        parts.extend(["Conversation:", conversation_block])
    parts.append(f"User: {user_message}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _generate_gemini_response(
    report_json: str,
    user_message: str,
    history: List[dict],
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> tuple[Optional[str], Optional[str]]:
    api_key = _get_gemini_api_key()
    if not api_key:
        return None, "Missing GEMINI_API_KEY. Set it and restart the app."
    try:
        from google import genai
        from google.genai import types
    except Exception:
        return None, "Missing dependency: google-genai."
    history = history[-CHAT_HISTORY_LIMIT:]
    prompt = _build_chat_prompt(report_json, history, user_message)
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=COACH_SYSTEM_PROMPT,
    )
    try:
        client = st.session_state.get("gemini_client")
        if client is None:
            client = genai.Client(api_key=api_key)
            st.session_state["gemini_client"] = client
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
    except Exception as exc:
        message = f"Gemini request failed: {exc.__class__.__name__}: {exc}"
        if "closed" in str(exc).lower():
            try:
                client = genai.Client(api_key=api_key)
                st.session_state["gemini_client"] = client
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
            except Exception as exc_retry:
                return None, f"Gemini request failed: {exc_retry.__class__.__name__}: {exc_retry}"
        else:
            return None, message
    text = getattr(response, "text", None)
    if text is None:
        return "", None
    return text.strip(), None


def sidebar_options(calibration_default: Optional[str] = None) -> TouchOptions:
    # Sidebar branding
    st.sidebar.markdown(
        """
        <div style="text-align:center;padding:1rem 0 1.5rem 0;border-bottom:1px solid rgba(255,255,255,0.1);margin-bottom:1rem;">
            <div style="font-size:1.5rem;">⚽</div>
            <div style="font-size:0.9rem;font-weight:600;color:#f1f5f9;">Soccer AI</div>
            <div style="font-size:0.7rem;color:#64748b;">Settings</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Models Section
    with st.sidebar.expander("🤖 Models", expanded=True):
        det_options, pose_options = _list_model_options()
        det_weights = st.selectbox(
            "Detection weights",
            options=det_options,
            index=_default_index(det_options, Path(cfg.DETECTOR_WEIGHTS).name),
        )
        pose_weights = st.selectbox(
            "Pose weights",
            options=pose_options,
            index=_default_index(pose_options, Path(cfg.POSE_WEIGHTS).name),
        )
    
    # Visualization Section
    with st.sidebar.expander("🎨 Visualization", expanded=True):
        draw_vector = st.checkbox("Draw ball vector", value=cfg.DRAW_BALL_VECTOR)
        vector_scale = st.slider(
            "Vector scale",
            min_value=4.0,
            max_value=24.0,
            value=float(cfg.BALL_VECTOR_SCALE),
            step=1.0,
        )
        draw_trail = st.checkbox("Show ball trail", value=cfg.DRAW_BALL_TRAIL)
        show_speed = st.checkbox("Show ball speed", value=cfg.SHOW_BALL_SPEED)
        show_player_speed = st.checkbox("Show player speed", value=cfg.SHOW_PLAYER_SPEED)
        show_components = st.checkbox("Show velocity components", value=cfg.SHOW_BALL_COMPONENTS)
    
    # Ground Plane Section
    with st.sidebar.expander("🗺️ Ground Plane", expanded=False):
        draw_extended_ground = st.checkbox(
            "Draw extended ground plane",
            value=cfg.DRAW_EXTENDED_GROUND,
            help="Show the extrapolated ground area beyond the calibration rectangle.",
        )
        extended_ground_multiplier = st.slider(
            "Extended area multiplier",
            min_value=1.0,
            max_value=5.0,
            value=float(cfg.EXTENDED_GROUND_MULTIPLIER),
            step=0.5,
            help="How many times larger than the calibration rectangle to extend.",
        )
        draw_ground_grid = st.checkbox(
            "Draw ground grid",
            value=cfg.DRAW_GROUND_GRID,
            help="Show grid lines on the ground plane to visualize perspective.",
        )
        ground_grid_spacing = st.slider(
            "Grid spacing (meters)",
            min_value=1.0,
            max_value=10.0,
            value=float(cfg.GROUND_GRID_SPACING_M),
            step=0.5,
            help="Distance between grid lines in meters.",
        )
        grid_subdivisions = st.slider(
            "Grid line subdivisions",
            min_value=5,
            max_value=50,
            value=int(cfg.GRID_LINE_SUBDIVISIONS),
            step=5,
            help="Higher values = smoother perspective curves but slower rendering.",
        )
        show_distance_markers = st.checkbox(
            "Show distance markers",
            value=cfg.SHOW_GRID_DISTANCE_MARKERS,
            help="Display field coordinates at grid intersections for accuracy verification.",
        )
    
    # Touch Heuristics Section
    with st.sidebar.expander("⚽ Touch Detection", expanded=False):
        event_touch_enabled = st.checkbox(
            "Use event-touch heuristic", value=cfg.EVENT_TOUCH_ENABLED
        )
        event_touch_dist_ratio = st.slider(
            "Event-touch distance ratio",
            min_value=0.8,
            max_value=2.0,
            value=float(cfg.EVENT_TOUCH_DIST_RATIO),
            step=0.05,
        )
    
    # Calibration Section
    with st.sidebar.expander("📐 Calibration", expanded=False):
        calibration_file = st.file_uploader(
            "Upload calibration (JSON)",
            type=["json"],
            help="Upload the JSON produced by the cone calibration step.",
        )
        calibration_path = None
        if calibration_file is not None:
            tmp_calib = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            tmp_calib.write(calibration_file.read())
            tmp_calib.close()
            calibration_path = tmp_calib.name
        elif calibration_default:
            calibration_path = calibration_default

        if calibration_path:
            st.success(f"✓ {Path(calibration_path).name}")

        use_homography = st.checkbox(
            "Use homography for calculations",
            value=cfg.USE_HOMOGRAPHY,
            help="Toggle between homography-based meters and pixel logic.",
        )
    
    # Performance Section
    with st.sidebar.expander("⚡ Performance", expanded=False):
        display_stride = st.slider(
            "Display stride (emit every Nth frame)", min_value=1, max_value=5, value=1
        )
    
    # Accuracy Settings Section
    with st.sidebar.expander("🎯 Accuracy Settings", expanded=False):
        use_ema = st.checkbox(
            "Use EMA smoothing",
            value=cfg.USE_EMA_SMOOTHING,
            help="Exponential Moving Average is more responsive than median smoothing.",
        )
        ema_alpha = st.slider(
            "EMA smoothing factor",
            min_value=0.1,
            max_value=0.5,
            value=float(cfg.EMA_ALPHA),
            step=0.05,
            help="Lower = smoother but delayed, Higher = more responsive.",
        )
        min_movement = st.slider(
            "Min movement threshold (m)",
            min_value=0.01,
            max_value=0.10,
            value=float(cfg.MIN_MOVEMENT_M),
            step=0.01,
            help="Ignore movement below this threshold to prevent jitter accumulation.",
        )
        max_human_speed = st.slider(
            "Max human speed (m/s)",
            min_value=8.0,
            max_value=15.0,
            value=float(cfg.MAX_HUMAN_SPEED_MPS),
            step=0.5,
            help="Speeds above this are rejected as detection errors.",
        )
        max_gap_frames = st.slider(
            "Max tracking gap (frames)",
            min_value=2,
            max_value=15,
            value=int(cfg.MAX_SPEED_GAP_FRAMES),
            step=1,
            help="Reset tracking if player lost for this many frames.",
        )
    
    # Units Section
    with st.sidebar.expander("📏 Display Units", expanded=False):
        use_metric_display = st.radio(
            "Speed & Distance Units",
            options=["Metric (m/s, meters)", "Imperial (km/h, km)"],
            index=0 if cfg.USE_METRIC_DISPLAY else 1,
            help="Choose how speeds and distances are displayed.",
        ) == "Metric (m/s, meters)"

    return TouchOptions(
        detector_weights=det_weights,
        pose_weights=pose_weights,
        draw_ball_vector=draw_vector,
        ball_vector_scale=vector_scale,
        show_ball_speed=show_speed,
        show_player_speed=show_player_speed,
        show_ball_components=show_components,
        draw_ball_trail=draw_trail,
        event_touch_enabled=event_touch_enabled,
        event_touch_dist_ratio=event_touch_dist_ratio,
        display_stride=display_stride,
        calibration_path=calibration_path,
        use_homography=use_homography,
        draw_extended_ground=draw_extended_ground,
        extended_ground_multiplier=extended_ground_multiplier,
        draw_ground_grid=draw_ground_grid,
        ground_grid_spacing_m=ground_grid_spacing,
        grid_line_subdivisions=grid_subdivisions,
        show_grid_distance_markers=show_distance_markers,
        # Accuracy improvements
        min_movement_m=min_movement,
        max_speed_gap_frames=max_gap_frames,
        max_human_speed_mps=max_human_speed,
        max_ball_speed_mps=cfg.MAX_BALL_SPEED_MPS,
        use_ema_smoothing=use_ema,
        ema_alpha=ema_alpha,
        # Unit system
        use_metric_display=use_metric_display,
    )


def report_options():
    with st.sidebar.expander("📊 Report Options", expanded=False):
        selections = st.multiselect(
            "Include in report",
            options=REPORT_SECTIONS,
            default=REPORT_SECTIONS,
        )
    return set(selections)


def main():
    # Professional Header
    st.markdown(
        """
        <div class="header-container">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 class="header-title">⚽ Soccer AI</h1>
                    <p class="header-subtitle">AI-Powered Performance Analysis & Coaching</p>
                </div>
                <span class="version-badge">v1.0</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Video Input Section
    st.markdown(
        """<div class="section-container">
            <div class="section-header"><span class="section-icon">📤</span> Video Input</div>
        """,
        unsafe_allow_html=True,
    )
    
    col_upload, col_sample = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"], label_visibility="collapsed")
    with col_sample:
        samples = {
            "📹 Test video (main)": Path("bin/test_video.mp4"),
            "📹 Test video (alt)": Path("bin/test_vide.mp4"),
            "🚫 None": None,
        }
        sample_choice = st.selectbox("Or select a sample", options=list(samples.keys()), index=0, label_visibility="collapsed")
    
    
    st.markdown("</div>", unsafe_allow_html=True)

    video_path = None
    if uploaded is not None:
        upload_id = getattr(uploaded, "file_id", None)
        if not upload_id:
            upload_id = f"{uploaded.name}:{uploaded.size}:{uploaded.type}"
        cached_path = st.session_state.get("uploaded_video_path")
        if st.session_state.get("uploaded_video_id") != upload_id or not cached_path:
            st.session_state["uploaded_video_id"] = upload_id
            st.session_state["uploaded_video_path"] = _save_upload(uploaded)
        video_path = st.session_state.get("uploaded_video_path")
    elif samples[sample_choice] is not None:
        candidate = Path(os.getcwd()) / samples[sample_choice]
        if candidate.exists():
            video_path = str(candidate)

    total_frames = _video_frame_count(video_path) if video_path else 0

    if video_path and st.checkbox("Create calibration from cones", value=False):
        st.subheader("Calibration (cones)")
        st.caption(
            "Click four cones to define a rectangle on the ground plane, then drag to adjust."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            width_m = st.number_input(
                "Rectangle width (meters)",
                min_value=1.0,
                value=float(cfg.CALIB_RECT_WIDTH_M),
                step=0.5,
            )
        with col_b:
            height_m = st.number_input(
                "Rectangle height (meters)",
                min_value=1.0,
                value=float(cfg.CALIB_RECT_HEIGHT_M),
                step=0.5,
            )

        frame_idx = 0
        if total_frames:
            frame_idx = st.slider(
                "Pick calibration frame",
                min_value=0,
                max_value=max(0, total_frames - 1),
                value=0,
                step=1,
            )

        frame = _read_video_frame(video_path, frame_idx)
        if frame is None:
            st.error("Unable to read the calibration frame.")
        else:
            if st.session_state.get("calibration_video") != video_path:
                st.session_state["calibration_video"] = video_path
                st.session_state["calibration_points"] = []
                st.session_state["calibration_path"] = None
                st.session_state["calibration_auto_path"] = None
                st.session_state["calibration_frame_idx"] = frame_idx
                st.session_state["calibration_canvas_key"] = f"calibration_canvas_{uuid.uuid4().hex}"
                st.session_state["calibration_canvas_init"] = True

            if st.session_state.get("calibration_frame_idx") != frame_idx:
                st.session_state["calibration_frame_idx"] = frame_idx
                st.session_state["calibration_points"] = []
                st.session_state["calibration_auto_path"] = None
                st.session_state["calibration_canvas_key"] = f"calibration_canvas_{uuid.uuid4().hex}"
                st.session_state["calibration_canvas_init"] = True

            canvas_image, scale = _prepare_canvas_frame(frame)
            points = st.session_state.get("calibration_points", [])

            mode = st.radio(
                "Canvas mode",
                options=["Add points", "Adjust points"],
                horizontal=True,
                index=0 if len(points) < 4 else 1,
            )

            try:
                _ensure_canvas_compat()
                from streamlit_drawable_canvas import st_canvas
            except Exception:
                st.warning(
                    "Install `streamlit-drawable-canvas` to use the calibration overlay."
                )
                st.code("pip install streamlit-drawable-canvas")
                st.stop()

            canvas_key = st.session_state.get("calibration_canvas_key", "calibration_canvas")
            canvas_init = st.session_state.get("calibration_canvas_init", True)
            initial = _canvas_initial_drawing(points) if canvas_init and points else None
            drawing_mode = "point" if mode == "Add points" else "transform"
            canvas_result = st_canvas(
                fill_color="rgba(255, 196, 0, 0.6)",
                stroke_width=2,
                stroke_color="#ffc400",
                background_image=canvas_image,
                update_streamlit=True,
                width=canvas_image.width,
                height=canvas_image.height,
                drawing_mode=drawing_mode,
                point_display_radius=6,
                display_toolbar=True,
                key=canvas_key,
                initial_drawing=initial,
            )
            if canvas_init:
                st.session_state["calibration_canvas_init"] = False

            if canvas_result.json_data is not None:
                points = _extract_canvas_points(canvas_result.json_data)
                if len(points) > 4:
                    points = points[:4]
                st.session_state["calibration_points"] = points

            points_display = st.session_state.get("calibration_points", [])
            st.write(f"Points: {len(points_display)}/4")
            if points_display:
                coords = [
                    {"x": round(p[0] / scale, 1), "y": round(p[1] / scale, 1)}
                    for p in points_display
                ]
                st.dataframe(coords, hide_index=True, use_container_width=True)

            col_left, col_right = st.columns(2)
            with col_left:
                if st.button("Reset points"):
                    st.session_state["calibration_points"] = []
                    st.session_state["calibration_auto_path"] = None
                    st.session_state["calibration_path"] = None
                    st.session_state["calibration_canvas_key"] = f"calibration_canvas_{uuid.uuid4().hex}"
                    st.session_state["calibration_canvas_init"] = True
            with col_right:
                if st.button("Save calibration", type="primary"):
                    if len(points_display) != 4:
                        st.error("Select exactly 4 points before saving.")
                    else:
                        image_points = [
                            (p[0] / scale, p[1] / scale) for p in points_display
                        ]
                        calibration = build_calibration(
                            image_points=image_points,
                            field_width_m=width_m,
                            field_height_m=height_m,
                            reorder=True,
                        )
                        tmp_calib = tempfile.NamedTemporaryFile(
                            delete=False, suffix=".json"
                        )
                        tmp_calib.close()
                        save_calibration(tmp_calib.name, calibration)
                        st.session_state["calibration_path"] = tmp_calib.name
                        st.success("Calibration saved. It will be used for detection.")
                        calib_bytes = Path(tmp_calib.name).read_bytes()
                        st.download_button(
                            "Download calibration JSON",
                            data=calib_bytes,
                            file_name="cone_calibration.json",
                            mime="application/json",
                        )

            if len(points_display) == 4:
                image_points = [(p[0] / scale, p[1] / scale) for p in points_display]
                calibration = build_calibration(
                    image_points=image_points,
                    field_width_m=width_m,
                    field_height_m=height_m,
                    reorder=True,
                )
                auto_path = st.session_state.get("calibration_auto_path")
                if not auto_path:
                    tmp_calib = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
                    tmp_calib.close()
                    auto_path = tmp_calib.name
                    st.session_state["calibration_auto_path"] = auto_path
                save_calibration(auto_path, calibration)
                if st.session_state.get("calibration_path") is None:
                    st.session_state["calibration_path"] = auto_path
                st.success("Calibration active for detection.")

    options = sidebar_options(st.session_state.get("calibration_path"))
    report_sections = report_options()
    
    # Processing Options Section
    st.markdown(
        """<div class="section-container">
            <div class="section-header"><span class="section-icon">⚙️</span> Processing Options</div>
        """,
        unsafe_allow_html=True,
    )
    
    col_frames, col_viz1, col_viz2 = st.columns([2, 1, 1])
    with col_frames:
        max_frames = st.number_input(
            "Max frames to process (0 = full video)",
            min_value=0,
            step=50,
            value=0,
        )
    with col_viz1:
        st.checkbox(
            "🎬 Show preview",
            value=True,
            key="viz_toggle",
            help="Turn off to speed up processing.",
        )
    with col_viz2:
        st.checkbox(
            "🖥️ Fullscreen preview",
            value=False,
            key="fullscreen_toggle",
            help="Show the annotated frame across the full width.",
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Run Button
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶️  Run Detection", use_container_width=True, type="primary")
    st.markdown("<br>", unsafe_allow_html=True)

    # Live Dashboard Section (placeholders)
    st.markdown(
        """<div class="section-container">
            <div class="section-header"><span class="section-icon">📊</span> Live Analysis Dashboard</div>
        """,
        unsafe_allow_html=True,
    )
    
    progress = st.progress(0.0)
    
    col_metrics, col_preview = st.columns([3, 2])
    with col_metrics:
        stats_placeholder = st.empty()
    with col_preview:
        frame_placeholder_side = st.empty()
    
    st.markdown("</div>", unsafe_allow_html=True)
    frame_placeholder_full = st.empty()
    analysis_cache = st.session_state.get("analysis_cache")
    cache_video_path = None
    if isinstance(analysis_cache, dict):
        cache_video_path = analysis_cache.get("video_path")
    use_cache = False
    display_stride_used = options.display_stride
    max_frames_used = None if max_frames <= 0 else int(max_frames)

    if not run_btn:
        if not analysis_cache:
            return
        if cache_video_path and video_path and cache_video_path != video_path:
            return
        use_cache = True

    if use_cache:
        cached = analysis_cache if isinstance(analysis_cache, dict) else {}
        left = cached.get("left", 0)
        right = cached.get("right", 0)
        processed_frames = cached.get("processed_frames", 0)
        total_frames = cached.get("total_frames", total_frames)
        input_fps = cached.get("input_fps")
        display_stride_used = cached.get("display_stride", display_stride_used)
        max_frames_used = cached.get("max_frames", max_frames_used)
        last_avg_speed = cached.get("last_avg_speed")
        last_max_speed = cached.get("last_max_speed")
        speed_points = cached.get("speed_points", [])
        total_jumps = cached.get("total_jumps", 0)
        highest_jump_m = cached.get("highest_jump_m")
        highest_jump_px = cached.get("highest_jump_px")
        shot_log = cached.get("shot_log", [])
        shot_count = cached.get("shot_count", len(shot_log))
        snapshots = cached.get("snapshots", [])
        total_time_sec = cached.get("total_time_sec")
        total_distance_m = cached.get("total_distance_m")
        peak_accel_mps2 = cached.get("peak_accel_mps2")
        peak_decel_mps2 = cached.get("peak_decel_mps2")
        annotated_video_path = cached.get("annotated_video_path")
        frame_records = cached.get("frame_records", [])
        ground_overlay = cached.get("ground_overlay")
        options_snapshot = cached.get("options_snapshot", {})
        analysis_use_homography = cached.get("use_homography", options.use_homography)
        if input_fps is None and video_path:
            input_fps = _video_fps(video_path)
    else:
        if not video_path:
            st.error("Please upload a video or select a sample file.")
            return

        total_frames = total_frames or _video_frame_count(video_path)
        input_fps = _video_fps(video_path)
        output_fps = max(1.0, input_fps / max(1, options.display_stride))
        left = right = 0
        processed_frames = 0
        last_avg_speed = None
        last_max_speed = None
        speed_points = []
        total_jumps = 0
        highest_jump_m = None
        highest_jump_px = None
        shot_log = []
        shot_count = 0
        snapshots = []
        total_time_sec = None
        total_distance_m = None
        peak_accel_mps2 = None
        peak_decel_mps2 = None
        annotated_video_path = None
        frame_records = []
        ground_overlay = None
        options_snapshot = {
            "draw_ball_vector": options.draw_ball_vector,
            "ball_vector_scale": options.ball_vector_scale,
            "show_ball_speed": options.show_ball_speed,
            "draw_ball_trail": options.draw_ball_trail,
            "ball_trail_length": options.ball_trail_length,
            "ball_trail_max_gap_frames": options.ball_trail_max_gap_frames,
            "show_player_speed": options.show_player_speed,
        }
        analysis_use_homography = options.use_homography
        video_writer = None
        snapshot_dir = None
        last_shot_log_len = 0

        gen = run_touch_detection(
            video_path,
            options=options,
            max_frames=None if max_frames <= 0 else int(max_frames),
        )

        with st.spinner("Running touch detection..."):
            error_msg = None
            try:
                for result in gen:
                    left = result.left_touches
                    right = result.right_touches
                    total_touches = left + right
                    last_avg_speed = result.avg_speed_kmh
                    last_max_speed = result.max_speed_kmh
                    total_jumps = result.total_jumps
                    highest_jump_m = result.highest_jump_m
                    highest_jump_px = result.highest_jump_px
                    if result.shot_events is not None:
                        shot_log = result.shot_events
                    shot_count = result.shot_count
                    total_time_sec = result.total_time_sec
                    total_distance_m = result.total_distance_m
                    peak_accel_mps2 = result.peak_accel_mps2
                    peak_decel_mps2 = result.peak_decel_mps2
                    processed_frames += 1

                    avg_speed_text = _format_speed(last_avg_speed, options.use_metric_display)
                    max_speed_text = _format_speed(last_max_speed, options.use_metric_display)
                    jump_height_text = "--"
                    if highest_jump_m is not None:
                        jump_height_text = f"{highest_jump_m:.2f} m"
                    elif highest_jump_px is not None:
                        jump_height_text = f"{highest_jump_px:.0f} px"
                    last_force_text = "--"
                    if shot_log:
                        last_entry = shot_log[-1]
                        last_type = last_entry.get("type", "pass").capitalize()
                        force_kmh = last_entry.get("force_kmh")
                        force_px_s = last_entry.get("force_px_s")
                        if force_kmh is not None:
                            last_force_text = f"{last_type} - {_format_speed(force_kmh, options.use_metric_display)}"
                        elif force_px_s is not None:
                            last_force_text = f"{last_type} - {force_px_s:.1f} px/s"
                    time_text = _format_duration(total_time_sec)
                    distance_text = _format_distance(total_distance_m, options.use_metric_display)
                    accel_text = _format_accel(peak_accel_mps2)
                    decel_text = _format_accel(peak_decel_mps2)
                    overlay_lines = []
                    metric_cards = []
                    
                    if "Touches" in report_sections:
                        overlay_lines.append(f"Touches (L / R): {left} / {right}")
                        overlay_lines.append(f"Total ball touches: {total_touches}")
                        metric_cards.append(
                            f"""<div class="metric-card">
                                <div class="metric-icon">👟</div>
                                <div class="metric-label">Ball Touches</div>
                                <div class="metric-value metric-value-highlight">{left} / {right}</div>
                                <div class="metric-subvalue">Left / Right</div>
                            </div>"""
                        )
                    
                    if "Speed" in report_sections:
                        overlay_lines.append(f"Player speed (avg / max): {avg_speed_text} / {max_speed_text}")
                        metric_cards.append(
                            f"""<div class="metric-card">
                                <div class="metric-icon">⚡</div>
                                <div class="metric-label">Player Speed</div>
                                <div class="metric-value">{avg_speed_text}</div>
                                <div class="metric-subvalue">Max: {max_speed_text}</div>
                            </div>"""
                        )
                    
                    if "Jumps" in report_sections:
                        overlay_lines.append(f"Jumps / Highest: {total_jumps} / {jump_height_text}")
                        metric_cards.append(
                            f"""<div class="metric-card">
                                <div class="metric-icon">🦘</div>
                                <div class="metric-label">Jumps</div>
                                <div class="metric-value">{total_jumps}</div>
                                <div class="metric-subvalue">Max: {jump_height_text}</div>
                            </div>"""
                        )
                    
                    if "Shots" in report_sections:
                        overlay_lines.append(f"Shots / Last force: {shot_count} / {last_force_text}")
                        metric_cards.append(
                            f"""<div class="metric-card">
                                <div class="metric-icon">🎯</div>
                                <div class="metric-label">Shots</div>
                                <div class="metric-value">{shot_count}</div>
                                <div class="metric-subvalue">{last_force_text}</div>
                            </div>"""
                        )
                    
                    if "Time & Distance" in report_sections:
                        overlay_lines.append(f"Time analyzed / Distance: {time_text} / {distance_text}")
                        metric_cards.append(
                            f"""<div class="metric-card">
                                <div class="metric-icon">🏃</div>
                                <div class="metric-label">Distance</div>
                                <div class="metric-value">{distance_text}</div>
                                <div class="metric-subvalue">Time: {time_text}</div>
                            </div>"""
                        )
                    
                    if "Acceleration" in report_sections:
                        overlay_lines.append(f"Accel / Decel (peak): {accel_text} / {decel_text}")
                        metric_cards.append(
                            f"""<div class="metric-card">
                                <div class="metric-icon">📈</div>
                                <div class="metric-label">Acceleration</div>
                                <div class="metric-value">{accel_text}</div>
                                <div class="metric-subvalue">Decel: {decel_text}</div>
                            </div>"""
                        )
                    
                    # Frame counter badge
                    if "Processing info" in report_sections:
                        overlay_lines.append(f"Frame: {result.frame_idx}")
                    
                    # Render metrics grid
                    if metric_cards:
                        grid_html = '<div class="metrics-grid">' + ''.join(metric_cards) + '</div>'
                        if "Processing info" in report_sections:
                            grid_html += f'<div style="text-align:right;margin-top:0.5rem;"><span style="background:rgba(34,211,238,0.1);color:#22d3ee;padding:0.25rem 0.75rem;border-radius:999px;font-size:0.8rem;font-weight:600;">Frame {result.frame_idx}</span></div>'
                        stats_placeholder.markdown(grid_html, unsafe_allow_html=True)
                    else:
                        stats_placeholder.empty()
                    overlay_frame = result.annotated.copy()
                    if overlay_lines:
                        _draw_stats_overlay(overlay_frame, overlay_lines)
                    if result.frame_meta is not None:
                        if ground_overlay is None and "ground_overlay" in result.frame_meta:
                            ground_overlay = result.frame_meta.get("ground_overlay")
                        analysis_use_homography = result.frame_meta.get(
                            "use_homography", analysis_use_homography
                        )
                        frame_records.append(
                            {
                                "frame_idx": result.frame_idx,
                                "meta": result.frame_meta,
                                "stats": {
                                    "left": left,
                                    "right": right,
                                    "avg_speed_kmh": result.avg_speed_kmh,
                                    "max_speed_kmh": result.max_speed_kmh,
                                    "total_jumps": result.total_jumps,
                                    "highest_jump_m": result.highest_jump_m,
                                    "highest_jump_px": result.highest_jump_px,
                                    "shot_count": result.shot_count,
                                    "pass_count": result.pass_count,
                                    "total_time_sec": result.total_time_sec,
                                    "total_distance_m": result.total_distance_m,
                                    "peak_accel_mps2": result.peak_accel_mps2,
                                    "peak_decel_mps2": result.peak_decel_mps2,
                                },
                            }
                        )
                    if "Snapshots" in report_sections and len(snapshots) < SNAPSHOT_MAX:
                        if len(shot_log) > last_shot_log_len:
                            if snapshot_dir is None:
                                snapshot_dir = Path(
                                    tempfile.mkdtemp(prefix="soccer_snapshots_")
                                )
                            new_events = shot_log[last_shot_log_len:]
                            for event in new_events:
                                if len(snapshots) >= SNAPSHOT_MAX:
                                    break
                                event_type = event.get("type", "event")
                                event_no = event.get("shot", len(snapshots) + 1)
                                event_frame = event.get("frame_idx")
                                filename = f"{event_type}_{event_no}_frame_{event_frame or result.frame_idx}.jpg"
                                snapshot_path = snapshot_dir / filename
                                width, height = _save_snapshot(overlay_frame, snapshot_path)
                                snapshots.append(
                                    {
                                        "id": len(snapshots) + 1,
                                        "type": event_type,
                                        "shot": event_no,
                                        "frame_idx": event_frame,
                                        "time_sec": event.get("time_sec"),
                                        "kick_frame": event.get("kick_frame"),
                                        "image_path": str(snapshot_path),
                                        "width": width,
                                        "height": height,
                                    }
                                )
                            last_shot_log_len = len(shot_log)
                    if video_writer is None:
                        tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        annotated_video_path = tmp_video.name
                        tmp_video.close()
                        video_writer = _init_video_writer(
                            annotated_video_path,
                            overlay_frame,
                            output_fps,
                        )
                        if video_writer is None:
                            Path(annotated_video_path).unlink(missing_ok=True)
                            annotated_video_path = None
                    if video_writer is not None:
                        video_writer.write(overlay_frame)

                    if result.avg_speed_kmh is not None:
                        speed_points.append(
                            {
                                "frame": result.frame_idx,
                                "avg_speed_kmh": result.avg_speed_kmh,
                                "max_speed_kmh": result.max_speed_kmh,
                            }
                        )

                    if st.session_state.get("viz_toggle", True):
                        caption = f"Frame {result.frame_idx}"
                        frame_rgb = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
                        if st.session_state.get("fullscreen_toggle", False):
                            frame_placeholder_side.empty()
                            frame_placeholder_full.image(
                                frame_rgb,
                                caption=caption,
                                channels="RGB",
                                width=frame_rgb.shape[1],
                            )
                        else:
                            frame_placeholder_full.empty()
                            frame_placeholder_side.image(
                                frame_rgb,
                                caption=caption,
                                channels="RGB",
                                width=320,
                            )

                    if total_frames:
                        progress.progress(min(1.0, result.frame_idx / total_frames))
            except Exception as exc:
                error_msg = str(exc)
            finally:
                if video_writer is not None:
                    video_writer.release()
                gen.close()
            if error_msg:
                if annotated_video_path:
                    Path(annotated_video_path).unlink(missing_ok=True)
                st.error(f"Detection failed: {error_msg}")
                return

        previous_cache = st.session_state.get("analysis_cache")
        previous_path = None
        if isinstance(previous_cache, dict):
            previous_path = previous_cache.get("annotated_video_path")
            for snap in previous_cache.get("snapshots", []):
                snap_path = snap.get("image_path")
                if snap_path:
                    Path(snap_path).unlink(missing_ok=True)
        if previous_path and previous_path != annotated_video_path:
            Path(previous_path).unlink(missing_ok=True)
        st.session_state["analysis_cache"] = {
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "input_fps": input_fps,
            "display_stride": options.display_stride,
            "max_frames": max_frames_used,
            "video_path": video_path,
            "left": left,
            "right": right,
            "last_avg_speed": last_avg_speed,
            "last_max_speed": last_max_speed,
            "total_jumps": total_jumps,
            "highest_jump_m": highest_jump_m,
            "highest_jump_px": highest_jump_px,
            "shot_log": shot_log,
            "shot_count": shot_count,
            "total_time_sec": total_time_sec,
            "total_distance_m": total_distance_m,
            "peak_accel_mps2": peak_accel_mps2,
            "peak_decel_mps2": peak_decel_mps2,
            "speed_points": speed_points,
            "snapshots": snapshots,
            "annotated_video_path": annotated_video_path,
            "frame_records": frame_records,
            "ground_overlay": ground_overlay,
            "options_snapshot": options_snapshot,
            "use_homography": analysis_use_homography,
        }

    progress.progress(1.0)
    st.success("✅ Analysis Complete!")

    report_data = {"report_sections": _report_section_list(report_sections)}
    if "Processing info" in report_sections:
        report_data["processing"] = {
            "processed_frames": processed_frames,
            "total_frames": total_frames or None,
            "input_fps": input_fps,
            "display_stride": display_stride_used,
            "max_frames": max_frames_used,
        }
    if "Touches" in report_sections:
        report_data["touches"] = {
            "left": left,
            "right": right,
            "total": left + right,
        }
    if "Speed" in report_sections:
        report_data["speed"] = {
            "avg_kmh": last_avg_speed,
            "max_kmh": last_max_speed,
        }
    if "Jumps" in report_sections:
        report_data["jumps"] = {
            "total": total_jumps,
            "highest_m": highest_jump_m,
            "highest_px": highest_jump_px,
        }
    if "Shots" in report_sections:
        report_data["shots"] = {
            "count": shot_count,
        }
    if "Time & Distance" in report_sections:
        report_data["time_distance"] = {
            "total_time_sec": total_time_sec,
            "total_distance_m": total_distance_m,
        }
    if "Acceleration" in report_sections:
        report_data["acceleration"] = {
            "peak_accel_mps2": peak_accel_mps2,
            "peak_decel_mps2": peak_decel_mps2,
        }
    if "Shot log" in report_sections:
        report_data["shot_log"] = shot_log
    if "Speed chart" in report_sections:
        report_data["speed_points"] = speed_points
    if "Snapshots" in report_sections:
        report_data["snapshots"] = snapshots

    trimmed_report, truncation = _trim_report_for_context(
        report_data, MAX_REPORT_CHARS
    )
    report_json = json.dumps(trimmed_report, indent=2, ensure_ascii=True)
    llm_report_data = dict(report_data)
    llm_report_data.pop("speed_points", None)
    llm_report_data.pop("snapshots", None)
    llm_trimmed_report, llm_truncation = _trim_report_for_context(
        llm_report_data, MAX_REPORT_CHARS
    )
    llm_report_json = json.dumps(llm_trimmed_report, indent=2, ensure_ascii=True)
    st.session_state["report_json"] = report_json
    st.session_state["llm_report_json"] = llm_report_json
    st.session_state["llm_truncation"] = llm_truncation
    if st.session_state.get("chat_video_path") != video_path:
        st.session_state["chat_video_path"] = video_path
        st.session_state["chat_messages"] = []
    
    # Results Section
    st.markdown(
        """<div class="section-container">
            <div class="section-header"><span class="section-icon">📈</span> Analysis Results</div>
        """,
        unsafe_allow_html=True,
    )
    
    # Create a tab-like selector (persists across reruns)
    tab_labels = [
        "📊 Summary",
        "🎬 Player",
        "🎯 Shot Log",
        "📈 Speed Chart",
        "📸 Snapshots",
        "💬 Coach Chat",
        "💾 Export",
    ]
    active_tab = st.radio(
        "Results",
        tab_labels,
        horizontal=True,
        label_visibility="collapsed",
        key="results_tab",
    )
    
    # Tab 1: Summary
    if active_tab == "📊 Summary":
        completion_display = _format_duration(total_time_sec)
        distance_display = _format_distance(total_distance_m)
        accel_display = _format_accel(peak_accel_mps2)
        decel_display = _format_accel(peak_decel_mps2)
        
        # Final metrics grid
        summary_cards = []
        
        if "Touches" in report_sections:
            summary_cards.append(
                f"""<div class="metric-card">
                    <div class="metric-icon">👟</div>
                    <div class="metric-label">Final Touches</div>
                    <div class="metric-value metric-value-highlight">{left} / {right}</div>
                    <div class="metric-subvalue">Left / Right • Total: {left + right}</div>
                </div>"""
            )
        
        if "Speed" in report_sections and (last_avg_speed is not None or last_max_speed is not None):
            speed_unit = "m/s" if options.use_metric_display else "km/h"
            avg_display = _format_speed(last_avg_speed, options.use_metric_display) if last_avg_speed is not None else "--"
            max_display = _format_speed(last_max_speed, options.use_metric_display) if last_max_speed is not None else "--"
            summary_cards.append(
                f"""<div class="metric-card">
                    <div class="metric-icon">⚡</div>
                    <div class="metric-label">Speed</div>
                    <div class="metric-value">{avg_display} <span style="font-size:0.8rem;color:var(--text-secondary);">avg</span></div>
                    <div class="metric-subvalue">Peak: {max_display}</div>
                </div>"""
            )
        
        if "Time & Distance" in report_sections:
            summary_cards.append(
                f"""<div class="metric-card">
                    <div class="metric-icon">🏃</div>
                    <div class="metric-label">Distance Covered</div>
                    <div class="metric-value">{distance_display}</div>
                    <div class="metric-subvalue">Duration: {completion_display}</div>
                </div>"""
            )
        
        if "Jumps" in report_sections:
            jump_display = "--"
            if highest_jump_m is not None:
                jump_display = f"{highest_jump_m:.2f} m"
            elif highest_jump_px is not None:
                jump_display = f"{highest_jump_px:.0f} px"
            summary_cards.append(
                f"""<div class="metric-card">
                    <div class="metric-icon">🦘</div>
                    <div class="metric-label">Jumps</div>
                    <div class="metric-value">{total_jumps}</div>
                    <div class="metric-subvalue">Highest: {jump_display}</div>
                </div>"""
            )
        
        if "Shots" in report_sections:
            summary_cards.append(
                f"""<div class="metric-card">
                    <div class="metric-icon">🎯</div>
                    <div class="metric-label">Shots Detected</div>
                    <div class="metric-value metric-value-highlight">{shot_count}</div>
                    <div class="metric-subvalue">Events in shot log</div>
                </div>"""
            )
        
        if "Acceleration" in report_sections:
            summary_cards.append(
                f"""<div class="metric-card">
                    <div class="metric-icon">📈</div>
                    <div class="metric-label">Peak Acceleration</div>
                    <div class="metric-value">{accel_display}</div>
                    <div class="metric-subvalue">Decel: {decel_display}</div>
                </div>"""
            )
        
        if "Processing info" in report_sections:
            summary_cards.append(
                f"""<div class="metric-card">
                    <div class="metric-icon">🎬</div>
                    <div class="metric-label">Frames Processed</div>
                    <div class="metric-value">{processed_frames}</div>
                    <div class="metric-subvalue">of {total_frames or 'N/A'} total</div>
                </div>"""
            )
        
        if summary_cards:
            st.markdown('<div class="metrics-grid">' + ''.join(summary_cards) + '</div>', unsafe_allow_html=True)
    # Tab 2: Player
    elif active_tab == "🎬 Player":
        if not video_path:
            st.info("ℹ️ Upload a video or select a sample to use the player.")
        elif not frame_records:
            st.info("ℹ️ Run detection to generate frame data for the player.")
        else:
            frame_lookup = {
                rec.get("frame_idx"): rec
                for rec in frame_records
                if rec.get("frame_idx") is not None
            }
            available_frames = sorted(frame_lookup.keys())
            if not available_frames:
                st.info("ℹ️ Frame metadata is empty. Re-run detection.")
            else:
                if display_stride_used > 1:
                    st.caption(
                        "Note: overlay data is recorded every "
                        f"{display_stride_used} frames. For full coverage, set Display stride to 1 and re-run."
                    )

                defaults = options_snapshot or {}
                default_vector_scale = float(
                    defaults.get("ball_vector_scale", cfg.BALL_VECTOR_SCALE)
                )
                default_trail_len = int(
                    defaults.get("ball_trail_length", cfg.BALL_TRAIL_LENGTH)
                )
                default_trail_gap = int(
                    defaults.get("ball_trail_max_gap_frames", cfg.BALL_TRAIL_MAX_GAP_FRAMES)
                )

                with st.expander("Overlay controls", expanded=True):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        show_players = st.checkbox("Player boxes", value=True, key="player_show_players")
                        show_ids = st.checkbox("Player IDs", value=True, key="player_show_ids")
                        show_feet = st.checkbox("Feet markers", value=True, key="player_show_feet")
                    with col_b:
                        show_ball = st.checkbox("Ball marker", value=True, key="player_show_ball")
                        show_ball_vector = st.checkbox(
                            "Ball vector",
                            value=bool(defaults.get("draw_ball_vector", cfg.DRAW_BALL_VECTOR)),
                            key="player_show_ball_vector",
                        )
                        show_ball_speed = st.checkbox(
                            "Ball speed",
                            value=bool(defaults.get("show_ball_speed", cfg.SHOW_BALL_SPEED)),
                            key="player_show_ball_speed",
                        )
                    with col_c:
                        show_ball_trail = st.checkbox(
                            "Ball trail",
                            value=bool(defaults.get("draw_ball_trail", cfg.DRAW_BALL_TRAIL)),
                            key="player_show_ball_trail",
                        )
                        show_player_speed = st.checkbox(
                            "Player speed",
                            value=bool(defaults.get("show_player_speed", cfg.SHOW_PLAYER_SPEED)),
                            key="player_show_player_speed",
                        )
                        show_annotations = st.checkbox(
                            "Annotations",
                            value=True,
                            key="player_show_annotations",
                        )
                    if show_ball_trail:
                        trail_len = st.slider(
                            "Trail length (frames)",
                            min_value=2,
                            max_value=60,
                            value=max(2, default_trail_len),
                            step=1,
                            key="player_trail_len",
                        )
                    else:
                        trail_len = default_trail_len
                    vector_scale = st.slider(
                        "Vector scale",
                        min_value=4.0,
                        max_value=24.0,
                        value=default_vector_scale,
                        step=1.0,
                        key="player_vector_scale",
                    )

                max_frame = total_frames or available_frames[-1]
                default_frame = available_frames[-1]
                frame_idx = st.slider(
                    "Frame",
                    min_value=1,
                    max_value=max_frame,
                    value=default_frame,
                    step=1,
                    key="player_frame_idx",
                )

                frame_bgr = _read_video_frame(video_path, max(0, frame_idx - 1))
                if frame_bgr is None:
                    st.error("Unable to read the selected frame.")
                else:
                    overlay_rec = frame_lookup.get(frame_idx)
                    frame_render = frame_bgr.copy()
                    if overlay_rec is None:
                        st.info(
                            "Overlay data not available for this frame. Re-run with Display stride = 1 for full coverage."
                        )
                    else:
                        meta = overlay_rec.get("meta", {})
                        stats = overlay_rec.get("stats", {})
                        use_homography = bool(
                            meta.get("use_homography", analysis_use_homography)
                        )
                        if show_ball and show_ball_trail:
                            trail_points = _collect_ball_trail(
                                frame_lookup,
                                frame_idx,
                                max_len=max(2, trail_len),
                                max_gap_frames=max(1, default_trail_gap),
                            )
                            _draw_ball_trail_overlay(frame_render, trail_points)
                        if show_players:
                            _draw_player_overlays(
                                frame_render,
                                meta.get("players", []),
                                show_ids=show_ids,
                                show_feet=show_feet,
                                show_speed=show_player_speed,
                                use_metric_display=options.use_metric_display,
                            )
                        if show_ball:
                            _draw_ball_overlay(
                                frame_render,
                                meta.get("ball"),
                                show_vector=show_ball_vector,
                                show_speed=show_ball_speed,
                                use_homography=use_homography,
                                vector_scale=vector_scale,
                            )
                        if show_annotations:
                            event_overlay = meta.get("event_overlay")
                            if event_overlay:
                                _draw_event_overlay(frame_render, event_overlay)
                            overlay_lines = []
                            if "Touches" in report_sections:
                                overlay_lines.append(
                                    f"Touches (L / R): {stats.get('left', 0)} / {stats.get('right', 0)}"
                                )
                            if "Speed" in report_sections:
                                avg_speed_text = _format_speed(
                                    stats.get("avg_speed_kmh"),
                                    options.use_metric_display,
                                )
                                max_speed_text = _format_speed(
                                    stats.get("max_speed_kmh"),
                                    options.use_metric_display,
                                )
                                overlay_lines.append(
                                    f"Player speed (avg / max): {avg_speed_text} / {max_speed_text}"
                                )
                            if "Jumps" in report_sections:
                                jump_height_text = "--"
                                if stats.get("highest_jump_m") is not None:
                                    jump_height_text = f"{stats['highest_jump_m']:.2f} m"
                                elif stats.get("highest_jump_px") is not None:
                                    jump_height_text = f"{stats['highest_jump_px']:.0f} px"
                                overlay_lines.append(
                                    f"Jumps / Highest: {stats.get('total_jumps', 0)} / {jump_height_text}"
                                )
                            if "Shots" in report_sections:
                                overlay_lines.append(
                                    f"Shots / Passes: {stats.get('shot_count', 0)} / {stats.get('pass_count', 0)}"
                                )
                            if "Time & Distance" in report_sections:
                                overlay_lines.append(
                                    f"Time analyzed / Distance: {_format_duration(stats.get('total_time_sec'))} / "
                                    f"{_format_distance(stats.get('total_distance_m'), options.use_metric_display)}"
                                )
                            if "Acceleration" in report_sections:
                                overlay_lines.append(
                                    f"Accel / Decel (peak): {_format_accel(stats.get('peak_accel_mps2'))} / "
                                    f"{_format_accel(stats.get('peak_decel_mps2'))}"
                                )
                            if "Processing info" in report_sections:
                                overlay_lines.append(f"Frame: {frame_idx}")
                            if overlay_lines:
                                _draw_stats_overlay(frame_render, overlay_lines)

                    frame_rgb = cv2.cvtColor(frame_render, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption=f"Frame {frame_idx}", use_container_width=True)

    # Tab 2: Shot Log
    elif active_tab == "🎯 Shot Log":
        if "Shot log" in report_sections:
            if shot_log:
                st.markdown(f"**{shot_count}** shots/passes detected", unsafe_allow_html=True)
                shot_df = pd.DataFrame(shot_log)
                if "time_sec" in shot_df:
                    shot_df["time"] = shot_df["time_sec"].apply(
                        lambda t: f"{t:.2f}s" if pd.notna(t) else "--"
                    )
                else:
                    shot_df["time"] = "--"
                if "shot" not in shot_df:
                    shot_df["shot"] = range(1, len(shot_df) + 1)
                if "type" not in shot_df:
                    shot_df["type"] = "pass"
                def _fmt_force(row):
                    if pd.notna(row.get("force_kmh")):
                        return _format_speed(row['force_kmh'], options.use_metric_display)
                    if pd.notna(row.get("force_px_s")):
                        return f"{row['force_px_s']:.1f} px/s"
                    return "--"
                shot_df["force"] = shot_df.apply(_fmt_force, axis=1)
                display_cols = {
                    "shot": "Shot #",
                    "time": "Time",
                    "type": "Type",
                    "foot": "Foot",
                    "force": "Force",
                    "frame_idx": "Frame",
                    "track_id": "Track ID",
                }
                st.dataframe(
                    shot_df[list(display_cols.keys())].rename(columns=display_cols),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info("ℹ️ No shot events detected in this video.")
        else:
            st.info("ℹ️ Shot log not included in report options.")
    
    # Tab 3: Speed Chart
    elif active_tab == "📈 Speed Chart":
        if "Speed chart" in report_sections:
            if speed_points:
                st.markdown("**Player speed over time**")
                df = pd.DataFrame(speed_points).set_index("frame")
                st.line_chart(df)
            else:
                st.info("ℹ️ No speed measurements available for charting.")
        else:
            st.info("ℹ️ Speed chart not included in report options.")
    
    # Tab 4: Snapshots
    elif active_tab == "📸 Snapshots":
        if "Snapshots" in report_sections:
            if snapshots:
                st.markdown(f"**Captured snapshots (max {SNAPSHOT_MAX})**")
                cols = st.columns(3)
                for idx, snap in enumerate(snapshots):
                    caption_parts = []
                    if snap.get("type"):
                        caption_parts.append(str(snap.get("type")).capitalize())
                    if snap.get("shot"):
                        caption_parts.append(f"#{snap.get('shot')}")
                    if snap.get("time_sec") is not None:
                        caption_parts.append(f"{snap.get('time_sec'):.2f}s")
                    caption = " ".join(caption_parts) if caption_parts else "Snapshot"
                    with cols[idx % 3]:
                        st.image(
                            snap.get("image_path", ""),
                            caption=caption,
                            use_container_width=True,
                        )
            else:
                st.info("ℹ️ No snapshots captured yet.")
        else:
            st.info("ℹ️ Snapshots not included in report options.")

    # Tab 5: Coach Chat
    elif active_tab == "💬 Coach Chat":
        st.markdown("**Chat with your report**")
        llm_report_json = st.session_state.get("llm_report_json", report_json)
        llm_truncation = st.session_state.get("llm_truncation")
        st.caption(f"Report context size: ~{_estimate_tokens(llm_report_json)} tokens.")
        if llm_truncation:
            st.warning("⚠️ Report JSON was trimmed to stay under ~100k tokens.")

        prereq_error = _gemini_prereq_error()
        if prereq_error:
            st.info(prereq_error)

        model = st.text_input(
            "Gemini model",
            value=st.session_state.get("gemini_model", DEFAULT_GEMINI_MODEL),
            key="gemini_model",
        )
        temperature = st.slider(
            "Response creativity",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("gemini_temperature", DEFAULT_GEMINI_TEMPERATURE),
            step=0.05,
            key="gemini_temperature",
        )
        max_output_tokens = st.number_input(
            "Max response tokens",
            min_value=128,
            max_value=4096,
            value=st.session_state.get("gemini_max_output_tokens", DEFAULT_GEMINI_MAX_OUTPUT_TOKENS),
            step=128,
            key="gemini_max_output_tokens",
        )
        if st.button("Clear chat", use_container_width=True):
            st.session_state["chat_messages"] = []

        messages = st.session_state.get("chat_messages", [])
        for msg in messages:
            with st.chat_message(msg.get("role", "assistant")):
                st.markdown(msg.get("content", ""))

        prompt = st.chat_input("Ask about this report")
        if prompt:
            if prereq_error:
                st.error(prereq_error)
            elif not model.strip():
                st.error("Enter a Gemini model name to continue.")
            else:
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing report..."):
                        response_text, error = _generate_gemini_response(
                            report_json=llm_report_json,
                            user_message=prompt,
                            history=messages,
                            model=model.strip(),
                            temperature=float(temperature),
                            max_output_tokens=int(max_output_tokens),
                        )
                    if error:
                        st.error(error)
                    else:
                        st.markdown(response_text)
                        messages.append({"role": "user", "content": prompt})
                        messages.append({"role": "assistant", "content": response_text})
                        st.session_state["chat_messages"] = messages

    # Tab 6: Export
    elif active_tab == "💾 Export":
        st.markdown("**Export Analysis Data**")
        
        if truncation:
            st.warning("⚠️ Report JSON was trimmed to stay under ~100k tokens.")

        annotated_bytes = None
        annotated_ready = False
        if annotated_video_path:
            annotated_file = Path(annotated_video_path)
            if annotated_file.exists() and annotated_file.stat().st_size > 0:
                annotated_bytes = annotated_file.read_bytes()
                annotated_ready = True

        col_video, col_json = st.columns(2)
        with col_video:
            if annotated_ready:
                st.download_button(
                    "📥 Download Annotated Video",
                    data=annotated_bytes,
                    file_name="annotated_video.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )
            else:
                st.caption("Annotated video not available.")
        with col_json:
            st.download_button(
                "📥 Download Report JSON",
                data=report_json,
                file_name="soccer_report.json",
                mime="application/json",
                use_container_width=True,
            )

        if shot_log:
            shot_csv = pd.DataFrame(shot_log).to_csv(index=False)
            st.download_button(
                "📥 Download Shot Log CSV",
                data=shot_csv,
                file_name="shot_log.csv",
                mime="text/csv",
                use_container_width=True,
            )
        
        with st.expander("🔍 Preview JSON"):
            st.code(report_json, language="json")
    
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
