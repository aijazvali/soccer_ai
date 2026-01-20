import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

import cv2
import pandas as pd
from PIL import Image
import streamlit as st

import soccer_ai.config as cfg
from soccer_ai.calibration import build_calibration, save_calibration
from soccer_ai.options import TouchOptions
from soccer_ai.pipelines import run_touch_detection


st.set_page_config(page_title="Soccer Touch Detection", layout="wide")

st.markdown(
    """
    <style>
    body {
        background: radial-gradient(circle at 20% 20%, #0f172a 0, #0b1020 40%, #070b17 100%);
        color: #e2e8f0;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background: linear-gradient(120deg, #22d3ee, #6366f1);
        color: white;
        border: none;
        padding: 0.6rem 1.1rem;
        border-radius: 12px;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(99,102,241,0.25);
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #22d3ee, #6366f1);
    }
    .metric-card {
        padding: 1rem 1.2rem;
        background: rgba(255,255,255,0.04);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _save_upload(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[-1] or ".mp4"
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


def _format_distance(meters: Optional[float]) -> str:
    if meters is None or meters < 0:
        return "--"
    if meters >= 1000:
        return f"{meters / 1000:.2f} km"
    return f"{meters:.1f} m"


def _format_accel(accel: Optional[float]) -> str:
    if accel is None:
        return "--"
    return f"{accel:.2f} m/s^2"


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
]
MAX_REPORT_TOKENS = 100000
TOKEN_CHAR_RATIO = 4
MAX_REPORT_CHARS = MAX_REPORT_TOKENS * TOKEN_CHAR_RATIO


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


def sidebar_options(calibration_default: Optional[str] = None) -> TouchOptions:
    st.sidebar.header("Models")
    det_options, pose_options = _list_model_options()
    det_weights = st.sidebar.selectbox(
        "Detection weights",
        options=det_options,
        index=_default_index(det_options, Path(cfg.DETECTOR_WEIGHTS).name),
    )
    pose_weights = st.sidebar.selectbox(
        "Pose weights",
        options=pose_options,
        index=_default_index(pose_options, Path(cfg.POSE_WEIGHTS).name),
    )

    st.sidebar.header("Visualization")
    draw_vector = st.sidebar.checkbox("Draw ball vector", value=cfg.DRAW_BALL_VECTOR)
    vector_scale = st.sidebar.slider(
        "Vector scale",
        min_value=4.0,
        max_value=24.0,
        value=float(cfg.BALL_VECTOR_SCALE),
        step=1.0,
    )
    show_speed = st.sidebar.checkbox("Show ball speed", value=cfg.SHOW_BALL_SPEED)
    show_player_speed = st.sidebar.checkbox(
        "Show player speed", value=cfg.SHOW_PLAYER_SPEED
    )
    show_components = st.sidebar.checkbox(
        "Show velocity components", value=cfg.SHOW_BALL_COMPONENTS
    )
    
    st.sidebar.subheader("Ground Plane Overlay")
    draw_extended_ground = st.sidebar.checkbox(
        "Draw extended ground plane",
        value=cfg.DRAW_EXTENDED_GROUND,
        help="Show the extrapolated ground area beyond the calibration rectangle.",
    )
    extended_ground_multiplier = st.sidebar.slider(
        "Extended area multiplier",
        min_value=1.0,
        max_value=5.0,
        value=float(cfg.EXTENDED_GROUND_MULTIPLIER),
        step=0.5,
        help="How many times larger than the calibration rectangle to extend.",
    )
    draw_ground_grid = st.sidebar.checkbox(
        "Draw ground grid",
        value=cfg.DRAW_GROUND_GRID,
        help="Show grid lines on the ground plane to visualize perspective.",
    )
    ground_grid_spacing = st.sidebar.slider(
        "Grid spacing (meters)",
        min_value=1.0,
        max_value=10.0,
        value=float(cfg.GROUND_GRID_SPACING_M),
        step=0.5,
        help="Distance between grid lines in meters.",
    )

    st.sidebar.header("Touch heuristics")
    event_touch_enabled = st.sidebar.checkbox(
        "Use event-touch heuristic", value=cfg.EVENT_TOUCH_ENABLED
    )
    event_touch_dist_ratio = st.sidebar.slider(
        "Event-touch distance ratio",
        min_value=0.8,
        max_value=2.0,
        value=float(cfg.EVENT_TOUCH_DIST_RATIO),
        step=0.05,
    )

    st.sidebar.header("Calibration")
    calibration_file = st.sidebar.file_uploader(
        "Cone calibration (JSON)",
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
        st.sidebar.caption(f"Using calibration: {Path(calibration_path).name}")

    use_homography = st.sidebar.checkbox(
        "Use homography for calculations",
        value=cfg.USE_HOMOGRAPHY,
        help="Toggle between homography-based meters and pixel logic.",
    )

    st.sidebar.header("Performance")
    display_stride = st.sidebar.slider(
        "Display stride (emit every Nth frame)", min_value=1, max_value=5, value=1
    )

    return TouchOptions(
        detector_weights=det_weights,
        pose_weights=pose_weights,
        draw_ball_vector=draw_vector,
        ball_vector_scale=vector_scale,
        show_ball_speed=show_speed,
        show_player_speed=show_player_speed,
        show_ball_components=show_components,
        event_touch_enabled=event_touch_enabled,
        event_touch_dist_ratio=event_touch_dist_ratio,
        display_stride=display_stride,
        calibration_path=calibration_path,
        use_homography=use_homography,
        draw_extended_ground=draw_extended_ground,
        extended_ground_multiplier=extended_ground_multiplier,
        draw_ground_grid=draw_ground_grid,
        ground_grid_spacing_m=ground_grid_spacing,
    )


def report_options():
    st.sidebar.header("Report")
    selections = st.sidebar.multiselect(
        "Include in report",
        options=REPORT_SECTIONS,
        default=REPORT_SECTIONS,
    )
    return set(selections)


def main():
    st.title("Soccer Analysis")

    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    samples = {
        "Test video (main)": Path("bin/test_video.mp4"),
        "Test video (alt)": Path("bin/test_vide.mp4"),
        "None": None,
    }
    sample_choice = st.selectbox("Or pick a sample in the repo", options=list(samples.keys()), index=0)

    video_path = None
    if uploaded is not None:
        video_path = _save_upload(uploaded)
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
    col_left, col_right = st.columns([2, 1])
    with col_left:
        max_frames = st.number_input(
            "Max frames to process (0 = full video)",
            min_value=0,
            step=50,
            value=0,
        )
    with col_right:
        st.markdown("**Visualization**")
        st.checkbox(
            "Show preview during processing",
            value=True,
            key="viz_toggle",
            help="Turn off to speed up processing.",
        )
        st.checkbox(
            "Full-screen preview",
            value=False,
            key="fullscreen_toggle",
            help="Show the annotated frame across the full width.",
        )

    run_btn = st.button("Run detection", use_container_width=True)

    col_stats, col_preview = st.columns([2, 1])
    progress = col_stats.progress(0.0)
    stats_placeholder = col_stats.empty()
    frame_placeholder_side = col_preview.empty()
    frame_placeholder_full = st.empty()

    if not run_btn:
        return

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
    total_time_sec = None
    total_distance_m = None
    peak_accel_mps2 = None
    peak_decel_mps2 = None
    annotated_video_path = None
    video_writer = None

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

                avg_speed_text = (
                    f"{last_avg_speed:.1f} km/h" if last_avg_speed is not None else "--"
                )
                max_speed_text = (
                    f"{last_max_speed:.1f} km/h" if last_max_speed is not None else "--"
                )
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
                        last_force_text = f"{last_type} - {force_kmh:.1f} km/h"
                    elif force_px_s is not None:
                        last_force_text = f"{last_type} - {force_px_s:.1f} px/s"
                time_text = _format_duration(total_time_sec)
                distance_text = _format_distance(total_distance_m)
                accel_text = _format_accel(peak_accel_mps2)
                decel_text = _format_accel(peak_decel_mps2)
                overlay_lines = []
                card_blocks = []
                if "Touches" in report_sections:
                    overlay_lines.append(f"Touches (L / R): {left} / {right}")
                    overlay_lines.append(f"Total ball touches: {total_touches}")
                    card_blocks.append(
                        f"""
                        <div style="font-size:0.9rem;color:#cbd5e1;">Touches (L / R)</div>
                        <div style="font-size:1.8rem;font-weight:700;">{left} / {right}</div>
                        """
                    )
                if "Speed" in report_sections:
                    overlay_lines.append(
                        f"Player speed (avg / max): {avg_speed_text} / {max_speed_text}"
                    )
                    card_blocks.append(
                        f"""
                        <div style="font-size:0.9rem;color:#cbd5e1;margin-top:0.5rem;">Player speed (avg / max)</div>
                        <div style="font-size:1.2rem;font-weight:600;">{avg_speed_text} / {max_speed_text}</div>
                        """
                    )
                if "Jumps" in report_sections:
                    overlay_lines.append(
                        f"Jumps / Highest: {total_jumps} / {jump_height_text}"
                    )
                    card_blocks.append(
                        f"""
                        <div style="font-size:0.9rem;color:#cbd5e1;margin-top:0.5rem;">Jumps / Highest</div>
                        <div style="font-size:1.2rem;font-weight:600;">{total_jumps} / {jump_height_text}</div>
                        """
                    )
                if "Shots" in report_sections:
                    overlay_lines.append(
                        f"Shots / Last force: {shot_count} / {last_force_text}"
                    )
                    card_blocks.append(
                        f"""
                        <div style="font-size:0.9rem;color:#cbd5e1;margin-top:0.5rem;">Shots / Last force</div>
                        <div style="font-size:1.2rem;font-weight:600;">{shot_count} / {last_force_text}</div>
                        """
                    )
                if "Time & Distance" in report_sections:
                    overlay_lines.append(
                        f"Time analyzed / Distance: {time_text} / {distance_text}"
                    )
                    card_blocks.append(
                        f"""
                        <div style="font-size:0.9rem;color:#cbd5e1;margin-top:0.5rem;">Time / Distance</div>
                        <div style="font-size:1.2rem;font-weight:600;">{time_text} / {distance_text}</div>
                        """
                    )
                if "Acceleration" in report_sections:
                    overlay_lines.append(
                        f"Accel / Decel (peak): {accel_text} / {decel_text}"
                    )
                    card_blocks.append(
                        f"""
                        <div style="font-size:0.9rem;color:#cbd5e1;margin-top:0.5rem;">Accel / Decel (peak)</div>
                        <div style="font-size:1.2rem;font-weight:600;">{accel_text} / {decel_text}</div>
                        """
                    )
                if "Processing info" in report_sections:
                    overlay_lines.append(f"Frame: {result.frame_idx}")
                    card_blocks.append(
                        f"""
                        <div style="font-size:0.8rem;color:#94a3b8;">Frame {result.frame_idx}</div>
                        """
                    )
                card_html = "\n".join(card_blocks)
                if card_html:
                    stats_placeholder.markdown(
                        f"""
                        <div class="metric-card">
                            {card_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    stats_placeholder.empty()
                overlay_frame = result.annotated.copy()
                if overlay_lines:
                    _draw_stats_overlay(overlay_frame, overlay_lines)
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
                            use_column_width=True,
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

    progress.progress(1.0)
    st.success("Done")
    completion_display = _format_duration(total_time_sec)
    distance_display = _format_distance(total_distance_m)
    accel_display = _format_accel(peak_accel_mps2)
    decel_display = _format_accel(peak_decel_mps2)
    if "Processing info" in report_sections:
        st.write(f"Processed frames: {processed_frames}")
    if "Touches" in report_sections:
        st.write(f"Final touches — Left: **{left}**, Right: **{right}**")
    if "Time & Distance" in report_sections:
        st.write(f"Total completion time: **{completion_display}**")
        st.write(f"Total distance covered: **{distance_display}**")
    if "Acceleration" in report_sections:
        st.write(
            f"Peak acceleration / deceleration: **{accel_display} / {decel_display}**"
        )
    if annotated_video_path:
        annotated_file = Path(annotated_video_path)
        if annotated_file.exists() and annotated_file.stat().st_size > 0:
            annotated_bytes = annotated_file.read_bytes()
            annotated_file.unlink(missing_ok=True)
            st.download_button(
                "Download annotated video",
                data=annotated_bytes,
                file_name="annotated_video.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
    if "Speed" in report_sections and (last_avg_speed is not None or last_max_speed is not None):
        avg_display = f"{last_avg_speed:.1f} km/h" if last_avg_speed is not None else "--"
        max_display = f"{last_max_speed:.1f} km/h" if last_max_speed is not None else "--"
        st.write(f"Average player speed: **{avg_display}**")
        st.write(f"Maximum player speed: **{max_display}**")
    jump_display = None
    if highest_jump_m is not None:
        jump_display = f"{highest_jump_m:.2f} m"
    elif highest_jump_px is not None:
        jump_display = f"{highest_jump_px:.0f} px"
    if "Jumps" in report_sections:
        st.write(f"Total jumps: **{total_jumps}**")
        if jump_display is not None:
            st.write(f"Highest jump: **{jump_display}**")

    if "Shot log" in report_sections:
        if shot_log:
            st.subheader("Shot log")
            st.write(f"Shots detected: **{shot_count}**")
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
                    return f"{row['force_kmh']:.1f} km/h"
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
            st.info("No shot events detected.")

    if "Speed chart" in report_sections:
        if speed_points:
            st.subheader("Speed over time")
            df = pd.DataFrame(speed_points).set_index("frame")
            st.line_chart(df)
        else:
            st.info("No speed measurements available for charting.")

    report_data = {"report_sections": _report_section_list(report_sections)}
    if "Processing info" in report_sections:
        report_data["processing"] = {
            "processed_frames": processed_frames,
            "total_frames": total_frames or None,
            "input_fps": input_fps,
            "display_stride": options.display_stride,
            "max_frames": None if max_frames <= 0 else int(max_frames),
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

    trimmed_report, truncation = _trim_report_for_context(
        report_data, MAX_REPORT_CHARS
    )
    report_json = json.dumps(trimmed_report, indent=2, ensure_ascii=True)
    if truncation:
        st.warning(
            "Report JSON was trimmed to stay under ~100k tokens."
        )
    st.subheader("Report JSON")
    st.download_button(
        "Download report JSON",
        data=report_json,
        file_name="soccer_report.json",
        mime="application/json",
        use_container_width=True,
    )
    with st.expander("Preview JSON"):
        st.code(report_json, language="json")


if __name__ == "__main__":
    main()
