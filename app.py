from __future__ import annotations

from pathlib import Path
import html
import json
import tempfile
import uuid
from typing import Iterable, List, Tuple, Optional

import streamlit as st

import soccer_ai.config as cfg
from soccer_ai.calibration import build_calibration, save_calibration
from tests import registry
from tests.utils import (
    build_report_payload,
    canvas_initial_drawing,
    card,
    collect_ball_trail,
    draw_ball_overlay,
    draw_ball_trail_overlay,
    draw_event_overlay,
    draw_player_overlays,
    draw_stats_overlay,
    ensure_canvas_compat,
    extract_canvas_points,
    format_accel,
    format_accel_mps2,
    format_distance_m,
    format_speed,
    format_speed_mps,
    format_time,
    gemini_prereq_error,
    generate_gemini_response,
    inject_base_styles,
    export_annotated_video,
    prepare_canvas_frame,
    probe_video,
    read_video_frame,
    render_header,
    save_uploaded_file,
    DEFAULT_GEMINI_MAX_OUTPUT_TOKENS,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMINI_TEMPERATURE,
)


def _init_state(default_test: str, default_model: str, default_pose_model: str) -> None:
    defaults = {
        "selected_test": default_test,
        "selected_model": default_model,
        "selected_pose_model": default_pose_model,
        "uploaded_video_path": None,
        "uploaded_video_name": None,
        "uploaded_video_size": None,
        "analysis_result": None,
        "analysis_error": None,
        "analysis_runtime": None,
        "export_annotated_path": None,
        "use_metric_display": True,
        "calibration_path": None,
        "calibration_points": [],
        "calibration_frame_idx": 0,
        "calibration_canvas_key": f"calibration_canvas_{uuid.uuid4().hex}",
        "calibration_canvas_init": True,
        "calibration_video": None,
        "calibration_auto_path": None,
        "agility_use_homography": cfg.USE_HOMOGRAPHY,
        "agility_display_stride": 1,
        "agility_max_frames": 0,
        "agility_live_preview": True,
        "agility_live_stride": 3,
        "agility_preview_width": 320,
        "ball_detection_conf": float(cfg.DET_CONF),
        "ball_detection_imgsz": 640,
        "ball_hold_frames": int(cfg.BALL_HOLD_FRAMES),
        "ball_smoothing": int(cfg.BALL_SMOOTHING),
        "ball_throw_release_speed_mps": 3.0,
        "ball_throw_release_speed_px_s": 120.0,
        "ball_throw_release_window_frames": 8,
        "ball_throw_missing_ball_frames": 8,
        "ball_throw_use_player_height": True,
        "ball_throw_min_height_ratio": 0.1,
        "ball_throw_ground_ratio": 0.92,
        "ball_throw_ground_hold_frames": 3,
        "juggling_gap_threshold_s": 1.0,
        "juggling_missing_ball_frames": 10,
        "juggling_ground_ratio": 0.92,
        "juggling_ground_hold_frames": 3,
        "juggling_use_player_height": True,
        "juggling_min_height_ratio": 0.1,
        "juggling_stability_window": 10,
        "juggling_touch_window_seconds": 2.0,
        "sprint_contact_ratio": 0.9,
        "sprint_contact_cooldown_frames": 6,
        "sprint_stride_window_s": 1.0,
        "sprint_split_distances": "5,10,20,30",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _chunk(items: List[Tuple[str, object]], size: int) -> Iterable[List[Tuple[str, object]]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _reset_calibration_state() -> None:
    st.session_state["calibration_points"] = []
    st.session_state["calibration_frame_idx"] = 0
    st.session_state["calibration_canvas_key"] = f"calibration_canvas_{uuid.uuid4().hex}"
    st.session_state["calibration_canvas_init"] = True
    st.session_state["calibration_video"] = None
    st.session_state["calibration_auto_path"] = None
    st.session_state["calibration_path"] = None
    st.session_state["video_probe"] = None


def _tooltip_html(label: str, text: str) -> str:
    escaped = html.escape(text).replace("\n", "&#10;")
    return f'<span class="tooltip-chip" title="{escaped}">{label}</span>'

def _discover_models() -> tuple[List[str], List[str]]:
    search_paths = [Path("."), Path("models")]
    candidates: List[Path] = []
    for base in search_paths:
        if base.exists():
            candidates.extend(base.glob("*.pt"))

    detection: List[str] = []
    pose: List[str] = []
    for model_path in sorted(candidates):
        name = model_path.name.lower()
        if "pose" in name:
            pose.append(str(model_path))
        else:
            detection.append(str(model_path))

    if not detection:
        detection = ["Auto"]
    if not pose:
        pose = ["Auto"]
    return detection, pose


def _format_model_label(value: str) -> str:
    if value == "Auto":
        return "Auto"
    return Path(value).name


st.set_page_config(page_title="SoccerAI Tests", layout="wide", page_icon="⚽")

inject_base_styles()

test_names = registry.get_test_names()
det_models, pose_models = _discover_models()
_init_state(test_names[0], det_models[0], pose_models[0])

with st.sidebar:
    st.markdown("## Navigation")

    selected = st.radio(
        "Select Test",
        test_names,
        index=test_names.index(st.session_state.selected_test),
    )
    if selected != st.session_state.selected_test:
        st.session_state.selected_test = selected
        st.session_state.analysis_result = None
        st.session_state.analysis_error = None
        st.session_state.analysis_runtime = None
        st.session_state.export_annotated_path = None

    uploaded_file = st.file_uploader(
        "Upload Video",
        type=["mp4", "mov", "avi", "mkv"],
        accept_multiple_files=False,
    )

    st.markdown("## Models")
    selected_model = st.selectbox(
        "Detection Model",
        det_models,
        index=det_models.index(st.session_state.selected_model)
        if st.session_state.selected_model in det_models
        else 0,
        format_func=_format_model_label,
    )
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.session_state.analysis_result = None
        st.session_state.analysis_error = None

    selected_pose_model = st.selectbox(
        "Pose Model",
        pose_models,
        index=pose_models.index(st.session_state.selected_pose_model)
        if st.session_state.selected_pose_model in pose_models
        else 0,
        format_func=_format_model_label,
    )
    if selected_pose_model != st.session_state.selected_pose_model:
        st.session_state.selected_pose_model = selected_pose_model
        st.session_state.analysis_result = None
        st.session_state.analysis_error = None

    st.markdown("## Units")
    st.session_state.use_metric_display = (
        st.radio(
            "Speed & Distance",
            options=["m/s, meters", "km/h, km"],
            index=0 if st.session_state.use_metric_display else 1,
        )
        == "m/s, meters"
    )

    st.markdown("## Detection")
    ball_conf = st.slider(
        "Ball detection sensitivity",
        min_value=0.05,
        max_value=0.6,
        value=float(st.session_state.ball_detection_conf),
        step=0.01,
        help="Lower = more sensitive (detects smaller/blurred balls) but may add false positives.",
    )
    if ball_conf != st.session_state.ball_detection_conf:
        st.session_state.ball_detection_conf = ball_conf
        st.session_state.analysis_result = None
        st.session_state.analysis_error = None

    ball_imgsz = st.select_slider(
        "Detection resolution (imgsz)",
        options=[320, 480, 640, 800, 960, 1280],
        value=int(st.session_state.ball_detection_imgsz),
        help="Higher values improve small-ball detection but run slower.",
    )
    if ball_imgsz != st.session_state.ball_detection_imgsz:
        st.session_state.ball_detection_imgsz = ball_imgsz
        st.session_state.analysis_result = None
        st.session_state.analysis_error = None

    ball_hold = st.slider(
        "Ball hold frames",
        min_value=0,
        max_value=10,
        value=int(st.session_state.ball_hold_frames),
        step=1,
        help="Keep last ball position for this many missing frames.",
    )
    if ball_hold != st.session_state.ball_hold_frames:
        st.session_state.ball_hold_frames = ball_hold
        st.session_state.analysis_result = None
        st.session_state.analysis_error = None

    ball_smoothing = st.slider(
        "Ball smoothing window",
        min_value=1,
        max_value=12,
        value=int(st.session_state.ball_smoothing),
        step=1,
        help="Smooth ball detections across this many frames.",
    )
    if ball_smoothing != st.session_state.ball_smoothing:
        st.session_state.ball_smoothing = ball_smoothing
        st.session_state.analysis_result = None
        st.session_state.analysis_error = None


    if uploaded_file is not None:
        is_new_file = (
            uploaded_file.name != st.session_state.uploaded_video_name
            or uploaded_file.size != st.session_state.uploaded_video_size
        )
        if is_new_file:
            saved_path = save_uploaded_file(uploaded_file)
            st.session_state.uploaded_video_path = saved_path
            st.session_state.uploaded_video_name = uploaded_file.name
            st.session_state.uploaded_video_size = uploaded_file.size
            st.session_state.analysis_result = None
            st.session_state.analysis_error = None
            st.session_state.analysis_runtime = None
            st.session_state.export_annotated_path = None
            _reset_calibration_state()

    if st.button("Clear Video"):
        st.session_state.uploaded_video_path = None
        st.session_state.uploaded_video_name = None
        st.session_state.uploaded_video_size = None
        st.session_state.analysis_result = None
        st.session_state.analysis_error = None
        st.session_state.analysis_runtime = None
        st.session_state.export_annotated_path = None
        _reset_calibration_state()

    # Agility settings moved into the main page (not sidebar)

definition = registry.get_test_definition(st.session_state.selected_test)
video_path = st.session_state.uploaded_video_path
has_video = bool(video_path and Path(video_path).exists())
ready_to_analyze = bool(definition and has_video)

render_header(
    title="SoccerAI",
    subtitle="Select a test, upload a video, and generate structured placeholder results.",
    selected_test=definition.name,
    has_video=has_video,
    ready=ready_to_analyze,
)

with card("Video"):
    st.markdown(f"**Selected Test:** {definition.name}")
    st.markdown(f"<span class='muted'>{definition.description}</span>", unsafe_allow_html=True)

    if has_video:
        st.success(f"Video loaded: {st.session_state.uploaded_video_name}")
        st.caption(f"Stored at {video_path}")
        probe = probe_video(video_path)
        st.session_state["video_probe"] = probe
        if probe:
            details = []
            if probe.get("frame_count"):
                details.append(f"{probe['frame_count']} frames")
            if probe.get("fps"):
                details.append(f"{probe['fps']:.1f} fps")
            if details:
                st.caption(" · ".join(details))

    else:
        st.info("Upload a video to unlock preview and analysis.")

if st.session_state.selected_test in (
    "Agility",
    "Dribbling",
    "Sprint",
    "Counter Movement Jump (CMJ)",
    "Drop Jump",
    "Endurance",
):
    if st.session_state.selected_test == "Agility":
        settings_title = "Agility Settings"
    elif st.session_state.selected_test == "Dribbling":
        settings_title = "Dribbling Settings"
    elif st.session_state.selected_test == "Sprint":
        settings_title = "Sprint Settings"
    elif st.session_state.selected_test == "Counter Movement Jump (CMJ)":
        settings_title = "CMJ Settings"
    elif st.session_state.selected_test == "Endurance":
        settings_title = "Endurance Settings"
    else:
        settings_title = "Drop Jump Settings"
    with card(settings_title, "Calibration and analysis controls."):
        col_left, col_right = st.columns(2)
        with col_left:
            calibration_file = st.file_uploader(
                "Upload calibration (JSON)",
                type=["json"],
                key="agility_calibration_upload",
                help="Use a saved cone calibration to enable meter-based distance.",
            )
            if calibration_file is not None:
                tmp_calib = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
                tmp_calib.write(calibration_file.read())
                tmp_calib.close()
                st.session_state["calibration_path"] = tmp_calib.name
            if st.session_state.get("calibration_path"):
                st.success(
                    f"Calibration active: {Path(st.session_state['calibration_path']).name}"
                )
        with col_right:
            st.session_state.agility_use_homography = st.checkbox(
                "Use homography for distance/speed",
                value=st.session_state.agility_use_homography,
                help="Enable meter-based distance using calibration.",
            )
            st.session_state.agility_display_stride = st.slider(
                "Display stride (emit every Nth frame)",
                min_value=1,
                max_value=5,
                value=int(st.session_state.agility_display_stride),
                step=1,
            )
            st.session_state.agility_max_frames = st.number_input(
                "Max frames to process (0 = full video)",
                min_value=0,
                step=50,
                value=int(st.session_state.agility_max_frames),
            )

        col_live_a, _ = st.columns(2)
        with col_live_a:
            st.session_state.agility_live_preview = st.checkbox(
                "Show live preview",
                value=st.session_state.agility_live_preview,
            )
        st.session_state.agility_preview_width = st.slider(
            "Preview width (px)",
            min_value=240,
            max_value=520,
            value=int(st.session_state.agility_preview_width),
            step=20,
        )

        if st.session_state.selected_test == "Sprint":
            st.markdown("#### Stride Detection")
            col_stride_a, col_stride_b = st.columns(2)
            with col_stride_a:
                st.session_state.sprint_contact_ratio = st.slider(
                    "Foot contact ratio",
                    min_value=0.7,
                    max_value=0.98,
                    value=float(st.session_state.sprint_contact_ratio),
                    step=0.02,
                    help="Higher = foot must be closer to ground in the bbox to count a contact.",
                )
                st.session_state.sprint_contact_cooldown_frames = st.slider(
                    "Contact cooldown (frames)",
                    min_value=2,
                    max_value=20,
                    value=int(st.session_state.sprint_contact_cooldown_frames),
                    step=1,
                    help="Minimum frames between consecutive foot contacts.",
                )
            with col_stride_b:
                st.session_state.sprint_stride_window_s = st.slider(
                    "Stride window (seconds)",
                    min_value=0.4,
                    max_value=2.0,
                    value=float(st.session_state.sprint_stride_window_s),
                    step=0.1,
                    help="Window used to compute stride frequency.",
                )
                st.session_state.sprint_split_distances = st.text_input(
                    "Split distances (m)",
                    value=st.session_state.sprint_split_distances,
                    help="Comma-separated distances, e.g. 5,10,20,30",
                )

        if has_video:
            enable_calibration = st.checkbox(
                "Create calibration from cones",
                value=False,
                key="agility_calibration_toggle",
            )
            if not enable_calibration:
                st.caption("Enable to mark four cone points for homography.")
            if enable_calibration:
                with st.expander("Calibration (cones)", expanded=True):
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
                            key="calibration_width_m",
                        )
                    with col_b:
                        height_m = st.number_input(
                            "Rectangle height (meters)",
                            min_value=1.0,
                            value=float(cfg.CALIB_RECT_HEIGHT_M),
                            step=0.5,
                            key="calibration_height_m",
                        )

                    probe = st.session_state.get("video_probe")
                    total_frames = probe.get("frame_count") if probe else 0
                    frame_idx = 0
                    if total_frames:
                        frame_idx = st.slider(
                            "Pick calibration frame",
                            min_value=0,
                            max_value=max(0, int(total_frames) - 1),
                            value=0,
                            step=1,
                            key="calibration_frame_picker",
                        )

                    frame = read_video_frame(video_path, frame_idx)
                    if frame is None:
                        st.error("Unable to read the calibration frame.")
                    else:
                        if st.session_state.get("calibration_video") != video_path:
                            st.session_state["calibration_video"] = video_path
                            st.session_state["calibration_points"] = []
                            st.session_state["calibration_path"] = None
                            st.session_state["calibration_auto_path"] = None
                            st.session_state["calibration_frame_idx"] = frame_idx
                            st.session_state["calibration_canvas_key"] = (
                                f"calibration_canvas_{uuid.uuid4().hex}"
                            )
                            st.session_state["calibration_canvas_init"] = True

                        if st.session_state.get("calibration_frame_idx") != frame_idx:
                            st.session_state["calibration_frame_idx"] = frame_idx
                            st.session_state["calibration_points"] = []
                            st.session_state["calibration_auto_path"] = None
                            st.session_state["calibration_canvas_key"] = (
                                f"calibration_canvas_{uuid.uuid4().hex}"
                            )
                            st.session_state["calibration_canvas_init"] = True

                        try:
                            canvas_image, scale = prepare_canvas_frame(frame)
                        except RuntimeError as exc:
                            st.error(str(exc))
                            canvas_image = None
                            scale = 1.0

                        if canvas_image is not None:
                            points = st.session_state.get("calibration_points", [])
                            mode = st.radio(
                                "Canvas mode",
                                options=["Add points", "Adjust points"],
                                horizontal=True,
                                index=0 if len(points) < 4 else 1,
                                key="calibration_mode",
                            )

                            try:
                                ensure_canvas_compat()
                                from streamlit_drawable_canvas import st_canvas
                            except Exception:
                                st.warning(
                                    "Install `streamlit-drawable-canvas` to use the calibration overlay."
                                )
                                st.code("pip install streamlit-drawable-canvas")
                                st.stop()

                            canvas_key = st.session_state.get(
                                "calibration_canvas_key", "calibration_canvas"
                            )
                            canvas_init = st.session_state.get("calibration_canvas_init", True)
                            initial = (
                                canvas_initial_drawing(points)
                                if canvas_init and points
                                else None
                            )
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
                                points = extract_canvas_points(canvas_result.json_data)
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
                                if st.button("Reset points", key="calibration_reset"):
                                    st.session_state["calibration_points"] = []
                                    st.session_state["calibration_auto_path"] = None
                                    st.session_state["calibration_path"] = None
                                    st.session_state["calibration_canvas_key"] = (
                                        f"calibration_canvas_{uuid.uuid4().hex}"
                                    )
                                    st.session_state["calibration_canvas_init"] = True
                            with col_right:
                                if st.button(
                                    "Save calibration",
                                    type="primary",
                                    key="calibration_save",
                                ):
                                    if len(points_display) != 4:
                                        st.error("Select exactly 4 points before saving.")
                                    else:
                                        image_points = [
                                            (p[0] / scale, p[1] / scale)
                                            for p in points_display
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
                                        st.success(
                                            "Calibration saved. It will be used for analysis."
                                        )
                                        calib_bytes = Path(tmp_calib.name).read_bytes()
                                        st.download_button(
                                            "Download calibration JSON",
                                            data=calib_bytes,
                                            file_name="cone_calibration.json",
                                            mime="application/json",
                                        )

                            if len(points_display) == 4:
                                image_points = [
                                    (p[0] / scale, p[1] / scale) for p in points_display
                                ]
                                calibration = build_calibration(
                                    image_points=image_points,
                                    field_width_m=width_m,
                                    field_height_m=height_m,
                                    reorder=True,
                                )
                                auto_path = st.session_state.get("calibration_auto_path")
                                if not auto_path:
                                    tmp_calib = tempfile.NamedTemporaryFile(
                                        delete=False, suffix=".json"
                                    )
                                    tmp_calib.close()
                                    auto_path = tmp_calib.name
                                    st.session_state["calibration_auto_path"] = auto_path
                                save_calibration(auto_path, calibration)
                                if st.session_state.get("calibration_path") is None:
                                    st.session_state["calibration_path"] = auto_path
                                st.success("Calibration active for analysis.")
        else:
            st.info("Upload a video to enable calibration tools.")

if st.session_state.selected_test == "Juggling":
    with card("Juggling Settings", "Touch cadence and control thresholds."):
        col_left, col_right = st.columns(2)
        with col_left:
            st.session_state.juggling_gap_threshold_s = st.slider(
                "Streak gap threshold (seconds)",
                min_value=0.4,
                max_value=2.0,
                value=float(st.session_state.juggling_gap_threshold_s),
                step=0.1,
                help="Break the streak if time between touches exceeds this.",
            )
            st.session_state.juggling_missing_ball_frames = st.slider(
                "Ball missing frames before drop",
                min_value=3,
                max_value=30,
                value=int(st.session_state.juggling_missing_ball_frames),
                step=1,
                help="Break the streak if the ball disappears for this many frames.",
            )
            st.session_state.juggling_stability_window = st.slider(
                "Stability window (touches)",
                min_value=4,
                max_value=30,
                value=int(st.session_state.juggling_stability_window),
                step=1,
                help="Rolling window size for stability metrics.",
            )
            st.session_state.juggling_touch_window_seconds = st.slider(
                "Touch rate window (seconds)",
                min_value=1.0,
                max_value=5.0,
                value=float(st.session_state.juggling_touch_window_seconds),
                step=0.5,
                help="Window used to compute live touch rate.",
            )
        with col_right:
            st.session_state.juggling_use_player_height = st.checkbox(
                "Use player height for ball height",
                value=st.session_state.juggling_use_player_height,
                help="Estimate ball height using player bounding box scale.",
            )
            st.session_state.juggling_min_height_ratio = st.slider(
                "Min height ratio (drop)",
                min_value=0.02,
                max_value=0.3,
                value=float(st.session_state.juggling_min_height_ratio),
                step=0.01,
                help="Drop if ball height (relative to player height) falls below this.",
                disabled=not st.session_state.juggling_use_player_height,
            )
            st.session_state.juggling_ground_ratio = st.slider(
                "Ground Y ratio (image)",
                min_value=0.85,
                max_value=0.98,
                value=float(st.session_state.juggling_ground_ratio),
                step=0.01,
                help="Drop if ball center is near the bottom of the frame.",
                disabled=st.session_state.juggling_use_player_height,
            )
            st.session_state.juggling_ground_hold_frames = st.slider(
                "Ground hold frames",
                min_value=1,
                max_value=10,
                value=int(st.session_state.juggling_ground_hold_frames),
                step=1,
                help="Require this many near-ground frames to drop the streak.",
            )

        st.session_state.agility_display_stride = st.slider(
            "Display stride (emit every Nth frame)",
            min_value=1,
            max_value=5,
            value=int(st.session_state.agility_display_stride),
            step=1,
        )
        st.session_state.agility_max_frames = st.number_input(
            "Max frames to process (0 = full video)",
            min_value=0,
            step=50,
            value=int(st.session_state.agility_max_frames),
        )
        st.session_state.agility_live_preview = st.checkbox(
            "Show live preview",
            value=st.session_state.agility_live_preview,
        )
        st.session_state.agility_preview_width = st.slider(
            "Preview width (px)",
            min_value=240,
            max_value=520,
            value=int(st.session_state.agility_preview_width),
            step=20,
        )

if st.session_state.selected_test == "Ball Throw":
    with card("Ball Throw Settings", "Release detection and tracking thresholds."):
        col_left, col_right = st.columns(2)
        with col_left:
            st.session_state.ball_throw_release_speed_mps = st.slider(
                "Release speed threshold (m/s)",
                min_value=0.5,
                max_value=15.0,
                value=float(st.session_state.ball_throw_release_speed_mps),
                step=0.5,
                help="Used when homography is available.",
            )
            st.session_state.ball_throw_release_speed_px_s = st.slider(
                "Release speed threshold (px/s)",
                min_value=20.0,
                max_value=400.0,
                value=float(st.session_state.ball_throw_release_speed_px_s),
                step=10.0,
                help="Fallback when homography is unavailable.",
            )
            st.session_state.ball_throw_release_window_frames = st.slider(
                "Release window (frames)",
                min_value=2,
                max_value=20,
                value=int(st.session_state.ball_throw_release_window_frames),
                step=1,
                help="Max frames to capture peak release speed.",
            )
        with col_right:
            st.session_state.ball_throw_missing_ball_frames = st.slider(
                "Ball missing frames",
                min_value=2,
                max_value=30,
                value=int(st.session_state.ball_throw_missing_ball_frames),
                step=1,
                help="Reset ball tracking when missing this long.",
            )
            st.session_state.ball_throw_use_player_height = st.checkbox(
                "Use player height for ball height",
                value=st.session_state.ball_throw_use_player_height,
            )
            st.session_state.ball_throw_min_height_ratio = st.slider(
                "Min height ratio (drop)",
                min_value=0.02,
                max_value=0.3,
                value=float(st.session_state.ball_throw_min_height_ratio),
                step=0.01,
                help="Height ratio below this is considered near-ground.",
                disabled=not st.session_state.ball_throw_use_player_height,
            )
            st.session_state.ball_throw_ground_ratio = st.slider(
                "Ground Y ratio (image)",
                min_value=0.85,
                max_value=0.98,
                value=float(st.session_state.ball_throw_ground_ratio),
                step=0.01,
                help="Fallback ground threshold when player height is unavailable.",
                disabled=st.session_state.ball_throw_use_player_height,
            )
            st.session_state.ball_throw_ground_hold_frames = st.slider(
                "Ground hold frames",
                min_value=1,
                max_value=10,
                value=int(st.session_state.ball_throw_ground_hold_frames),
                step=1,
                help="Require this many near-ground frames to mark landing.",
            )

        st.session_state.agility_display_stride = st.slider(
            "Display stride (emit every Nth frame)",
            min_value=1,
            max_value=5,
            value=int(st.session_state.agility_display_stride),
            step=1,
        )
        st.session_state.agility_max_frames = st.number_input(
            "Max frames to process (0 = full video)",
            min_value=0,
            step=50,
            value=int(st.session_state.agility_max_frames),
        )
        st.session_state.agility_live_preview = st.checkbox(
            "Show live preview",
            value=st.session_state.agility_live_preview,
        )
        st.session_state.agility_preview_width = st.slider(
            "Preview width (px)",
            min_value=240,
            max_value=520,
            value=int(st.session_state.agility_preview_width),
            step=20,
        )

with card("Analysis", "Run the placeholder pipeline and inspect outputs."):
    analyze_clicked = st.button("Analyze", disabled=not ready_to_analyze)
    live_placeholders = {}
    if st.session_state.selected_test in (
        "Agility",
        "Dribbling",
        "Juggling",
        "Ball Throw",
        "Sprint",
        "Counter Movement Jump (CMJ)",
        "Drop Jump",
        "Endurance",
    ):
        st.markdown("### Live Dashboard")
        live_container = st.container()
        with live_container:
            col_preview, col_metrics = st.columns([1, 1.4])
            with col_preview:
                live_frame_placeholder = st.empty()
                tooltip_placeholder = st.empty()
            with col_metrics:
                metrics_placeholder = st.empty()
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
        live_placeholders = {
            "frame": live_frame_placeholder,
            "tooltip": tooltip_placeholder,
            "metrics": metrics_placeholder,
            "progress": progress_bar,
            "status": status_placeholder,
        }

    if analyze_clicked:
        with st.spinner("Analyzing video..."):
            try:
                runtime_store = {
                    "frame_records": [],
                    "snapshots": [],
                    "shot_log": [],
                    "display_stride": st.session_state.get("agility_display_stride", 1),
                }
                st.session_state["analysis_runtime"] = runtime_store
                probe = st.session_state.get("video_probe") or probe_video(video_path)
                total_frames = probe.get("frame_count") if probe else None
                max_frames_setting = st.session_state.get("agility_max_frames")
                if total_frames and max_frames_setting:
                    try:
                        max_frames_val = int(max_frames_setting)
                        if max_frames_val > 0:
                            total_frames = min(total_frames, max_frames_val)
                    except (TypeError, ValueError):
                        pass

                def live_callback(payload: dict) -> None:
                    if not live_placeholders:
                        return
                    use_metric_display = st.session_state.get("use_metric_display", True)
                    metrics = payload.get("metrics", {})

                    if st.session_state.selected_test == "Dribbling":
                        with live_placeholders["metrics"].container():
                            row_one = st.columns(3)
                            row_one[0].metric(
                                "Total Time", format_time(metrics.get("total_time_s"))
                            )
                            row_one[1].metric(
                                "Total Distance",
                                format_distance_m(
                                    metrics.get("total_distance_m"), use_metric_display
                                ),
                            )
                            row_one[2].metric(
                                "Touches",
                                "--" if metrics.get("touch_count") is None else f"{metrics.get('touch_count')}",
                            )

                            row_two = st.columns(3)
                            row_two[0].metric(
                                "Touch Rate",
                                "--" if metrics.get("touch_rate") is None else f"{metrics.get('touch_rate'):.2f}/s",
                            )
                            row_two[1].metric(
                                "Average Speed",
                                format_speed_mps(
                                    metrics.get("avg_speed_mps"), use_metric_display
                                ),
                            )
                            row_two[2].metric(
                                "Max Speed",
                                format_speed_mps(
                                    metrics.get("max_speed_mps"), use_metric_display
                                ),
                            )

                            row_three = st.columns(3)
                            left_touches = metrics.get("left_touches")
                            right_touches = metrics.get("right_touches")
                            row_three[0].metric(
                                "Left Touches",
                                "--" if left_touches is None else f"{left_touches}",
                            )
                            row_three[1].metric(
                                "Right Touches",
                                "--" if right_touches is None else f"{right_touches}",
                            )
                            touches_per_min_live = metrics.get("touches_per_min")
                            row_three[2].metric(
                                "Touches / Min",
                                "--"
                                if touches_per_min_live is None
                                else f"{touches_per_min_live:.1f}",
                            )
                    elif st.session_state.selected_test == "Sprint":
                        with live_placeholders["metrics"].container():
                            row_one = st.columns(3)
                            row_one[0].metric(
                                "Total Time", format_time(metrics.get("total_time_s"))
                            )
                            row_one[1].metric(
                                "Total Distance",
                                format_distance_m(
                                    metrics.get("total_distance_m"), use_metric_display
                                ),
                            )
                            row_one[2].metric(
                                "Average Speed",
                                format_speed_mps(
                                    metrics.get("avg_speed_mps"), use_metric_display
                                ),
                            )

                            row_two = st.columns(3)
                            row_two[0].metric(
                                "Max Speed",
                                format_speed_mps(
                                    metrics.get("max_speed_mps"), use_metric_display
                                ),
                            )
                            row_two[1].metric(
                                "Peak Accel",
                                format_accel_mps2(metrics.get("peak_accel_mps2")),
                            )
                            row_two[2].metric(
                                "Stride Rate",
                                "--"
                                if metrics.get("stride_rate_hz") is None
                                else f"{metrics.get('stride_rate_hz'):.2f} Hz",
                            )
                    elif st.session_state.selected_test == "Endurance":
                        with live_placeholders["metrics"].container():
                            row_one = st.columns(3)
                            row_one[0].metric(
                                "Total Time", format_time(metrics.get("total_time_s"))
                            )
                            row_one[1].metric(
                                "Total Distance",
                                format_distance_m(
                                    metrics.get("total_distance_m"), use_metric_display
                                ),
                            )
                            row_one[2].metric(
                                "Average Speed",
                                format_speed_mps(
                                    metrics.get("avg_speed_mps"), use_metric_display
                                ),
                            )

                            row_two = st.columns(3)
                            row_two[0].metric(
                                "Max Speed",
                                format_speed_mps(
                                    metrics.get("max_speed_mps"), use_metric_display
                                ),
                            )
                            row_two[1].metric(
                                "Total Turns",
                                "--"
                                if metrics.get("total_turns") is None
                                else f"{metrics.get('total_turns')}",
                            )
                            turns_per_min_live = metrics.get("turns_per_min")
                            row_two[2].metric(
                                "Turns / Min",
                                "--"
                                if turns_per_min_live is None
                                else f"{turns_per_min_live:.1f}",
                            )

                            row_three = st.columns(3)
                            hr_live = metrics.get("heart_rate_bpm")
                            row_three[0].metric(
                                "Heart Rate",
                                "--" if hr_live is None else f"{hr_live:.0f} bpm",
                            )
                            fatigue_live = metrics.get("fatigue_index")
                            row_three[1].metric(
                                "Fatigue",
                                "--"
                                if fatigue_live is None
                                else f"{fatigue_live * 100:.0f}%",
                            )
                            turn_rate_live = metrics.get("turn_rate_deg_s")
                            row_three[2].metric(
                                "Turn Rate",
                                "--"
                                if turn_rate_live is None
                                else f"{turn_rate_live:.1f} deg/s",
                            )
                    elif st.session_state.selected_test == "Ball Throw":
                        with live_placeholders["metrics"].container():
                            row_one = st.columns(3)
                            row_one[0].metric(
                                "Total Time", format_time(metrics.get("total_time_s"))
                            )
                            release_speed = metrics.get("release_speed_mps")
                            release_speed_px = metrics.get("release_speed_px_s")
                            release_display = (
                                format_speed_mps(release_speed, use_metric_display)
                                if release_speed is not None
                                else (
                                    "--"
                                    if release_speed_px is None
                                    else f"{release_speed_px:.1f} px/s"
                                )
                            )
                            row_one[1].metric("Release Speed", release_display)
                            release_angle = metrics.get("release_angle_deg")
                            row_one[2].metric(
                                "Release Angle",
                                "--" if release_angle is None else f"{release_angle:.1f}°",
                            )

                            row_two = st.columns(3)
                            hand_force = metrics.get("hand_force_mps2")
                            hand_force_px = metrics.get("hand_force_px_s2")
                            hand_force_display = (
                                format_accel_mps2(hand_force)
                                if hand_force is not None
                                else (
                                    "--"
                                    if hand_force_px is None
                                    else f"{hand_force_px:.1f} px/s²"
                                )
                            )
                            row_two[0].metric("Hand Force", hand_force_display)
                            throw_dist = metrics.get("throw_distance_m")
                            throw_dist_px = metrics.get("throw_distance_px")
                            throw_display = (
                                format_distance_m(throw_dist, use_metric_display)
                                if throw_dist is not None
                                else ("--" if throw_dist_px is None else f"{throw_dist_px:.1f} px")
                            )
                            row_two[1].metric("Throw Distance", throw_display)
                            arm_angle = metrics.get("dominant_arm_angle_deg")
                            row_two[2].metric(
                                "Arm Angle",
                                "--" if arm_angle is None else f"{arm_angle:.1f}°",
                            )
                    elif st.session_state.selected_test == "Juggling":
                        with live_placeholders["metrics"].container():
                            row_one = st.columns(3)
                            row_one[0].metric(
                                "Total Time", format_time(metrics.get("total_time_s"))
                            )
                            row_one[1].metric(
                                "Touches",
                                "--"
                                if metrics.get("touch_count") is None
                                else f"{metrics.get('touch_count')}",
                            )
                            row_one[2].metric(
                                "Max Streak",
                                "--"
                                if metrics.get("max_consecutive_touches") is None
                                else f"{metrics.get('max_consecutive_touches')}",
                            )

                            row_two = st.columns(3)
                            row_two[0].metric(
                                "Left Touches",
                                "--"
                                if metrics.get("left_touches") is None
                                else f"{metrics.get('left_touches')}",
                            )
                            row_two[1].metric(
                                "Right Touches",
                                "--"
                                if metrics.get("right_touches") is None
                                else f"{metrics.get('right_touches')}",
                            )
                            touches_per_min_live = metrics.get("touches_per_min")
                            row_two[2].metric(
                                "Touches / Min",
                                "--"
                                if touches_per_min_live is None
                                else f"{touches_per_min_live:.1f}",
                            )
                    elif st.session_state.selected_test == "Counter Movement Jump (CMJ)":
                        with live_placeholders["metrics"].container():
                            row_one = st.columns(3)
                            total_jumps = metrics.get("total_jumps")
                            row_one[0].metric(
                                "Total Jumps",
                                "--" if total_jumps is None else f"{total_jumps}",
                            )
                            highest_display = "--"
                            if metrics.get("highest_jump_m") is not None:
                                highest_display = f"{metrics.get('highest_jump_m'):.2f} m"
                            elif metrics.get("highest_jump_px") is not None:
                                highest_display = f"{metrics.get('highest_jump_px'):.0f} px"
                            row_one[1].metric("Highest Jump", highest_display)

                            current_display = "--"
                            if metrics.get("current_height_m") is not None:
                                current_display = f"{metrics.get('current_height_m'):.2f} m"
                            elif metrics.get("current_height_px") is not None:
                                current_display = f"{metrics.get('current_height_px'):.0f} px"
                            row_one[2].metric("Current Height", current_display)

                            row_two = st.columns(3)
                            last_display = "--"
                            if metrics.get("last_jump_height_m") is not None:
                                last_display = f"{metrics.get('last_jump_height_m'):.2f} m"
                            elif metrics.get("last_jump_height_px") is not None:
                                last_display = f"{metrics.get('last_jump_height_px'):.0f} px"
                            row_two[0].metric("Last Jump", last_display)

                            flight_time = metrics.get("last_flight_time_s")
                            row_two[1].metric(
                                "Last Flight",
                                "--" if flight_time is None else f"{flight_time:.2f} s",
                            )
                            row_two[2].metric(
                                "Airborne",
                                "Yes" if metrics.get("jump_active") else "No",
                            )
                    elif st.session_state.selected_test == "Drop Jump":
                        with live_placeholders["metrics"].container():
                            row_one = st.columns(3)
                            total_jumps = metrics.get("total_jumps")
                            row_one[0].metric(
                                "Total Jumps",
                                "--" if total_jumps is None else f"{total_jumps}",
                            )
                            contact_time = metrics.get("last_contact_time_s")
                            row_one[1].metric(
                                "Last Contact",
                                "--" if contact_time is None else f"{contact_time:.2f} s",
                            )
                            last_height_display = "--"
                            if metrics.get("last_jump_height_m") is not None:
                                last_height_display = f"{metrics.get('last_jump_height_m'):.2f} m"
                            elif metrics.get("last_jump_height_px") is not None:
                                last_height_display = f"{metrics.get('last_jump_height_px'):.0f} px"
                            row_one[2].metric("Last Jump", last_height_display)

                            row_two = st.columns(3)
                            rsi_val = metrics.get("last_rsi")
                            row_two[0].metric(
                                "Last RSI",
                                "--" if rsi_val is None else f"{rsi_val:.2f}",
                            )
                            phase = metrics.get("phase")
                            row_two[1].metric(
                                "Phase",
                                "--" if not phase else str(phase).replace("_", " ").title(),
                            )
                            current_display = "--"
                            if metrics.get("current_height_m") is not None:
                                current_display = f"{metrics.get('current_height_m'):.2f} m"
                            elif metrics.get("current_height_px") is not None:
                                current_display = f"{metrics.get('current_height_px'):.0f} px"
                            row_two[2].metric("Current Height", current_display)
                    else:
                        with live_placeholders["metrics"].container():
                            row_one = st.columns(3)
                            row_one[0].metric(
                                "Total Time", format_time(metrics.get("total_time_s"))
                            )
                            row_one[1].metric(
                                "Total Distance",
                                format_distance_m(
                                    metrics.get("total_distance_m"), use_metric_display
                                ),
                            )
                            row_one[2].metric(
                                "Average Speed",
                                format_speed_mps(
                                    metrics.get("avg_speed_mps"), use_metric_display
                                ),
                            )

                            row_two = st.columns(3)
                            row_two[0].metric(
                                "Max Speed",
                                format_speed_mps(
                                    metrics.get("max_speed_mps"), use_metric_display
                                ),
                            )
                            row_two[1].metric(
                                "Peak Accel",
                                format_accel_mps2(metrics.get("peak_accel_mps2")),
                            )
                            row_two[2].metric(
                                "Peak Decel",
                                format_accel_mps2(metrics.get("peak_decel_mps2")),
                            )

                    if st.session_state.agility_live_preview:
                        frame_bgr = payload.get("frame_bgr")
                        if frame_bgr is not None:
                            try:
                                import cv2  # type: ignore

                                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                                live_placeholders["frame"].image(
                                    frame_rgb,
                                    caption=f"Live frame {payload.get('frame_idx')}",
                                    channels="RGB",
                                    width=st.session_state.agility_preview_width,
                                )
                            except Exception:
                                live_placeholders["frame"].info(
                                    "Live preview unavailable (OpenCV missing)."
                                )
                    else:
                        live_placeholders["frame"].info(
                            "Live preview disabled in settings."
                        )

                    if st.session_state.selected_test == "Juggling":
                        touch_tail = payload.get("touch_count_tail")
                        stability_tail = payload.get("control_stability_tail")
                        height_tail = payload.get("ball_height_tail")
                        tooltip_sections = []
                        if touch_tail is not None and not touch_tail.empty:
                            tooltip_sections.append(
                                "Touch count (latest):\n" + touch_tail.to_string(index=False)
                            )
                        if stability_tail is not None and not stability_tail.empty:
                            tooltip_sections.append(
                                "Control stability (latest):\n"
                                + stability_tail.to_string(index=False)
                            )
                        if height_tail is not None and not height_tail.empty:
                            tooltip_sections.append(
                                "Ball height (latest):\n" + height_tail.to_string(index=False)
                            )
                        if tooltip_sections:
                            tooltip_text = "\n\n".join(tooltip_sections)
                            live_placeholders["tooltip"].markdown(
                                f'<div class="tooltip-row">{_tooltip_html("Juggling Matrices", tooltip_text)}</div>',
                                unsafe_allow_html=True,
                            )
                    elif st.session_state.selected_test == "Counter Movement Jump (CMJ)":
                        height_tail = payload.get("jump_height_tail")
                        force_tail = payload.get("force_time_tail")
                        landing_tail = payload.get("landing_stability_tail")
                        tooltip_sections = []
                        if height_tail is not None and not height_tail.empty:
                            height_display = height_tail.copy()
                            if use_metric_display and "height_m" in height_display:
                                height_display = height_display.drop(
                                    columns=[col for col in ["height_px"] if col in height_display],
                                    errors="ignore",
                                )
                            elif not use_metric_display and "height_px" in height_display:
                                height_display = height_display.drop(
                                    columns=[col for col in ["height_m"] if col in height_display],
                                    errors="ignore",
                                )
                            tooltip_sections.append(
                                "Jump height (latest):\n" + height_display.to_string(index=False)
                            )
                        if force_tail is not None and not force_tail.empty:
                            force_display = force_tail.copy()
                            if use_metric_display and "accel_mps2" in force_display:
                                force_display = force_display.drop(
                                    columns=[col for col in ["accel_px_s2"] if col in force_display],
                                    errors="ignore",
                                )
                            elif not use_metric_display and "accel_px_s2" in force_display:
                                force_display = force_display.drop(
                                    columns=[col for col in ["accel_mps2"] if col in force_display],
                                    errors="ignore",
                                )
                            tooltip_sections.append(
                                "Force-time proxy (latest):\n" + force_display.to_string(index=False)
                            )
                        if landing_tail is not None and not landing_tail.empty:
                            tooltip_sections.append(
                                "Landing stability (latest):\n"
                                + landing_tail.to_string(index=False)
                            )
                        if tooltip_sections:
                            tooltip_text = "\n\n".join(tooltip_sections)
                            live_placeholders["tooltip"].markdown(
                                f'<div class="tooltip-row">{_tooltip_html("CMJ Matrices", tooltip_text)}</div>',
                                unsafe_allow_html=True,
                            )
                    elif st.session_state.selected_test == "Drop Jump":
                        ground_tail = payload.get("ground_contact_tail")
                        reactive_tail = payload.get("reactive_strength_tail")
                        landing_tail = payload.get("landing_force_tail")
                        tooltip_sections = []
                        if ground_tail is not None and not ground_tail.empty:
                            tooltip_sections.append(
                                "Ground contact (latest):\n" + ground_tail.to_string(index=False)
                            )
                        if reactive_tail is not None and not reactive_tail.empty:
                            tooltip_sections.append(
                                "Reactive strength (latest):\n"
                                + reactive_tail.to_string(index=False)
                            )
                        if landing_tail is not None and not landing_tail.empty:
                            tooltip_sections.append(
                                "Landing force (latest):\n"
                                + landing_tail.to_string(index=False)
                            )
                        if tooltip_sections:
                            tooltip_text = "\n\n".join(tooltip_sections)
                            live_placeholders["tooltip"].markdown(
                                f'<div class="tooltip-row">{_tooltip_html("Drop Jump Matrices", tooltip_text)}</div>',
                                unsafe_allow_html=True,
                            )
                    elif st.session_state.selected_test == "Sprint":
                        speed_tail = payload.get("speed_profile_tail")
                        stride_tail = payload.get("stride_rate_tail")
                        tooltip_sections = []
                        if speed_tail is not None and not speed_tail.empty:
                            speed_tail_display = speed_tail.copy()
                            if not use_metric_display:
                                if "speed_mps" in speed_tail_display:
                                    speed_tail_display["speed_kmh"] = (
                                        speed_tail_display.pop("speed_mps") * 3.6
                                    )
                                if "distance_m" in speed_tail_display:
                                    speed_tail_display["distance_km"] = (
                                        speed_tail_display.pop("distance_m") / 1000.0
                                    )
                            tooltip_sections.append(
                                "Speed profile (latest):\n"
                                + speed_tail_display.to_string(index=False)
                            )
                        if stride_tail is not None and not stride_tail.empty:
                            tooltip_sections.append(
                                "Stride rate (latest):\n"
                                + stride_tail.to_string(index=False)
                            )
                        if tooltip_sections:
                            tooltip_text = "\n\n".join(tooltip_sections)
                            live_placeholders["tooltip"].markdown(
                                f'<div class="tooltip-row">{_tooltip_html("Sprint Matrices", tooltip_text)}</div>',
                                unsafe_allow_html=True,
                            )
                    elif st.session_state.selected_test == "Ball Throw":
                        trajectory_tail = payload.get("trajectory_tail")
                        shoulder_tail = payload.get("shoulder_angle_tail")
                        release_tail = payload.get("release_velocity_tail")
                        tooltip_sections = []
                        if trajectory_tail is not None and not trajectory_tail.empty:
                            tooltip_sections.append(
                                "Trajectory (latest):\n" + trajectory_tail.to_string(index=False)
                            )
                        if shoulder_tail is not None and not shoulder_tail.empty:
                            tooltip_sections.append(
                                "Shoulder angles (latest):\n"
                                + shoulder_tail.to_string(index=False)
                            )
                        if release_tail is not None and not release_tail.empty:
                            tooltip_sections.append(
                                "Release velocity:\n" + release_tail.to_string(index=False)
                            )
                        if tooltip_sections:
                            tooltip_text = "\n\n".join(tooltip_sections)
                            live_placeholders["tooltip"].markdown(
                                f'<div class="tooltip-row">{_tooltip_html("Ball Throw Matrices", tooltip_text)}</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        speed_tail = payload.get("speed_profile_tail")
                        split_times = payload.get("split_times")
                        touch_rate_tail = payload.get("touch_rate_tail")
                        touch_log_tail = payload.get("touch_log_tail")
                        if speed_tail is not None:
                            speed_tail_display = speed_tail.copy()
                            if not use_metric_display:
                                if "avg_speed_mps" in speed_tail_display:
                                    speed_tail_display["avg_speed_kmh"] = (
                                        speed_tail_display.pop("avg_speed_mps") * 3.6
                                    )
                                if "max_speed_mps" in speed_tail_display:
                                    speed_tail_display["max_speed_kmh"] = (
                                        speed_tail_display.pop("max_speed_mps") * 3.6
                                    )
                                if "distance_m" in speed_tail_display:
                                    speed_tail_display["distance_km"] = (
                                        speed_tail_display.pop("distance_m") / 1000.0
                                    )

                            tooltip_sections = [
                                "Speed profile (latest rows):\n"
                                + speed_tail_display.to_string(index=False)
                            ]
                            if split_times is not None:
                                split_times_display = split_times.copy()
                                if not use_metric_display and "distance_m" in split_times_display:
                                    split_times_display["distance_km"] = (
                                        split_times_display.pop("distance_m") / 1000.0
                                    )
                                tooltip_sections.append(
                                    "Split times:\n" + split_times_display.to_string(index=False)
                                )
                            if touch_rate_tail is not None:
                                tooltip_sections.append(
                                    "Touch rate (latest rows):\n"
                                    + touch_rate_tail.to_string(index=False)
                                )
                            if touch_log_tail is not None:
                                tooltip_sections.append(
                                    "Touch log (latest):\n"
                                    + touch_log_tail.to_string(index=False)
                                )

                            label = (
                                "Dribbling Matrices"
                                if st.session_state.selected_test == "Dribbling"
                                else "Agility Matrices"
                            )
                            tooltip_text = "\n\n".join(tooltip_sections)
                            live_placeholders["tooltip"].markdown(
                                f'<div class="tooltip-row">{_tooltip_html(label, tooltip_text)}</div>',
                                unsafe_allow_html=True,
                            )

                    progress_value = payload.get("progress")
                    if progress_value is not None:
                        live_placeholders["progress"].progress(min(1.0, progress_value))
                    live_placeholders["status"].caption(
                        f"Frame {payload.get('frame_idx')} • Live analysis running"
                    )

                analysis_settings = {
                    "detector_weights": st.session_state.selected_model,
                    "pose_weights": st.session_state.selected_pose_model,
                    "calibration_path": st.session_state.get("calibration_path"),
                    "use_homography": st.session_state.get("agility_use_homography"),
                    "display_stride": st.session_state.get("agility_display_stride"),
                    "max_frames": st.session_state.get("agility_max_frames"),
                    "runtime_store": runtime_store,
                    "live_callback": live_callback
                    if st.session_state.selected_test
                    in (
                        "Agility",
                        "Dribbling",
                        "Juggling",
                        "Ball Throw",
                        "Sprint",
                        "Counter Movement Jump (CMJ)",
                        "Drop Jump",
                        "Endurance",
                    )
                    else None,
                    "live_stride": 3,
                    "live_tail_rows": 8,
                    "total_frames": total_frames,
                    "ball_conf": st.session_state.get("ball_detection_conf", cfg.DET_CONF),
                    "person_conf": cfg.DET_CONF,
                    "det_imgsz": st.session_state.get("ball_detection_imgsz", 640),
                    "ball_hold_frames": st.session_state.get("ball_hold_frames", cfg.BALL_HOLD_FRAMES),
                    "ball_smoothing": st.session_state.get("ball_smoothing", cfg.BALL_SMOOTHING),
                    "juggling_gap_threshold_s": st.session_state.get(
                        "juggling_gap_threshold_s", 1.0
                    ),
                    "juggling_missing_ball_frames": st.session_state.get(
                        "juggling_missing_ball_frames", 10
                    ),
                    "juggling_ground_ratio": st.session_state.get("juggling_ground_ratio", 0.92),
                    "juggling_ground_hold_frames": st.session_state.get(
                        "juggling_ground_hold_frames", 3
                    ),
                    "juggling_use_player_height": st.session_state.get(
                        "juggling_use_player_height", True
                    ),
                    "juggling_min_height_ratio": st.session_state.get(
                        "juggling_min_height_ratio", 0.1
                    ),
                    "juggling_stability_window": st.session_state.get(
                        "juggling_stability_window", 10
                    ),
                    "juggling_touch_window_seconds": st.session_state.get(
                        "juggling_touch_window_seconds", 2.0
                    ),
                    "ball_throw_release_speed_mps": st.session_state.get(
                        "ball_throw_release_speed_mps", 3.0
                    ),
                    "ball_throw_release_speed_px_s": st.session_state.get(
                        "ball_throw_release_speed_px_s", 120.0
                    ),
                    "ball_throw_release_window_frames": st.session_state.get(
                        "ball_throw_release_window_frames", 8
                    ),
                    "ball_throw_missing_ball_frames": st.session_state.get(
                        "ball_throw_missing_ball_frames", 8
                    ),
                    "ball_throw_use_player_height": st.session_state.get(
                        "ball_throw_use_player_height", True
                    ),
                    "ball_throw_min_height_ratio": st.session_state.get(
                        "ball_throw_min_height_ratio", 0.1
                    ),
                    "ball_throw_ground_ratio": st.session_state.get(
                        "ball_throw_ground_ratio", 0.92
                    ),
                    "ball_throw_ground_hold_frames": st.session_state.get(
                        "ball_throw_ground_hold_frames", 3
                    ),
                    "sprint_contact_ratio": st.session_state.get("sprint_contact_ratio", 0.9),
                    "sprint_contact_cooldown_frames": st.session_state.get(
                        "sprint_contact_cooldown_frames", 6
                    ),
                    "sprint_stride_window_s": st.session_state.get(
                        "sprint_stride_window_s", 1.0
                    ),
                    "sprint_split_distances": [
                        float(val.strip())
                        for val in str(
                            st.session_state.get("sprint_split_distances", "5,10,20,30")
                        ).split(",")
                        if val.strip()
                    ],
                }
                st.session_state.analysis_result = registry.run_analysis(
                    st.session_state.selected_test,
                    video_path,
                    analysis_settings,
                )
                st.session_state.analysis_error = None
            except Exception as exc:  # pragma: no cover - UI path
                st.session_state.analysis_result = None
                st.session_state.analysis_error = str(exc)

    if st.session_state.analysis_error:
        st.error(st.session_state.analysis_error)

    result = st.session_state.analysis_result
    if result:
        runtime_store = st.session_state.get("analysis_runtime")
        if not isinstance(runtime_store, dict):
            runtime_store = {}
        report_payload = build_report_payload(result, runtime_store, definition.description)
        if definition.name == "Agility":
            graph_label = "Agility Graph"
            speed_label = "Agility Speed"
        elif definition.name == "Dribbling":
            graph_label = "Dribbling Graph"
            speed_label = "Touch Rate"
        elif definition.name == "Ball Throw":
            graph_label = "Trajectory"
            speed_label = "Shoulder Angle"
        elif definition.name == "Sprint":
            graph_label = "Sprint Graph"
            speed_label = "Stride Rate"
        elif definition.name == "Endurance":
            graph_label = "Fatigue & Turns"
            speed_label = "Pace & HR"
        elif definition.name == "Juggling":
            graph_label = "Juggling Graph"
            speed_label = "Ball Height"
        elif definition.name == "Counter Movement Jump (CMJ)":
            graph_label = "Force-Time"
            speed_label = "Jump Height"
        elif definition.name == "Drop Jump":
            graph_label = "Ground Contact"
            speed_label = "Reactive Strength"
        else:
            graph_label = "Performance Graph"
            speed_label = "Speed Chart"

        tabs = st.tabs(
            [
                "Summary",
                "Player",
                graph_label,
                speed_label,
                "Snapshots",
                "Export",
                "Coach Chat",
            ]
        )

        speed_profile = result.matrices.get("speed_profile")
        split_times = result.matrices.get("split_times")

        with tabs[0]:
            st.markdown(f"**Status:** {result.status}")
            use_metric_display = st.session_state.get("use_metric_display", True)
            if definition.name == "Agility":
                metrics_data = result.metrics or {}
                row_one = st.columns(3)
                row_one[0].metric(
                    "Total Time", format_time(metrics_data.get("total_time_s"))
                )
                row_one[1].metric(
                    "Total Distance",
                    format_distance_m(metrics_data.get("total_distance_m"), use_metric_display),
                )
                row_one[2].metric(
                    "Average Speed",
                    format_speed_mps(metrics_data.get("avg_speed_mps"), use_metric_display),
                )
                row_two = st.columns(3)
                row_two[0].metric(
                    "Max Speed",
                    format_speed_mps(metrics_data.get("max_speed_mps"), use_metric_display),
                )
                row_two[1].metric(
                    "Peak Accel", format_accel_mps2(metrics_data.get("peak_accel_mps2"))
                )
                row_two[2].metric(
                    "Peak Decel", format_accel_mps2(metrics_data.get("peak_decel_mps2"))
                )
            elif definition.name == "Dribbling":
                metrics_data = result.metrics or {}
                touches = metrics_data.get("touch_count")
                touches_per_min = metrics_data.get("touches_per_min")
                touches_per_meter = metrics_data.get("touches_per_meter")
                left_touches = metrics_data.get("left_touches")
                right_touches = metrics_data.get("right_touches")
                left_share = None
                if touches:
                    left_share = (left_touches or 0) / touches * 100.0
                if not use_metric_display and metrics_data.get("total_distance_m"):
                    distance_km = metrics_data.get("total_distance_m") / 1000.0
                    touches_per_meter = (
                        touches / distance_km if distance_km > 0 else None
                    )

                row_one = st.columns(3)
                row_one[0].metric(
                    "Total Time", format_time(metrics_data.get("total_time_s"))
                )
                row_one[1].metric(
                    "Total Distance",
                    format_distance_m(metrics_data.get("total_distance_m"), use_metric_display),
                )
                row_one[2].metric(
                    "Touches", "--" if touches is None else f"{touches:d}"
                )
                row_two = st.columns(3)
                row_two[0].metric(
                    "Touches / Min",
                    "--" if touches_per_min is None else f"{touches_per_min:.1f}",
                )
                row_two[1].metric(
                    "Touches / km" if not use_metric_display else "Touches / m",
                    "--" if touches_per_meter is None else f"{touches_per_meter:.2f}",
                )
                row_two[2].metric(
                    "Max Speed",
                    format_speed_mps(metrics_data.get("max_speed_mps"), use_metric_display),
                )

                row_three = st.columns(3)
                row_three[0].metric(
                    "Left Touches",
                    "--" if left_touches is None else f"{left_touches:d}",
                )
                row_three[1].metric(
                    "Right Touches",
                    "--" if right_touches is None else f"{right_touches:d}",
                )
                row_three[2].metric(
                    "Left %",
                    "--" if left_share is None else f"{left_share:.0f}%",
                )
            elif definition.name == "Juggling":
                metrics_data = result.metrics or {}
                touches = metrics_data.get("touch_count")
                max_streak = metrics_data.get("max_consecutive_touches")
                left_touches = metrics_data.get("left_touches")
                right_touches = metrics_data.get("right_touches")
                touches_per_min = metrics_data.get("touches_per_min")
                avg_interval = metrics_data.get("avg_touch_interval_s")
                stability_score = metrics_data.get("stability_score")
                avg_height = metrics_data.get("avg_ball_height")
                max_height = metrics_data.get("max_ball_height")
                drop_count = metrics_data.get("drop_count")

                row_one = st.columns(3)
                row_one[0].metric(
                    "Total Time", format_time(metrics_data.get("total_time_s"))
                )
                row_one[1].metric(
                    "Touches", "--" if touches is None else f"{touches:d}"
                )
                row_one[2].metric(
                    "Max Streak", "--" if max_streak is None else f"{max_streak:d}"
                )

                row_two = st.columns(3)
                row_two[0].metric(
                    "Left Touches",
                    "--" if left_touches is None else f"{left_touches:d}",
                )
                row_two[1].metric(
                    "Right Touches",
                    "--" if right_touches is None else f"{right_touches:d}",
                )
                row_two[2].metric(
                    "Touches / Min",
                    "--" if touches_per_min is None else f"{touches_per_min:.1f}",
                )

                row_three = st.columns(3)
                row_three[0].metric(
                    "Avg Interval (s)",
                    "--" if avg_interval is None else f"{avg_interval:.2f}",
                )
                row_three[1].metric(
                    "Stability",
                    "--" if stability_score is None else f"{stability_score:.2f}",
                )
                row_three[2].metric(
                    "Drops",
                    "--" if drop_count is None else f"{drop_count:d}",
                )

                height_parts = []
                if avg_height is not None:
                    height_parts.append(f"Avg height: {avg_height:.2f}")
                if max_height is not None:
                    height_parts.append(f"Max height: {max_height:.2f}")
                if height_parts:
                    st.caption(" · ".join(height_parts))
            elif definition.name == "Counter Movement Jump (CMJ)":
                metrics_data = result.metrics or {}
                total_jumps = metrics_data.get("total_jumps")
                highest_jump_m = metrics_data.get("highest_jump_m")
                highest_jump_px = metrics_data.get("highest_jump_px")
                avg_jump_m = metrics_data.get("avg_jump_height_m")
                avg_jump_px = metrics_data.get("avg_jump_height_px")
                avg_flight_time = metrics_data.get("avg_flight_time_s")
                avg_stability = metrics_data.get("avg_landing_stability")
                best_stability = metrics_data.get("best_landing_stability")

                row_one = st.columns(3)
                row_one[0].metric(
                    "Total Time", format_time(metrics_data.get("total_time_s"))
                )
                row_one[1].metric(
                    "Total Jumps", "--" if total_jumps is None else f"{total_jumps}"
                )
                highest_display = "--"
                if highest_jump_m is not None:
                    highest_display = f"{highest_jump_m:.2f} m"
                elif highest_jump_px is not None:
                    highest_display = f"{highest_jump_px:.0f} px"
                row_one[2].metric("Highest Jump", highest_display)

                row_two = st.columns(3)
                avg_display = "--"
                if avg_jump_m is not None:
                    avg_display = f"{avg_jump_m:.2f} m"
                elif avg_jump_px is not None:
                    avg_display = f"{avg_jump_px:.0f} px"
                row_two[0].metric("Avg Jump", avg_display)
                row_two[1].metric(
                    "Avg Flight",
                    "--" if avg_flight_time is None else f"{avg_flight_time:.2f} s",
                )
                row_two[2].metric(
                    "Best Stability",
                    "--" if best_stability is None else f"{best_stability:.2f}",
                )

                if avg_stability is not None:
                    st.caption(f"Avg landing stability: {avg_stability:.2f}")
            elif definition.name == "Drop Jump":
                metrics_data = result.metrics or {}
                total_jumps = metrics_data.get("total_jumps")
                last_contact = metrics_data.get("last_contact_time_s")
                last_rsi = metrics_data.get("last_rsi")
                best_rsi = metrics_data.get("best_rsi")
                avg_rsi = metrics_data.get("avg_rsi")
                best_contact = metrics_data.get("best_contact_time_s")
                avg_contact = metrics_data.get("avg_contact_time_s")
                best_jump_m = metrics_data.get("best_jump_height_m")
                best_jump_px = metrics_data.get("best_jump_height_px")
                avg_jump_m = metrics_data.get("avg_jump_height_m")
                avg_jump_px = metrics_data.get("avg_jump_height_px")

                row_one = st.columns(3)
                row_one[0].metric(
                    "Total Time", format_time(metrics_data.get("total_time_s"))
                )
                row_one[1].metric(
                    "Total Jumps", "--" if total_jumps is None else f"{total_jumps}"
                )
                row_one[2].metric(
                    "Last Contact",
                    "--" if last_contact is None else f"{last_contact:.2f} s",
                )

                row_two = st.columns(3)
                best_jump_display = "--"
                if best_jump_m is not None:
                    best_jump_display = f"{best_jump_m:.2f} m"
                elif best_jump_px is not None:
                    best_jump_display = f"{best_jump_px:.0f} px"
                row_two[0].metric("Best Jump", best_jump_display)
                avg_jump_display = "--"
                if avg_jump_m is not None:
                    avg_jump_display = f"{avg_jump_m:.2f} m"
                elif avg_jump_px is not None:
                    avg_jump_display = f"{avg_jump_px:.0f} px"
                row_two[1].metric("Avg Jump", avg_jump_display)
                row_two[2].metric(
                    "Best RSI",
                    "--" if best_rsi is None else f"{best_rsi:.2f}",
                )

                row_three = st.columns(3)
                row_three[0].metric(
                    "Avg RSI", "--" if avg_rsi is None else f"{avg_rsi:.2f}"
                )
                row_three[1].metric(
                    "Best Contact",
                    "--" if best_contact is None else f"{best_contact:.2f} s",
                )
                row_three[2].metric(
                    "Avg Contact",
                    "--" if avg_contact is None else f"{avg_contact:.2f} s",
                )

                if last_rsi is not None:
                    st.caption(f"Last RSI: {last_rsi:.2f}")
            elif definition.name == "Ball Throw":
                metrics_data = result.metrics or {}
                release_speed_mps = metrics_data.get("release_speed_mps")
                release_speed_px = metrics_data.get("release_speed_px_s")
                release_angle = metrics_data.get("release_angle_deg")
                max_height = metrics_data.get("max_ball_height")

                row_one = st.columns(3)
                row_one[0].metric(
                    "Total Time", format_time(metrics_data.get("total_time_s"))
                )
                release_display = (
                    format_speed_mps(release_speed_mps, use_metric_display)
                    if release_speed_mps is not None
                    else ("--" if release_speed_px is None else f"{release_speed_px:.1f} px/s")
                )
                row_one[1].metric("Release Speed", release_display)
                row_one[2].metric(
                    "Release Angle",
                    "--" if release_angle is None else f"{release_angle:.1f}°",
                )

                row_two = st.columns(3)
                throw_dist = metrics_data.get("throw_distance_m")
                throw_dist_px = metrics_data.get("throw_distance_px")
                throw_display = (
                    format_distance_m(throw_dist, use_metric_display)
                    if throw_dist is not None
                    else ("--" if throw_dist_px is None else f"{throw_dist_px:.1f} px")
                )
                row_two[0].metric("Throw Distance", throw_display)
                hand_force = metrics_data.get("hand_force_mps2")
                hand_force_px = metrics_data.get("hand_force_px_s2")
                hand_display = (
                    format_accel_mps2(hand_force)
                    if hand_force is not None
                    else ("--" if hand_force_px is None else f"{hand_force_px:.1f} px/s²")
                )
                row_two[1].metric("Hand Force", hand_display)
                row_two[2].metric(
                    "Max Height",
                    "--" if max_height is None else f"{max_height:.2f}",
                )

                if release_speed_mps is None and release_speed_px is None:
                    st.caption("Release not detected yet. Adjust the release thresholds if needed.")

                detect_rate = metrics_data.get("ball_detection_rate")
                if detect_rate is not None:
                    detected = metrics_data.get("ball_detected_frames", 0)
                    total = metrics_data.get("processed_frames", 0)
                    st.caption(f"Ball detection coverage: {detected}/{total} ({detect_rate:.0%})")
                ball_class_name = metrics_data.get("ball_class_name")
                ball_class_id = metrics_data.get("ball_class_id")
                if ball_class_name:
                    st.caption(f"Detector ball class: {ball_class_name} (id {ball_class_id})")
                else:
                    st.caption("Detector ball class: not found in model names")
            elif definition.name == "Sprint":
                metrics_data = result.metrics or {}
                row_one = st.columns(3)
                row_one[0].metric(
                    "Total Time", format_time(metrics_data.get("total_time_s"))
                )
                row_one[1].metric(
                    "Total Distance",
                    format_distance_m(metrics_data.get("total_distance_m"), use_metric_display),
                )
                row_one[2].metric(
                    "Average Speed",
                    format_speed_mps(metrics_data.get("avg_speed_mps"), use_metric_display),
                )

                row_two = st.columns(3)
                row_two[0].metric(
                    "Max Speed",
                    format_speed_mps(metrics_data.get("max_speed_mps"), use_metric_display),
                )
                row_two[1].metric(
                    "Peak Accel", format_accel_mps2(metrics_data.get("peak_accel_mps2"))
                )
                row_two[2].metric(
                    "Peak Decel", format_accel_mps2(metrics_data.get("peak_decel_mps2"))
                )

                row_three = st.columns(3)
                row_three[0].metric(
                    "Time to 90%",
                    format_time(metrics_data.get("time_to_90_pct_s")),
                )
                row_three[1].metric(
                    "Time to 95%",
                    format_time(metrics_data.get("time_to_95_pct_s")),
                )
                row_three[2].metric(
                    "Avg Stride Rate",
                    "--"
                    if metrics_data.get("avg_stride_rate_hz") is None
                    else f"{metrics_data.get('avg_stride_rate_hz'):.2f} Hz",
                )

                split_times = result.matrices.get("split_times")
                if split_times is not None and not split_times.empty:
                    split_display = split_times.copy()
                    if not use_metric_display and "distance_m" in split_display:
                        split_display["distance_km"] = split_display.pop("distance_m") / 1000.0
                    st.markdown("#### Split Times")
                    st.dataframe(split_display, use_container_width=True, hide_index=True)
            elif definition.name == "Endurance":
                metrics_data = result.metrics or {}
                row_one = st.columns(3)
                row_one[0].metric(
                    "Total Time", format_time(metrics_data.get("total_time_s"))
                )
                row_one[1].metric(
                    "Total Distance",
                    format_distance_m(metrics_data.get("total_distance_m"), use_metric_display),
                )
                row_one[2].metric(
                    "Average Speed",
                    format_speed_mps(metrics_data.get("avg_speed_mps"), use_metric_display),
                )

                row_two = st.columns(3)
                row_two[0].metric(
                    "Max Speed",
                    format_speed_mps(metrics_data.get("max_speed_mps"), use_metric_display),
                )
                total_turns = metrics_data.get("total_turns")
                row_two[1].metric(
                    "Total Turns", "--" if total_turns is None else f"{total_turns}"
                )
                turns_per_min = metrics_data.get("turns_per_min")
                row_two[2].metric(
                    "Turns / Min",
                    "--" if turns_per_min is None else f"{turns_per_min:.1f}",
                )

                row_three = st.columns(3)
                avg_hr = metrics_data.get("avg_heart_rate_bpm")
                max_hr = metrics_data.get("max_heart_rate_bpm")
                fatigue_index = metrics_data.get("fatigue_index")
                row_three[0].metric(
                    "Avg HR", "--" if avg_hr is None else f"{avg_hr:.0f} bpm"
                )
                row_three[1].metric(
                    "Max HR", "--" if max_hr is None else f"{max_hr:.0f} bpm"
                )
                row_three[2].metric(
                    "Fatigue",
                    "--"
                    if fatigue_index is None
                    else f"{fatigue_index * 100:.0f}%",
                )

                caption_parts = []
                avg_pace = metrics_data.get("avg_pace_min_per_km")
                if avg_pace is not None:
                    caption_parts.append(f"Avg pace: {avg_pace:.2f} min/km")
                avg_turn_angle = metrics_data.get("avg_turn_angle_deg")
                if avg_turn_angle is not None:
                    caption_parts.append(f"Avg turn angle: {avg_turn_angle:.1f} deg")
                max_turn_angle = metrics_data.get("max_turn_angle_deg")
                if max_turn_angle is not None:
                    caption_parts.append(f"Max turn angle: {max_turn_angle:.1f} deg")
                if caption_parts:
                    st.caption(" · ".join(caption_parts))
            else:
                metrics_items = list(result.metrics.items())
                for chunk in _chunk(metrics_items, 4):
                    columns = st.columns(len(chunk))
                    for column, (label, value) in zip(columns, chunk):
                        display_label = label.replace("_", " ").title()
                        column.metric(display_label, value)

            if definition.name == "Agility":
                if split_times is not None and not split_times.empty:
                    split_display = split_times.copy()
                    if not use_metric_display and "distance_m" in split_display:
                        split_display["distance_km"] = split_display.pop("distance_m") / 1000.0
                    st.markdown("#### Split Times")
                    st.dataframe(split_display, use_container_width=True, hide_index=True)
                else:
                    st.info("Split times will appear after analysis completes.")

            if definition.name == "Dribbling":
                touch_log = result.matrices.get("touch_log")
                if touch_log is not None and not touch_log.empty:
                    with st.expander("Touch Log", expanded=False):
                        st.dataframe(touch_log, use_container_width=True, hide_index=True)

        with tabs[1]:
            if not has_video:
                st.info("Upload a video to enable the player.")
            else:
                frame_records = runtime_store.get("frame_records", [])
                if not frame_records:
                    st.info("Run analysis to generate player overlays.")
                else:
                    frame_lookup = {
                        rec.get("frame_idx"): rec
                        for rec in frame_records
                        if rec.get("frame_idx") is not None
                    }
                    available_frames = sorted(frame_lookup.keys())
                    if not available_frames:
                        st.info("Overlay data is empty. Re-run analysis.")
                    else:
                        display_stride_used = runtime_store.get("display_stride", 1)
                        if display_stride_used and display_stride_used > 1:
                            st.caption(
                                "Note: overlay data is recorded every "
                                f"{display_stride_used} frames. For full coverage, set Display stride to 1."
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
                                    "Ball vector", value=True, key="player_show_ball_vector"
                                )
                                show_ball_speed = st.checkbox(
                                    "Ball speed", value=True, key="player_show_ball_speed"
                                )
                            with col_c:
                                show_ball_trail = st.checkbox(
                                    "Ball trail", value=True, key="player_show_ball_trail"
                                )
                                show_player_speed = st.checkbox(
                                    "Player speed", value=True, key="player_show_player_speed"
                                )
                                show_annotations = st.checkbox(
                                    "Annotations", value=True, key="player_show_annotations"
                                )
                            if show_ball_trail:
                                trail_len = st.slider(
                                    "Trail length (frames)",
                                    min_value=2,
                                    max_value=60,
                                    value=12,
                                    step=1,
                                    key="player_trail_len",
                                )
                            else:
                                trail_len = 0
                            vector_scale = st.slider(
                                "Vector scale",
                                min_value=4.0,
                                max_value=24.0,
                                value=10.0,
                                step=1.0,
                                key="player_vector_scale",
                            )

                        max_frame = available_frames[-1]
                        display_stride_used = max(1, int(display_stride_used))
                        if len(available_frames) <= 5000:
                            frame_idx = st.select_slider(
                                "Frame",
                                options=available_frames,
                                value=max_frame,
                                key="player_frame_idx",
                            )
                        else:
                            min_frame = available_frames[0]
                            frame_idx = st.slider(
                                "Frame",
                                min_value=min_frame,
                                max_value=max_frame,
                                value=max_frame,
                                step=display_stride_used,
                                key="player_frame_idx",
                            )

                        frame_bgr = read_video_frame(video_path, frame_idx)
                        if frame_bgr is None:
                            st.error("Unable to read the selected frame.")
                        else:
                            overlay_rec = frame_lookup.get(frame_idx)
                            frame_render = frame_bgr.copy()
                            if overlay_rec is None:
                                st.info(
                                    "Overlay data not available for this frame. Re-run with Display stride = 1."
                                )
                            else:
                                meta = overlay_rec.get("meta", {}) or {}
                                stats = overlay_rec.get("stats", {}) or {}
                                use_homography = bool(
                                    meta.get("use_homography", st.session_state.agility_use_homography)
                                )
                                if show_ball and show_ball_trail:
                                    trail_points = collect_ball_trail(
                                        frame_lookup,
                                        frame_idx,
                                        max_len=max(2, trail_len),
                                        max_gap_frames=int(cfg.BALL_TRAIL_MAX_GAP_FRAMES),
                                    )
                                    draw_ball_trail_overlay(frame_render, trail_points)
                                if show_players:
                                    draw_player_overlays(
                                        frame_render,
                                        meta.get("players", []),
                                        show_ids=show_ids,
                                        show_feet=show_feet,
                                        show_speed=show_player_speed,
                                        use_metric_display=st.session_state.get("use_metric_display", True),
                                    )
                                if show_ball:
                                    draw_ball_overlay(
                                        frame_render,
                                        meta.get("ball"),
                                        show_vector=show_ball_vector,
                                        show_speed=show_ball_speed,
                                        use_homography=use_homography,
                                        vector_scale=vector_scale,
                                    )
                                if show_annotations:
                                    draw_event_overlay(frame_render, meta.get("event_overlay"))

                                overlay_lines = []
                                use_metric_display = st.session_state.get("use_metric_display", True)
                                avg_speed_text = format_speed(
                                    stats.get("avg_speed_kmh"), use_metric_display
                                )
                                max_speed_text = format_speed(
                                    stats.get("max_speed_kmh"), use_metric_display
                                )
                                overlay_lines.append(
                                    f"Speed (avg / max): {avg_speed_text} / {max_speed_text}"
                                )
                                overlay_lines.append(
                                    f"Time / Distance: {format_time(stats.get('total_time_sec'))} / "
                                    f"{format_distance_m(stats.get('total_distance_m'), use_metric_display)}"
                                )
                                overlay_lines.append(
                                    f"Accel / Decel (peak): "
                                    f"{format_accel(stats.get('peak_accel_mps2'))} / "
                                    f"{format_accel(stats.get('peak_decel_mps2'))}"
                                )
                                if definition.name in ("Dribbling", "Juggling"):
                                    touches_val = stats.get("total_touches")
                                    if touches_val is not None:
                                        overlay_lines.append(f"Touches: {touches_val}")
                                    left_val = stats.get("left_touches")
                                    right_val = stats.get("right_touches")
                                    if left_val is not None or right_val is not None:
                                        left_disp = "--" if left_val is None else f"{left_val}"
                                        right_disp = "--" if right_val is None else f"{right_val}"
                                        overlay_lines.append(f"Left / Right: {left_disp} / {right_disp}")
                                    if definition.name == "Juggling":
                                        max_streak_val = stats.get("max_consecutive_touches")
                                        if max_streak_val is not None:
                                            overlay_lines.append(f"Max Streak: {max_streak_val}")
                                if definition.name == "Sprint":
                                    stride_rate = stats.get("stride_rate_hz")
                                    if stride_rate is not None:
                                        overlay_lines.append(f"Stride Rate: {stride_rate:.2f} Hz")
                                if definition.name == "Counter Movement Jump (CMJ)":
                                    total_jumps = stats.get("total_jumps", 0)
                                    jump_height_text = "--"
                                    if stats.get("highest_jump_m") is not None:
                                        jump_height_text = f"{stats.get('highest_jump_m'):.2f} m"
                                    elif stats.get("highest_jump_px") is not None:
                                        jump_height_text = f"{stats.get('highest_jump_px'):.0f} px"
                                    overlay_lines.append(
                                        f"Jumps / Highest: {total_jumps} / {jump_height_text}"
                                    )
                                if definition.name == "Drop Jump":
                                    total_jumps = stats.get("total_jumps", 0)
                                    last_contact = stats.get("last_contact_time_s")
                                    last_rsi = stats.get("last_rsi")
                                    last_height_text = "--"
                                    if stats.get("last_jump_height_m") is not None:
                                        last_height_text = f"{stats.get('last_jump_height_m'):.2f} m"
                                    elif stats.get("last_jump_height_px") is not None:
                                        last_height_text = f"{stats.get('last_jump_height_px'):.0f} px"
                                    contact_text = "--" if last_contact is None else f"{last_contact:.2f}s"
                                    rsi_text = "--" if last_rsi is None else f"{last_rsi:.2f}"
                                    overlay_lines.append(
                                        f"Drop Jumps: {total_jumps}  Contact: {contact_text}  RSI: {rsi_text}"
                                    )
                                    overlay_lines.append(f"Last Jump: {last_height_text}")
                                draw_stats_overlay(frame_render, overlay_lines)

                            try:
                                import cv2  # type: ignore

                                frame_rgb = cv2.cvtColor(frame_render, cv2.COLOR_BGR2RGB)
                                st.image(frame_rgb, caption=f"Frame {frame_idx}", use_container_width=True)
                            except Exception:
                                st.image(frame_render, caption=f"Frame {frame_idx}", use_container_width=True)

        with tabs[2]:
            if definition.name == "Ball Throw":
                trajectory = result.matrices.get("trajectory")
                if trajectory is not None and not trajectory.empty:
                    chart_cols = [col for col in ["x_norm", "height_proxy"] if col in trajectory.columns]
                    if not chart_cols and "y_norm" in trajectory.columns:
                        chart_cols = ["y_norm"]
                    if chart_cols:
                        chart_df = trajectory.set_index("time_s")[chart_cols].copy()
                        st.line_chart(chart_df)
                        st.caption("Normalized trajectory (x/height) over time.")
                    else:
                        st.info("Trajectory data will appear after analysis.")
                else:
                    st.info("Trajectory data will appear after analysis.")
            elif definition.name == "Sprint":
                acceleration_phase = result.matrices.get("acceleration_phase")
                if acceleration_phase is not None and not acceleration_phase.empty:
                    chart_cols = [col for col in ["speed_mps", "accel_mps2"] if col in acceleration_phase.columns]
                    chart_df = acceleration_phase.set_index("time_s")[chart_cols].copy()
                    if not st.session_state.get("use_metric_display", True) and "speed_mps" in chart_df:
                        chart_df["speed_kmh"] = chart_df.pop("speed_mps") * 3.6
                    st.line_chart(chart_df)
                    st.caption("Acceleration phase: speed and acceleration over time.")
                else:
                    st.info("Acceleration phase data will appear after analysis.")
            elif definition.name == "Juggling":
                control_stability = result.matrices.get("control_stability")
                if control_stability is not None and not control_stability.empty:
                    chart_cols = [
                        col
                        for col in ["stability_score", "touch_interval_s", "interval_std"]
                        if col in control_stability.columns
                    ]
                    if chart_cols:
                        chart_df = control_stability.set_index("time_s")[chart_cols].copy()
                        st.line_chart(chart_df)
                        st.caption("Touch interval and stability over time.")
                    else:
                        st.info("Control stability data will appear after analysis.")
                else:
                    st.info("Control stability data will appear after analysis.")
            elif definition.name == "Counter Movement Jump (CMJ)":
                force_time = result.matrices.get("force_time")
                if force_time is not None and not force_time.empty:
                    use_metric_display = st.session_state.get("use_metric_display", True)
                    chart_cols = []
                    if use_metric_display and "accel_mps2" in force_time.columns:
                        chart_cols.append("accel_mps2")
                    if not use_metric_display and "accel_px_s2" in force_time.columns:
                        chart_cols.append("accel_px_s2")
                    if "force_proxy" in force_time.columns:
                        chart_cols.append("force_proxy")
                    if chart_cols:
                        chart_df = force_time.set_index("time_s")[chart_cols].copy()
                        st.line_chart(chart_df)
                        st.caption("Force-time proxy from vertical acceleration.")
                    else:
                        st.info("Force-time data will appear after analysis.")
                else:
                    st.info("Force-time data will appear after analysis.")
            elif definition.name == "Drop Jump":
                ground_contact = result.matrices.get("ground_contact")
                if ground_contact is not None and not ground_contact.empty:
                    use_metric_display = st.session_state.get("use_metric_display", True)
                    chart_cols = []
                    if "contact_time_s" in ground_contact.columns:
                        chart_cols.append("contact_time_s")
                    if use_metric_display and "impact_velocity_mps" in ground_contact.columns:
                        chart_cols.append("impact_velocity_mps")
                    if not use_metric_display and "impact_velocity_px_s" in ground_contact.columns:
                        chart_cols.append("impact_velocity_px_s")
                    if use_metric_display and "peak_accel_mps2" in ground_contact.columns:
                        chart_cols.append("peak_accel_mps2")
                    if not use_metric_display and "peak_accel_px_s2" in ground_contact.columns:
                        chart_cols.append("peak_accel_px_s2")
                    if chart_cols:
                        chart_df = ground_contact.set_index("jump_id")[chart_cols].copy()
                        st.line_chart(chart_df)
                        st.caption("Ground contact time and landing intensity by jump.")
                    else:
                        st.info("Ground contact data will appear after analysis.")
                else:
                    st.info("Ground contact data will appear after analysis.")
            elif definition.name == "Endurance":
                fatigue_index = result.matrices.get("fatigue_index")
                if fatigue_index is not None and not fatigue_index.empty:
                    chart_cols = [
                        col
                        for col in ["fatigue_score", "rolling_speed_mps"]
                        if col in fatigue_index.columns
                    ]
                    if chart_cols:
                        chart_df = fatigue_index.set_index("time_s")[chart_cols].copy()
                        if (
                            not st.session_state.get("use_metric_display", True)
                            and "rolling_speed_mps" in chart_df.columns
                        ):
                            chart_df["rolling_speed_kmh"] = chart_df.pop("rolling_speed_mps") * 3.6
                        st.line_chart(chart_df)
                        st.caption("Fatigue and rolling speed over time.")
                    else:
                        st.info("Fatigue data will appear after analysis.")
                else:
                    st.info("Fatigue data will appear after analysis.")

                turn_profile = result.matrices.get("turn_profile")
                if turn_profile is not None and not turn_profile.empty:
                    chart_cols = [
                        col for col in ["turn_rate_deg_s", "turns"] if col in turn_profile.columns
                    ]
                    if chart_cols:
                        chart_df = turn_profile.set_index("time_s")[chart_cols].copy()
                        st.line_chart(chart_df)
                        st.caption("Turn rate and cumulative turns over time.")
            elif speed_profile is not None and not speed_profile.empty:
                if definition.name == "Dribbling":
                    chart_df = speed_profile.set_index("time_s")[
                        ["touch_rate", "distance_m"]
                    ].copy()
                    if not st.session_state.get("use_metric_display", True):
                        chart_df["distance_km"] = chart_df.pop("distance_m") / 1000.0
                    st.line_chart(chart_df)
                    st.caption("Touch rate and distance over time.")
                else:
                    chart_df = speed_profile.set_index("time_s")[
                        ["accel_mps2", "distance_m"]
                    ].copy()
                    if not st.session_state.get("use_metric_display", True):
                        chart_df["distance_km"] = chart_df.pop("distance_m") / 1000.0
                    st.line_chart(chart_df)
                    st.caption("Acceleration and distance over time.")
            else:
                st.info("Graph will appear after analysis.")

        with tabs[3]:
            if definition.name == "Ball Throw":
                shoulder_angle = result.matrices.get("shoulder_angle")
                if shoulder_angle is not None and not shoulder_angle.empty:
                    chart_cols = [
                        col
                        for col in [
                            "dominant_arm_angle_deg",
                            "left_arm_angle_deg",
                            "right_arm_angle_deg",
                            "shoulder_line_angle_deg",
                        ]
                        if col in shoulder_angle.columns
                    ]
                    if chart_cols:
                        chart_df = shoulder_angle.set_index("time_s")[chart_cols].copy()
                        st.line_chart(chart_df)
                        st.caption("Shoulder/arm angles over time.")
                    else:
                        st.info("Shoulder angles will appear after analysis.")
                else:
                    st.info("Shoulder angles will appear after analysis.")
            elif definition.name == "Sprint":
                stride_freq = result.matrices.get("stride_frequency")
                if stride_freq is not None and not stride_freq.empty:
                    chart_cols = [col for col in ["stride_rate_hz", "step_rate_hz"] if col in stride_freq.columns]
                    if chart_cols:
                        chart_df = stride_freq.set_index("time_s")[chart_cols].copy()
                        st.line_chart(chart_df)
                        st.caption("Stride and step rate over time.")
                    else:
                        st.info("Stride rate data will appear after analysis.")
                else:
                    st.info("Stride rate data will appear after analysis.")
            elif definition.name == "Juggling":
                ball_height = result.matrices.get("ball_height")
                if ball_height is not None and not ball_height.empty:
                    height_col = None
                    if "height_ratio" in ball_height.columns and ball_height["height_ratio"].notna().any():
                        height_col = "height_ratio"
                    elif "height_norm" in ball_height.columns and ball_height["height_norm"].notna().any():
                        height_col = "height_norm"
                    if height_col:
                        chart_df = ball_height.set_index("time_s")[[height_col]].copy()
                        chart_df = chart_df.rename(columns={height_col: "ball_height"})
                        st.line_chart(chart_df)
                        st.caption("Ball height proxy over time.")
                    else:
                        st.info("Ball height data will appear after analysis.")
                else:
                    st.info("Ball height data will appear after analysis.")
            elif definition.name == "Counter Movement Jump (CMJ)":
                jump_height = result.matrices.get("jump_height")
                use_metric_display = st.session_state.get("use_metric_display", True)
                if jump_height is not None and not jump_height.empty:
                    height_col = None
                    if use_metric_display and "height_m" in jump_height.columns:
                        height_col = "height_m"
                    elif "height_px" in jump_height.columns:
                        height_col = "height_px"
                    if height_col:
                        chart_df = jump_height.set_index("time_s")[[height_col]].copy()
                        chart_df = chart_df.rename(columns={height_col: "jump_height"})
                        st.line_chart(chart_df)
                        st.caption("Estimated jump height over time.")
                    else:
                        st.info("Jump height data will appear after analysis.")
                else:
                    st.info("Jump height data will appear after analysis.")

                landing_stability = result.matrices.get("landing_stability")
                if landing_stability is not None and not landing_stability.empty:
                    st.markdown("#### Landing Stability")
                    if "stability_score" in landing_stability.columns:
                        chart_df = landing_stability.set_index("time_s")[["stability_score"]].copy()
                        st.line_chart(chart_df)
                    else:
                        st.dataframe(landing_stability, use_container_width=True, hide_index=True)
            elif definition.name == "Drop Jump":
                reactive_strength = result.matrices.get("reactive_strength")
                use_metric_display = st.session_state.get("use_metric_display", True)
                if reactive_strength is not None and not reactive_strength.empty:
                    chart_cols = []
                    if "rsi" in reactive_strength.columns:
                        chart_cols.append("rsi")
                    if use_metric_display and "jump_height_m" in reactive_strength.columns:
                        chart_cols.append("jump_height_m")
                    if not use_metric_display and "jump_height_px" in reactive_strength.columns:
                        chart_cols.append("jump_height_px")
                    if "contact_time_s" in reactive_strength.columns:
                        chart_cols.append("contact_time_s")
                    if chart_cols:
                        chart_df = reactive_strength.set_index("jump_id")[chart_cols].copy()
                        st.line_chart(chart_df)
                        st.caption("Reactive strength index and jump height by jump.")
                    else:
                        st.info("Reactive strength data will appear after analysis.")
                else:
                    st.info("Reactive strength data will appear after analysis.")

                landing_force = result.matrices.get("landing_force")
                if landing_force is not None and not landing_force.empty:
                    st.markdown("#### Landing Force")
                    st.dataframe(landing_force, use_container_width=True, hide_index=True)
            elif definition.name == "Endurance":
                heart_rate = result.matrices.get("heart_rate_estimate")
                if heart_rate is not None and not heart_rate.empty:
                    if "heart_rate_bpm" in heart_rate.columns:
                        chart_df = heart_rate.set_index("time_s")[["heart_rate_bpm"]].copy()
                        st.line_chart(chart_df)
                        st.caption("Estimated heart rate over time.")
                    else:
                        st.info("Heart rate data will appear after analysis.")
                else:
                    st.info("Heart rate data will appear after analysis.")

                pace_profile = result.matrices.get("pace_profile")
                if pace_profile is not None and not pace_profile.empty:
                    if "pace_min_per_km" in pace_profile.columns:
                        chart_df = pace_profile.set_index("time_s")[["pace_min_per_km"]].copy()
                        st.line_chart(chart_df)
                        st.caption("Pace over time (min/km).")
                    else:
                        st.info("Pace data will appear after analysis.")
            elif speed_profile is not None and not speed_profile.empty:
                if definition.name == "Dribbling":
                    chart_df = speed_profile.set_index("time_s")[
                        ["avg_speed_mps", "max_speed_mps"]
                    ].copy()
                    if not st.session_state.get("use_metric_display", True):
                        chart_df["avg_speed_kmh"] = chart_df.pop("avg_speed_mps") * 3.6
                        chart_df["max_speed_kmh"] = chart_df.pop("max_speed_mps") * 3.6
                    st.line_chart(chart_df)
                    st.caption("Dribbling speed over time.")
                else:
                    chart_df = speed_profile.set_index("time_s")[
                        ["avg_speed_mps", "max_speed_mps"]
                    ].copy()
                    if not st.session_state.get("use_metric_display", True):
                        chart_df["avg_speed_kmh"] = chart_df.pop("avg_speed_mps") * 3.6
                        chart_df["max_speed_kmh"] = chart_df.pop("max_speed_mps") * 3.6
                    st.line_chart(chart_df)
                    st.caption("Average and max speed over time.")
            else:
                st.info("Chart will appear after analysis.")

        with tabs[4]:
            snapshots = runtime_store.get("snapshots", [])
            if snapshots:
                st.markdown(f"**Captured snapshots ({len(snapshots)})**")
                cols = st.columns(3)
                for idx, snap in enumerate(snapshots):
                    caption_parts = []
                    if snap.get("type"):
                        caption_parts.append(str(snap.get("type")).capitalize())
                    if snap.get("time_sec") is not None:
                        caption_parts.append(f"{snap.get('time_sec'):.2f}s")
                    if snap.get("frame_idx") is not None:
                        caption_parts.append(f"Frame {snap.get('frame_idx')}")
                    caption = " · ".join(caption_parts) if caption_parts else "Snapshot"
                    with cols[idx % 3]:
                        st.image(
                            snap.get("image_path", ""),
                            caption=caption,
                            use_container_width=True,
                        )
            else:
                st.info("No snapshots captured yet.")

        with tabs[5]:
            st.download_button(
                "Download report JSON",
                data=json.dumps(report_payload, indent=2),
                file_name="agility_report.json",
                mime="application/json",
            )
            frame_records = runtime_store.get("frame_records", [])
            if frame_records:
                st.markdown("#### Annotated Video")
                progress_export = st.progress(0)
                export_col_left, export_col_right = st.columns(2)
                with export_col_left:
                    if st.button("Build annotated video", key="export_annotated_video"):
                        try:
                            overlay_settings = {
                                "show_players": st.session_state.get("player_show_players", True),
                                "show_ids": st.session_state.get("player_show_ids", True),
                                "show_feet": st.session_state.get("player_show_feet", True),
                                "show_ball": st.session_state.get("player_show_ball", True),
                                "show_ball_vector": st.session_state.get("player_show_ball_vector", True),
                                "show_ball_speed": st.session_state.get("player_show_ball_speed", True),
                                "show_ball_trail": st.session_state.get("player_show_ball_trail", True),
                                "show_player_speed": st.session_state.get("player_show_player_speed", True),
                                "show_annotations": st.session_state.get("player_show_annotations", True),
                                "trail_len": st.session_state.get("player_trail_len", 12),
                                "trail_max_gap": cfg.BALL_TRAIL_MAX_GAP_FRAMES,
                                "vector_scale": st.session_state.get("player_vector_scale", 10.0),
                                "use_metric_display": st.session_state.get("use_metric_display", True),
                                "use_homography": st.session_state.get("agility_use_homography"),
                                "display_stride": runtime_store.get("display_stride", 1),
                            }
                            def _progress_cb(value: float) -> None:
                                progress_export.progress(value)

                            export_path = export_annotated_video(
                                video_path=video_path,
                                frame_records=frame_records,
                                settings=overlay_settings,
                                output_path=None,
                                progress_cb=_progress_cb,
                            )
                            st.session_state["export_annotated_path"] = export_path
                            st.success("Annotated video ready.")
                        except Exception as exc:
                            st.error(str(exc))
                with export_col_right:
                    export_path = st.session_state.get("export_annotated_path")
                    if export_path and Path(export_path).exists():
                        st.download_button(
                            "Download annotated video",
                            data=Path(export_path).read_bytes(),
                            file_name="agility_annotated.mp4",
                            mime="video/mp4",
                        )
            else:
                st.caption("Run analysis to enable annotated video export.")
            if speed_profile is not None and not speed_profile.empty:
                st.download_button(
                    "Download speed profile CSV",
                    data=speed_profile.to_csv(index=False),
                    file_name="agility_speed_profile.csv",
                    mime="text/csv",
                )
            if split_times is not None and not split_times.empty:
                st.download_button(
                    "Download split times CSV",
                    data=split_times.to_csv(index=False),
                    file_name="agility_split_times.csv",
                    mime="text/csv",
                )
            if definition.name == "Counter Movement Jump (CMJ)":
                force_time = result.matrices.get("force_time")
                if force_time is not None and not force_time.empty:
                    st.download_button(
                        "Download force-time CSV",
                        data=force_time.to_csv(index=False),
                        file_name="cmj_force_time.csv",
                        mime="text/csv",
                    )
                jump_height = result.matrices.get("jump_height")
                if jump_height is not None and not jump_height.empty:
                    st.download_button(
                        "Download jump height CSV",
                        data=jump_height.to_csv(index=False),
                        file_name="cmj_jump_height.csv",
                        mime="text/csv",
                    )
                landing_stability = result.matrices.get("landing_stability")
                if landing_stability is not None and not landing_stability.empty:
                    st.download_button(
                        "Download landing stability CSV",
                        data=landing_stability.to_csv(index=False),
                        file_name="cmj_landing_stability.csv",
                        mime="text/csv",
                    )
            if definition.name == "Drop Jump":
                ground_contact = result.matrices.get("ground_contact")
                if ground_contact is not None and not ground_contact.empty:
                    st.download_button(
                        "Download ground contact CSV",
                        data=ground_contact.to_csv(index=False),
                        file_name="drop_jump_ground_contact.csv",
                        mime="text/csv",
                    )
                reactive_strength = result.matrices.get("reactive_strength")
                if reactive_strength is not None and not reactive_strength.empty:
                    st.download_button(
                        "Download reactive strength CSV",
                        data=reactive_strength.to_csv(index=False),
                        file_name="drop_jump_reactive_strength.csv",
                        mime="text/csv",
                    )
                landing_force = result.matrices.get("landing_force")
                if landing_force is not None and not landing_force.empty:
                    st.download_button(
                        "Download landing force CSV",
                        data=landing_force.to_csv(index=False),
                        file_name="drop_jump_landing_force.csv",
                        mime="text/csv",
                    )
        with tabs[6]:
            prereq_error = gemini_prereq_error()
            if prereq_error:
                st.info(prereq_error)

            report_json = json.dumps(report_payload, indent=2)
            if st.session_state.get("chat_video_path") != video_path:
                st.session_state["chat_video_path"] = video_path
                st.session_state["chat_messages"] = []

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
                value=st.session_state.get(
                    "gemini_max_output_tokens", DEFAULT_GEMINI_MAX_OUTPUT_TOKENS
                ),
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
                            response_text, error = generate_gemini_response(
                                report_json=report_json,
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
    else:
        st.info("Run analysis to generate placeholder results.")
