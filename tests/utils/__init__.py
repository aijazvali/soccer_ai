from .ui import inject_base_styles, render_header, card, render_pills
from .video_io import save_uploaded_file, probe_video
from .calibration_ui import (
    ensure_canvas_compat,
    read_video_frame,
    prepare_canvas_frame,
    canvas_initial_drawing,
    extract_canvas_points,
)
from .player_overlay import (
    format_speed,
    format_accel,
    draw_stats_overlay,
    draw_event_overlay,
    draw_player_overlays,
    draw_ball_overlay,
    draw_ball_trail_overlay,
    collect_ball_trail,
)
from .formatting import (
    format_time,
    format_distance_m,
    format_speed_mps,
    format_accel_mps2,
)
from .chat import (
    CHAT_HISTORY_LIMIT,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMINI_TEMPERATURE,
    DEFAULT_GEMINI_MAX_OUTPUT_TOKENS,
    COACH_SYSTEM_PROMPT,
    get_gemini_api_key,
    gemini_prereq_error,
    build_chat_prompt,
    generate_gemini_response,
)
from .reporting import build_report_payload, downsample_records
from .video_export import export_annotated_video

__all__ = [
    "inject_base_styles",
    "render_header",
    "card",
    "render_pills",
    "save_uploaded_file",
    "probe_video",
    "ensure_canvas_compat",
    "read_video_frame",
    "prepare_canvas_frame",
    "canvas_initial_drawing",
    "extract_canvas_points",
    "format_speed",
    "format_accel",
    "draw_stats_overlay",
    "draw_event_overlay",
    "draw_player_overlays",
    "draw_ball_overlay",
    "draw_ball_trail_overlay",
    "collect_ball_trail",
    "format_time",
    "format_distance_m",
    "format_speed_mps",
    "format_accel_mps2",
    "CHAT_HISTORY_LIMIT",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_GEMINI_TEMPERATURE",
    "DEFAULT_GEMINI_MAX_OUTPUT_TOKENS",
    "COACH_SYSTEM_PROMPT",
    "get_gemini_api_key",
    "gemini_prereq_error",
    "build_chat_prompt",
    "generate_gemini_response",
    "build_report_payload",
    "downsample_records",
    "export_annotated_video",
]
