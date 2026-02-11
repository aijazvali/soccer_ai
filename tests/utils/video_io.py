from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from uuid import uuid4


def save_uploaded_file(uploaded_file, base_dir: str = ".cache/uploads") -> str:
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    original_suffix = Path(uploaded_file.name).suffix
    suffix = original_suffix if original_suffix else ".mp4"
    filename = f"{uuid4().hex}{suffix}"
    file_path = base_path / filename

    with file_path.open("wb") as handle:
        handle.write(uploaded_file.getbuffer())

    return str(file_path)


def probe_video(video_path: str) -> Dict[str, Any] | None:
    try:
        import cv2  # type: ignore
    except Exception:
        return None

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return None

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()

    return {
        "fps": float(fps) if fps else None,
        "frame_count": int(frame_count) if frame_count else None,
    }
