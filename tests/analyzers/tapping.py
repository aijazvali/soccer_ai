from __future__ import annotations

from .base import AnalysisResult, build_dummy_result


def analyze(video_path: str, settings: dict) -> AnalysisResult:
    test_name = settings.get("test_name", "Unknown Test")
    expected_matrices = settings.get("expected_matrices", [])
    return build_dummy_result(test_name, expected_matrices, settings)
