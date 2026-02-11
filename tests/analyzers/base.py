from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class AnalysisResult:
    test_name: str
    status: str
    metrics: Dict[str, float | int | str]
    matrices: Dict[str, pd.DataFrame]
    artifacts: Dict[str, str]
    logs: List[str]


def _deterministic_seed(test_name: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(test_name))


def _build_dummy_metrics(test_name: str, seed: int) -> Dict[str, float | int | str]:
    score = 70 + (seed % 25)
    duration = round(4.0 + (seed % 9) * 0.5, 2)
    effort = round(6.5 + (seed % 7) * 0.3, 2)
    return {
        "test": test_name,
        "score": score,
        "duration_s": duration,
        "effort_index": effort,
        "quality": "Placeholder",
    }


def _build_dummy_matrices(
    expected_names: List[str],
    seed: int,
    rows: int = 60,
) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    time_s = np.linspace(0, 3, rows)
    matrices: Dict[str, pd.DataFrame] = {}

    for index, name in enumerate(expected_names):
        phase = (index + 1) * 0.35
        velocity = np.sin(time_s * (1.5 + index * 0.1) + phase) * (1 + 0.1 * index)
        accel = np.cos(time_s * (1.2 + index * 0.1) + phase) * 0.6
        angle = 30 + 10 * np.sin(time_s * 0.8 + phase)
        noise = rng.normal(0, 0.02, size=rows)

        frame = pd.DataFrame(
            {
                "time_s": np.round(time_s, 3),
                "velocity": np.round(velocity + noise, 3),
                "accel": np.round(accel + noise, 3),
                "angle_deg": np.round(angle + noise * 10, 2),
            }
        )
        matrices[name] = frame

    return matrices


def build_dummy_result(
    test_name: str,
    expected_matrices: List[str],
    settings: Dict | None = None,
) -> AnalysisResult:
    seed = _deterministic_seed(test_name)
    metrics = _build_dummy_metrics(test_name, seed)
    matrices = _build_dummy_matrices(expected_matrices, seed)

    logs = [
        "Loaded video",
        "Extracted frames (stub)",
        "Computed matrices (stub)",
        "Generated placeholder metrics",
    ]

    return AnalysisResult(
        test_name=test_name,
        status="ok",
        metrics=metrics,
        matrices=matrices,
        artifacts={},
        logs=logs,
    )
