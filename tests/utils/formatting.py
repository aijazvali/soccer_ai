from __future__ import annotations

from typing import Optional


def format_time(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes:d}m {rem:04.1f}s"


def format_distance_m(meters: Optional[float], use_metric: bool = True) -> str:
    if meters is None or meters < 0:
        return "N/A"
    if use_metric:
        return f"{meters:.2f} m"
    return f"{meters / 1000:.2f} km"


def format_speed_mps(speed_mps: Optional[float], use_metric: bool = True) -> str:
    if speed_mps is None:
        return "N/A"
    if use_metric:
        return f"{speed_mps:.2f} m/s"
    return f"{speed_mps * 3.6:.1f} km/h"


def format_accel_mps2(accel: Optional[float]) -> str:
    if accel is None:
        return "N/A"
    return f"{accel:.2f} m/s^2"
