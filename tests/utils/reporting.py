from __future__ import annotations

from typing import Any, Dict, List, Optional


def downsample_records(records: List[dict], max_items: int) -> List[dict]:
    if max_items <= 0:
        return []
    if len(records) <= max_items:
        return list(records)
    step = max(1, len(records) // max_items)
    sampled = records[::step]
    if sampled and sampled[-1] is not records[-1]:
        sampled[-1] = records[-1]
    return sampled[:max_items]


def build_report_payload(
    result: Any, runtime_store: Dict[str, Any], test_description: Optional[str] = None
) -> Dict[str, Any]:
    matrices_payload = {
        name: frame.to_dict(orient="records") for name, frame in result.matrices.items()
    }
    speed_profile = matrices_payload.get("speed_profile", [])
    if isinstance(speed_profile, list) and len(speed_profile) > 2000:
        matrices_payload["speed_profile"] = downsample_records(speed_profile, 2000)

    return {
        "test": result.test_name,
        "test_name": result.test_name,
        "test_description": test_description,
        "status": result.status,
        "metrics": result.metrics,
        "matrices": matrices_payload,
        "snapshots": runtime_store.get("snapshots", []),
        "shot_log": runtime_store.get("shot_log", []),
    }
