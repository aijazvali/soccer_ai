from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from .analyzers import (
    agility,
    ball_throw,
    cmj,
    dribbling,
    drop_jump,
    endurance,
    juggling,
    linear_sprint,
    mobility,
    passing,
    side_bridge,
    sprint,
    tapping,
)
from .analyzers.base import AnalysisResult


@dataclass(frozen=True)
class TestDefinition:
    name: str
    description: str
    expected_matrices: List[str]
    analyzer: Callable[[str, Dict], AnalysisResult]


TEST_DEFINITIONS: List[TestDefinition] = [
    TestDefinition(
        name="Agility",
        description="Rapid change of direction, balance, and control.",
        expected_matrices=["speed_profile", "split_times"],
        analyzer=agility.analyze,
    ),
    TestDefinition(
        name="Ball Throw",
        description="Upper-body power with release mechanics.",
        expected_matrices=["release_velocity", "trajectory", "shoulder_angle"],
        analyzer=ball_throw.analyze,
    ),
    TestDefinition(
        name="Counter Movement Jump (CMJ)",
        description="Explosive jump with stretch-shortening cycle.",
        expected_matrices=["force_time", "jump_height", "landing_stability"],
        analyzer=cmj.analyze,
    ),
    TestDefinition(
        name="Dribbling",
        description="Ball control under movement and direction changes.",
        expected_matrices=["speed_profile", "touch_log", "touch_rate"],
        analyzer=dribbling.analyze,
    ),
    TestDefinition(
        name="Drop Jump",
        description="Reactive strength and ground contact response.",
        expected_matrices=["ground_contact", "reactive_strength", "landing_force"],
        analyzer=drop_jump.analyze,
    ),
    TestDefinition(
        name="Endurance",
        description="Sustained output and fatigue response over time.",
        expected_matrices=[
            "pace_profile",
            "heart_rate_estimate",
            "fatigue_index",
            "turn_profile",
            "turn_events",
        ],
        analyzer=endurance.analyze,
    ),
    TestDefinition(
        name="Juggling",
        description="Repeated ball contacts with stability and control.",
        expected_matrices=["touch_count", "control_stability", "ball_height"],
        analyzer=juggling.analyze,
    ),
    TestDefinition(
        name="Linear Sprint",
        description="Straight-line acceleration and top speed.",
        expected_matrices=["split_times", "velocity_profile", "stride_length"],
        analyzer=linear_sprint.analyze,
    ),
    TestDefinition(
        name="Mobility",
        description="Joint range, symmetry, and movement quality.",
        expected_matrices=["range_of_motion", "joint_angles", "symmetry_index"],
        analyzer=mobility.analyze,
    ),
    TestDefinition(
        name="Passing",
        description="Ball strike accuracy and velocity.",
        expected_matrices=["ball_speed", "accuracy_map", "foot_contact"],
        analyzer=passing.analyze,
    ),
    TestDefinition(
        name="Side Bridge",
        description="Core endurance and hip alignment stability.",
        expected_matrices=["hold_duration", "hip_alignment", "stability_index"],
        analyzer=side_bridge.analyze,
    ),
    TestDefinition(
        name="Sprint",
        description="Short burst acceleration and stride mechanics.",
        expected_matrices=["acceleration_phase", "top_speed", "stride_frequency"],
        analyzer=sprint.analyze,
    ),
    TestDefinition(
        name="Tapping",
        description="Rhythm, cadence, and contact consistency.",
        expected_matrices=["contact_frequency", "rhythm_consistency", "impact_profile"],
        analyzer=tapping.analyze,
    ),
]

TEST_REGISTRY: Dict[str, TestDefinition] = {test.name: test for test in TEST_DEFINITIONS}


def get_test_names() -> List[str]:
    return [test.name for test in TEST_DEFINITIONS]


def get_test_definition(test_name: str) -> TestDefinition:
    if test_name not in TEST_REGISTRY:
        raise KeyError(f"Unknown test: {test_name}")
    return TEST_REGISTRY[test_name]


def run_analysis(test_name: str, video_path: str, extra_settings: Dict | None = None) -> AnalysisResult:
    definition = get_test_definition(test_name)
    settings: Dict = {
        "test_name": definition.name,
        "expected_matrices": definition.expected_matrices,
        "description": definition.description,
    }
    if extra_settings:
        settings.update(extra_settings)
    return definition.analyzer(video_path, settings)
