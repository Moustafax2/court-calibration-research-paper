"""Shared canonical reference points for manual and imported annotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple


@dataclass(frozen=True)
class ReferencePoint:
    name: str
    court_xy: Tuple[float, float]


def basketball_left_reference_points() -> List[ReferencePoint]:
    return [
        ReferencePoint("a", (-8.5852, 1.8034)),
        ReferencePoint("b", (-14.351, 1.8034)),
        ReferencePoint("c", (-14.351, -1.8034)),
        ReferencePoint("d", (-8.5852, -1.8034)),
        ReferencePoint("u_baseline", (-14.351, 2.7686)),
        ReferencePoint("u_3pt", (-14.351, 6.575425)),
        ReferencePoint("u_corner", (-14.351, 7.6454)),
        ReferencePoint("coach", (-5.7658, 7.6454)),
        ReferencePoint("coach_2", (-2.7176, 7.6454)),
        ReferencePoint("l_baseline", (-14.351, -2.7686)),
        ReferencePoint("l_3pt", (-14.351, -6.575425)),
        ReferencePoint("l_corner", (-14.351, -7.6454)),
        ReferencePoint("hash_top_1", (-10.9474, 1.8034)),
        ReferencePoint("hash_top_2", (-9.9822, 1.8034)),
        ReferencePoint("hash_top_3", (-9.017, 1.8034)),
        ReferencePoint("hash_bottom_1", (-10.9474, -1.8034)),
        ReferencePoint("hash_bottom_2", (-9.9822, -1.8034)),
        ReferencePoint("hash_bottom_3", (-9.017, -1.8034)),
        ReferencePoint("upper_center", (0.0, 7.6454)),
        ReferencePoint("lower_center", (0.0, -7.6454)),
    ]


def basketball_right_reference_points() -> List[ReferencePoint]:
    return [
        ReferencePoint("a", (8.5852, 1.8034)),
        ReferencePoint("b", (14.351, 1.8034)),
        ReferencePoint("c", (14.351, -1.8034)),
        ReferencePoint("d", (8.5852, -1.8034)),
        ReferencePoint("u_baseline", (14.351, 2.7686)),
        ReferencePoint("u_3pt", (14.351, 6.575425)),
        ReferencePoint("u_corner", (14.351, 7.6454)),
        ReferencePoint("coach", (5.7658, 7.6454)),
        ReferencePoint("coach_2", (2.7176, 7.6454)),
        ReferencePoint("l_baseline", (14.351, -2.7686)),
        ReferencePoint("l_3pt", (14.351, -6.575425)),
        ReferencePoint("l_corner", (14.351, -7.6454)),
        ReferencePoint("hash_top_1", (10.9474, 1.8034)),
        ReferencePoint("hash_top_2", (9.9822, 1.8034)),
        ReferencePoint("hash_top_3", (9.017, 1.8034)),
        ReferencePoint("hash_bottom_1", (10.9474, -1.8034)),
        ReferencePoint("hash_bottom_2", (9.9822, -1.8034)),
        ReferencePoint("hash_bottom_3", (9.017, -1.8034)),
        ReferencePoint("upper_center", (0.0, 7.6454)),
        ReferencePoint("lower_center", (0.0, -7.6454)),
    ]


def basketball_full_reference_points() -> List[ReferencePoint]:
    half_x = 14.351
    half_y = 7.6454
    return [
        ReferencePoint("corner_tl", (-half_x, half_y)),
        ReferencePoint("corner_tr", (half_x, half_y)),
        ReferencePoint("corner_br", (half_x, -half_y)),
        ReferencePoint("corner_bl", (-half_x, -half_y)),
        ReferencePoint("upper_center", (0.0, half_y)),
        ReferencePoint("lower_center", (0.0, -half_y)),
    ]


def reference_points_for_sport(
    sport: str, side: Literal["left", "right", "full"]
) -> List[ReferencePoint]:
    key = sport.lower()
    if key != "basketball":
        raise ValueError(f"No reference-point template defined for sport: {sport}")
    if side == "left":
        return basketball_left_reference_points()
    if side == "right":
        return basketball_right_reference_points()
    return basketball_full_reference_points()
