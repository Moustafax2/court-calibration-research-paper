"""Canonical basketball court geometry and mask generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import cv2
import numpy as np
from src.camera.sports import get_sport_spec


@dataclass(frozen=True)
class CourtDimensions:
    """Court dimensions in canonical court units."""

    length: float = 28.702
    width: float = 15.2908


@dataclass(frozen=True)
class CourtRegion:
    """Named polygon region in canonical court coordinates."""

    class_id: int
    name: str
    polygon_xy: np.ndarray  # shape: (N, 2), float32


def build_default_four_region_model(
    dims: CourtDimensions | None = None,
) -> List[CourtRegion]:
    """
    Build a 4-region partition of the full court using quadrants.

    Canonical coordinate frame (center-origin):
    - x axis: along court length [-length/2, +length/2]
    - y axis: along court width  [-width/2, +width/2]
    - origin: center of court
    """
    dims = dims or CourtDimensions()
    half_x = dims.length / 2.0
    half_y = dims.width / 2.0

    polygons: Dict[str, np.ndarray] = {
        "left_upper": np.array(
            [[-half_x, 0.0], [0.0, 0.0], [0.0, half_y], [-half_x, half_y]]
        ),
        "left_lower": np.array(
            [[-half_x, -half_y], [0.0, -half_y], [0.0, 0.0], [-half_x, 0.0]]
        ),
        "right_upper": np.array(
            [[0.0, 0.0], [half_x, 0.0], [half_x, half_y], [0.0, half_y]]
        ),
        "right_lower": np.array(
            [[0.0, -half_y], [half_x, -half_y], [half_x, 0.0], [0.0, 0.0]]
        ),
    }

    regions: List[CourtRegion] = []
    for idx, (name, poly) in enumerate(polygons.items(), start=1):
        regions.append(
            CourtRegion(class_id=idx, name=name, polygon_xy=poly.astype(np.float32))
        )
    return regions


def build_basketball_paper_region_model() -> List[CourtRegion]:
    """
    Build a paper-aligned 4-region basketball layout in center-origin coordinates.

    Region ids (category-based, mirrored by side):
    1 = half-court area (both sides)
    2 = three-point area (both sides)
    3 = key/paint area (both sides)

    Draw order:
    - half first
    - three-point second (overrides half)
    - key third (overrides three-point/half)
    """
    half_x = 14.351
    half_y = 7.6454
    key_x = 8.5852
    key_half_y = 1.8034
    corner_y = 6.575425
    hoop_x_left = -12.776
    hoop_x_right = 12.776
    arc_r = 6.75
    arc_n = 48

    # Arc endpoints where the corner-three line meets the arc.
    # x = cx +/- sqrt(r^2 - y^2), using the side that points to center court.
    arc_dx = float(np.sqrt(max(1e-8, arc_r**2 - corner_y**2)))
    left_arc_x = hoop_x_left + arc_dx
    right_arc_x = hoop_x_right - arc_dx

    y_vals = np.linspace(corner_y, -corner_y, arc_n, dtype=np.float32)
    left_arc_x_vals = hoop_x_left + np.sqrt(np.maximum(0.0, arc_r**2 - y_vals**2))
    right_arc_x_vals = hoop_x_right - np.sqrt(np.maximum(0.0, arc_r**2 - y_vals**2))

    left_three_poly = np.vstack(
        [
            np.array([[-half_x, half_y], [-half_x, corner_y], [left_arc_x, corner_y]], dtype=np.float32),
            np.stack([left_arc_x_vals, y_vals], axis=1).astype(np.float32),
            np.array([[-half_x, -corner_y], [-half_x, -half_y]], dtype=np.float32),
        ]
    )
    right_three_poly = np.vstack(
        [
            np.array([[half_x, half_y], [half_x, corner_y], [right_arc_x, corner_y]], dtype=np.float32),
            np.stack([right_arc_x_vals, y_vals], axis=1).astype(np.float32),
            np.array([[half_x, -corner_y], [half_x, -half_y]], dtype=np.float32),
        ]
    )

    regions: List[CourtRegion] = [
        CourtRegion(
            class_id=1,
            name="left_half",
            polygon_xy=np.array(
                [[-half_x, -half_y], [0.0, -half_y], [0.0, half_y], [-half_x, half_y]],
                dtype=np.float32,
            ),
        ),
        CourtRegion(
            class_id=1,
            name="right_half",
            polygon_xy=np.array(
                [[0.0, -half_y], [half_x, -half_y], [half_x, half_y], [0.0, half_y]],
                dtype=np.float32,
            ),
        ),
        CourtRegion(class_id=2, name="left_three_pt", polygon_xy=left_three_poly.astype(np.float32)),
        CourtRegion(class_id=2, name="right_three_pt", polygon_xy=right_three_poly.astype(np.float32)),
        CourtRegion(
            class_id=3,
            name="left_key",
            polygon_xy=np.array(
                [[-half_x, -key_half_y], [-key_x, -key_half_y], [-key_x, key_half_y], [-half_x, key_half_y]],
                dtype=np.float32,
            ),
        ),
        CourtRegion(
            class_id=3,
            name="right_key",
            polygon_xy=np.array(
                [[key_x, -key_half_y], [half_x, -key_half_y], [half_x, key_half_y], [key_x, key_half_y]],
                dtype=np.float32,
            ),
        ),
    ]
    return regions


def build_four_region_model_for_sport(sport: str) -> List[CourtRegion]:
    """Build 4-region model based on a sport's canonical court dimensions."""
    spec = get_sport_spec(sport)
    if spec.name == "basketball":
        return build_basketball_paper_region_model()

    dims = CourtDimensions(length=spec.court_size[0], width=spec.court_size[1])
    return build_default_four_region_model(dims=dims)


def generate_semantic_mask(
    image_height: int,
    image_width: int,
    homography_image_from_court: np.ndarray,
    regions: Iterable[CourtRegion],
) -> np.ndarray:
    """
    Render region labels into image space from court polygons.

    Returns:
    - Integer mask with shape (H, W)
    - 0 = background, 1..N = region class ids
    """
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    h_mat = np.asarray(homography_image_from_court, dtype=np.float64)

    for region in regions:
        poly = region.polygon_xy.reshape(-1, 1, 2).astype(np.float32)
        warped = cv2.perspectiveTransform(poly, h_mat).reshape(-1, 2)
        warped_i32 = np.round(warped).astype(np.int32)
        cv2.fillPoly(mask, [warped_i32], color=int(region.class_id))

    return mask
