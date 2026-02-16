"""Canonical basketball court geometry and mask generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import cv2
import numpy as np
from src.camera.sports import get_sport_spec


@dataclass(frozen=True)
class CourtDimensions:
    """Court dimensions in feet."""

    length_ft: float = 94.0
    width_ft: float = 50.0


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

    Canonical coordinate frame:
    - x axis: along court length [0, length_ft]
    - y axis: along court width  [0, width_ft]
    - origin: top-left corner in top-view layout
    """
    dims = dims or CourtDimensions()
    mid_x = dims.length_ft / 2.0
    mid_y = dims.width_ft / 2.0

    polygons: Dict[str, np.ndarray] = {
        "near_left": np.array([[0, 0], [mid_x, 0], [mid_x, mid_y], [0, mid_y]]),
        "near_right": np.array(
            [[0, mid_y], [mid_x, mid_y], [mid_x, dims.width_ft], [0, dims.width_ft]]
        ),
        "far_left": np.array(
            [[mid_x, 0], [dims.length_ft, 0], [dims.length_ft, mid_y], [mid_x, mid_y]]
        ),
        "far_right": np.array(
            [
                [mid_x, mid_y],
                [dims.length_ft, mid_y],
                [dims.length_ft, dims.width_ft],
                [mid_x, dims.width_ft],
            ]
        ),
    }

    regions: List[CourtRegion] = []
    for idx, (name, poly) in enumerate(polygons.items(), start=1):
        regions.append(
            CourtRegion(class_id=idx, name=name, polygon_xy=poly.astype(np.float32))
        )
    return regions


def build_four_region_model_for_sport(sport: str) -> List[CourtRegion]:
    """Build 4-region model based on a sport's canonical court dimensions."""
    spec = get_sport_spec(sport)
    dims = CourtDimensions(length_ft=spec.court_size_ft[0], width_ft=spec.court_size_ft[1])
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
