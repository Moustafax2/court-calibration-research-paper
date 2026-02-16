"""Annotation schema and validation for court calibration datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

VALID_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class FrameAnnotation:
    """Single frame annotation entry loaded from JSONL."""

    frame_path: Path
    homography_image_from_court: np.ndarray  # shape: (3, 3), float64
    split: str
    video_id: str | None = None
    frame_index: int | None = None

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "FrameAnnotation":
        frame_path_raw = data.get("frame_path")
        if not isinstance(frame_path_raw, str) or not frame_path_raw:
            raise ValueError("frame_path must be a non-empty string.")

        split = data.get("split")
        if not isinstance(split, str) or split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}.")

        h_raw = data.get("homography")
        h = np.asarray(h_raw, dtype=np.float64)
        if h.shape != (3, 3):
            raise ValueError("homography must be a 3x3 numeric matrix.")
        if not np.isfinite(h).all():
            raise ValueError("homography contains non-finite values.")
        if abs(np.linalg.det(h)) < 1e-12:
            raise ValueError("homography is singular (non-invertible).")

        video_id = data.get("video_id")
        if video_id is not None and not isinstance(video_id, str):
            raise ValueError("video_id must be a string when provided.")

        frame_index = data.get("frame_index")
        if frame_index is not None and not isinstance(frame_index, int):
            raise ValueError("frame_index must be an integer when provided.")

        return FrameAnnotation(
            frame_path=Path(frame_path_raw),
            homography_image_from_court=h,
            split=split,
            video_id=video_id,
            frame_index=frame_index,
        )
