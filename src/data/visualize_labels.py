"""Visualization utilities for semantic label masks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


# BGR colors
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (90, 90, 90),      # background (gray)
    1: (70, 200, 70),     # half-court
    2: (255, 170, 30),    # three-point area
    3: (60, 80, 255),     # key
}

CLASS_NAMES: Dict[int, str] = {
    0: "background",
    1: "half_court",
    2: "three_pt",
    3: "key",
}


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS.items():
        out[mask == cls_id] = color
    return out


def _draw_legend(image: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    x0, y0 = 20, 20
    box_h = 24
    box_w = 26
    pad_y = 8

    cv2.rectangle(canvas, (x0 - 10, y0 - 10), (x0 + 280, y0 + 4 * (box_h + pad_y) + 8), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        "Semantic Regions",
        (x0, y0 - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    yy = y0 + 12
    for cls_id in [0, 1, 2, 3]:
        color = CLASS_COLORS[cls_id]
        name = CLASS_NAMES[cls_id]
        cv2.rectangle(canvas, (x0, yy), (x0 + box_w, yy + box_h), color, -1)
        cv2.rectangle(canvas, (x0, yy), (x0 + box_w, yy + box_h), (255, 255, 255), 1)
        cv2.putText(
            canvas,
            f"{cls_id}: {name}",
            (x0 + box_w + 10, yy + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        yy += box_h + pad_y
    return canvas


def visualize_label_overlay(
    frame_path: Path,
    mask_path: Path,
    output_path: Path,
    alpha: float = 0.45,
    draw_center_line: bool = True,
) -> dict:
    """Overlay semantic mask on frame and save visualized image."""
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Could not read frame: {frame_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    if mask.ndim == 3:
        mask = mask[..., 0]

    h, w = frame.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    color_mask = _colorize_mask(mask)
    blended = cv2.addWeighted(frame, 1.0 - alpha, color_mask, alpha, 0.0)

    if draw_center_line:
        cv2.line(blended, (w // 2, 0), (w // 2, h - 1), (255, 255, 255), 1, cv2.LINE_AA)

    blended = _draw_legend(blended)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), blended)
    if not ok:
        raise RuntimeError(f"Failed to write visualization image: {output_path}")

    unique_vals = sorted(int(v) for v in np.unique(mask))
    return {
        "frame_path": str(frame_path),
        "mask_path": str(mask_path),
        "output_path": str(output_path),
        "unique_mask_values": unique_vals,
    }
