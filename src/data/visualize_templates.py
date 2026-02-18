"""Visualization utilities for pose template masks."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (35, 35, 35),
    1: (70, 200, 70),
    2: (255, 170, 30),
    3: (60, 80, 255),
}


def _colorize(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS.items():
        out[mask == cls_id] = color
    return out


def visualize_templates_grid(
    templates_dir: Path,
    output_path: Path,
    max_templates: int | None = 100,
    cols: int = 10,
    tile_width: int = 220,
    tile_height: int = 124,
) -> dict:
    templates_dir = Path(templates_dir).resolve()
    output_path = Path(output_path).resolve()
    files = sorted(templates_dir.glob("template_*.png"))
    if not files:
        raise FileNotFoundError(f"No template_*.png found in {templates_dir}")

    if max_templates is not None:
        m = int(max_templates)
        if m <= 0:
            raise ValueError("max_templates must be > 0 when provided.")
        files = files[:m]

    cols = max(1, int(cols))
    rows = int(ceil(len(files) / cols))
    pad = 8
    label_h = 24
    canvas_w = cols * (tile_width + pad) + pad
    canvas_h = rows * (tile_height + label_h + pad) + pad
    canvas = np.full((canvas_h, canvas_w, 3), 20, dtype=np.uint8)

    for i, p in enumerate(files):
        r = i // cols
        c = i % cols
        x0 = pad + c * (tile_width + pad)
        y0 = pad + r * (tile_height + label_h + pad)

        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if m.ndim == 3:
            m = m[..., 0]
        tile = _colorize(m.astype(np.uint8))
        tile = cv2.resize(tile, (tile_width, tile_height), interpolation=cv2.INTER_NEAREST)
        canvas[y0 : y0 + tile_height, x0 : x0 + tile_width] = tile
        cv2.rectangle(canvas, (x0, y0), (x0 + tile_width, y0 + tile_height), (200, 200, 200), 1)
        cv2.putText(
            canvas,
            p.stem,
            (x0 + 4, y0 + tile_height + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), canvas)
    if not ok:
        raise RuntimeError(f"Failed to write template visualization: {output_path}")

    return {
        "templates_dir": str(templates_dir),
        "output_path": str(output_path),
        "num_templates_shown": len(files),
        "grid_cols": cols,
        "tile_size": [tile_width, tile_height],
    }

