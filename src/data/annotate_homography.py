"""Interactive homography annotation for single frames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal, Tuple

import cv2
import numpy as np

from src.camera.sports import get_sport_spec
from src.data.reference_points import ReferencePoint, reference_points_for_sport


def _draw_overlay(
    image: np.ndarray,
    selected: list[tuple[ReferencePoint, tuple[int, int]]],
    current: ReferencePoint | None,
    help_text: str,
) -> np.ndarray:
    canvas = image.copy()
    for idx, (rp, (x, y)) in enumerate(selected, start=1):
        cv2.circle(canvas, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(
            canvas,
            f"{idx}:{rp.name}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    if current is not None:
        cv2.putText(
            canvas,
            f"Click point for: {current.name} court={current.court_xy}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        help_text,
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _compute_homography(
    selected: list[tuple[ReferencePoint, tuple[int, int]]],
) -> np.ndarray:
    if len(selected) < 4:
        raise ValueError("Need at least 4 clicked correspondences to compute homography.")

    court_pts = np.array([rp.court_xy for rp, _ in selected], dtype=np.float64)
    image_pts = np.array([xy for _, xy in selected], dtype=np.float64)
    h, _inliers = cv2.findHomography(court_pts, image_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if h is None:
        raise RuntimeError("cv2.findHomography failed. Try more accurate point clicks.")
    return h


def annotate_single_frame(
    image_path: Path,
    manifest_path: Path,
    sport: str,
    split: str,
    side: Literal["left", "right", "full"] = "left",
    video_id: str | None = None,
    frame_index: int | None = None,
    project_root: Path = Path("."),
    preview_path: Path | None = None,
) -> dict:
    spec = get_sport_spec(sport)  # validates sport
    refs = reference_points_for_sport(sport, side=side)

    image_abs = image_path if image_path.is_absolute() else (project_root / image_path).resolve()
    image = cv2.imread(str(image_abs), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not open image: {image_abs}")

    selected: list[tuple[ReferencePoint, tuple[int, int]]] = []
    current_idx = 0
    clicked: list[tuple[int, int]] = []

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((int(x), int(y)))

    window_name = "annotate-homography"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    help_text = "L-click: set point | n: skip | u: undo | c: compute/save | q: quit"

    while True:
        current = refs[current_idx] if current_idx < len(refs) else None

        while clicked:
            xy = clicked.pop(0)
            if current is not None:
                selected.append((current, xy))
                current_idx += 1
                current = refs[current_idx] if current_idx < len(refs) else None

        frame = _draw_overlay(image=image, selected=selected, current=current, help_text=help_text)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            cv2.destroyAllWindows()
            raise RuntimeError("Annotation cancelled by user.")
        if key == ord("u") and selected:
            selected.pop()
            current_idx = max(0, current_idx - 1)
        if key == ord("n") and current_idx < len(refs):
            current_idx += 1
        if key == ord("c"):
            break
        if current_idx >= len(refs):
            # No more prompts; allow compute/save with 'c' or continue undo.
            pass

    cv2.destroyAllWindows()

    h = _compute_homography(selected)

    manifest_abs = manifest_path if manifest_path.is_absolute() else (project_root / manifest_path).resolve()
    manifest_abs.parent.mkdir(parents=True, exist_ok=True)

    try:
        frame_rel = image_abs.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        frame_rel = image_path.as_posix()
    row = {
        "frame_path": frame_rel,
        "homography": h.tolist(),
        "split": split,
        "side": side,
        "video_id": video_id,
        "frame_index": frame_index,
        "num_points": len(selected),
        "correspondences": [
            {
                "name": rp.name,
                "court_xy": [float(rp.court_xy[0]), float(rp.court_xy[1])],
                "image_xy": [int(xy[0]), int(xy[1])],
            }
            for rp, xy in selected
        ],
    }

    with manifest_abs.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    preview_written = None
    if preview_path is not None:
        preview_abs = preview_path if preview_path.is_absolute() else (project_root / preview_path).resolve()
        preview_abs.parent.mkdir(parents=True, exist_ok=True)
        preview = image.copy()
        court_w, court_h = spec.court_size
        half_x = court_w / 2.0
        half_y = court_h / 2.0
        court_outline = np.array(
            [[-half_x, half_y], [half_x, half_y], [half_x, -half_y], [-half_x, -half_y]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(court_outline, h).reshape(-1, 2).astype(np.int32)
        cv2.polylines(preview, [warped], isClosed=True, color=(255, 0, 0), thickness=2)
        ok = cv2.imwrite(str(preview_abs), preview)
        if ok:
            preview_written = str(preview_abs)

    return {
        "image_path": str(image_abs),
        "manifest_path": str(manifest_abs),
        "side": side,
        "split": split,
        "num_points": len(selected),
        "preview_path": preview_written,
    }
