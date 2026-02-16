"""Generate semantic region masks from annotation manifest + homographies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from src.camera.court_model import build_four_region_model_for_sport, generate_semantic_mask
from src.data.annotation_schema import FrameAnnotation


def _resolve(project_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _regions_for_side(regions, side: str | None):
    if side is None or side == "full":
        return list(regions)
    if side == "left":
        return [r for r in regions if r.name.startswith("left_")]
    if side == "right":
        return [r for r in regions if r.name.startswith("right_")]
    return list(regions)


def _reprojection_rmse(h: np.ndarray, correspondences: list[dict[str, Any]] | None) -> float | None:
    if not correspondences:
        return None
    if len(correspondences) < 4:
        return None

    court_pts = []
    image_pts = []
    for c in correspondences:
        try:
            cx, cy = c["court_xy"]
            ix, iy = c["image_xy"]
            court_pts.append([float(cx), float(cy)])
            image_pts.append([float(ix), float(iy)])
        except Exception:  # noqa: BLE001
            return None

    court_arr = np.asarray(court_pts, dtype=np.float64).reshape(-1, 1, 2)
    image_arr = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    projected = cv2.perspectiveTransform(court_arr, h).reshape(-1, 2)
    err = projected - image_arr
    rmse = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
    return rmse


def generate_labels_from_manifest(
    manifest_path: Path,
    output_dir: Path,
    sport: str,
    project_root: Path = Path("."),
    overwrite: bool = False,
    max_rmse: float | None = None,
) -> Dict[str, Any]:
    """Generate one PNG mask per manifest row and write an index JSONL."""
    manifest = Path(manifest_path).resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    regions = build_four_region_model_for_sport(sport)
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "labels_index.jsonl"

    generated = 0
    skipped = 0
    skipped_by_rmse = 0
    errors: list[str] = []
    index_rows: list[dict[str, Any]] = []
    rmses: list[float] = []

    with manifest.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                raw = json.loads(text)
                ann = FrameAnnotation.from_dict(raw)
                side = raw.get("side", ann.side)
                correspondences = raw.get("correspondences")
                frame_abs = _resolve(project_root, ann.frame_path)
                image = cv2.imread(str(frame_abs), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"failed to load frame: {frame_abs}")
                h, w = image.shape[:2]
                h_mat = ann.homography_image_from_court

                rmse = _reprojection_rmse(h_mat, correspondences)
                if rmse is not None:
                    rmses.append(rmse)
                    if max_rmse is not None and rmse > float(max_rmse):
                        skipped_by_rmse += 1
                        continue

                rel_stub = ann.frame_path.as_posix().replace("/", "__")
                mask_name = f"{line_num:06d}__{rel_stub}.png"
                mask_path = out_dir / mask_name
                if mask_path.exists() and not overwrite:
                    skipped += 1
                else:
                    regions_for_row = _regions_for_side(regions, side)
                    mask = generate_semantic_mask(
                        image_height=h,
                        image_width=w,
                        homography_image_from_court=h_mat,
                        regions=regions_for_row,
                    )
                    ok = cv2.imwrite(str(mask_path), mask)
                    if not ok:
                        raise RuntimeError(f"failed to write mask: {mask_path}")
                    generated += 1

                index_rows.append(
                    {
                        "frame_path": ann.frame_path.as_posix(),
                        "mask_path": mask_path.as_posix(),
                        "split": ann.split,
                        "sport": sport,
                        "side": side,
                        "rmse": rmse,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"line {line_num}: {exc}")

    with index_path.open("w", encoding="utf-8") as out_f:
        for row in index_rows:
            out_f.write(json.dumps(row) + "\n")

    return {
        "manifest_path": str(manifest),
        "output_dir": str(out_dir),
        "index_path": str(index_path),
        "generated": generated,
        "skipped": skipped,
        "skipped_by_rmse": skipped_by_rmse,
        "rmse_mean": float(np.mean(rmses)) if rmses else None,
        "rmse_max": float(np.max(rmses)) if rmses else None,
        "num_errors": len(errors),
        "first_errors": errors[:20],
    }
