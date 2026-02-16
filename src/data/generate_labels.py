"""Generate semantic region masks from annotation manifest + homographies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import cv2

from src.camera.court_model import build_four_region_model_for_sport, generate_semantic_mask
from src.data.annotation_schema import FrameAnnotation


def _resolve(project_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def generate_labels_from_manifest(
    manifest_path: Path,
    output_dir: Path,
    sport: str,
    project_root: Path = Path("."),
    overwrite: bool = False,
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
    errors: list[str] = []
    index_rows: list[dict[str, Any]] = []

    with manifest.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                raw = json.loads(text)
                ann = FrameAnnotation.from_dict(raw)
                frame_abs = _resolve(project_root, ann.frame_path)
                image = cv2.imread(str(frame_abs), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"failed to load frame: {frame_abs}")
                h, w = image.shape[:2]

                rel_stub = ann.frame_path.as_posix().replace("/", "__")
                mask_name = f"{line_num:06d}__{rel_stub}.png"
                mask_path = out_dir / mask_name
                if mask_path.exists() and not overwrite:
                    skipped += 1
                else:
                    mask = generate_semantic_mask(
                        image_height=h,
                        image_width=w,
                        homography_image_from_court=ann.homography_image_from_court,
                        regions=regions,
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
        "num_errors": len(errors),
        "first_errors": errors[:20],
    }
