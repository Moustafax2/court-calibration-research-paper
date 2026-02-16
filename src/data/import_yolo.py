"""Import YOLO keypoint labels and convert to homography manifest rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Tuple

import cv2
import numpy as np

from src.data.reference_points import reference_points_for_sport


def _parse_side_mapping(text: str | None) -> dict[int, str]:
    if not text:
        return {}
    out: dict[int, str] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        k, v = item.split(":")
        out[int(k)] = v.strip()
    return out


def _iter_image_files(images_dir: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _find_label_file(labels_dir: Path, image_path: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def _parse_yolo_pose_line(line: str) -> dict[str, Any]:
    vals = [float(v) for v in line.strip().split()]
    if len(vals) < 5:
        raise ValueError("YOLO line too short.")
    class_id = int(vals[0])
    bbox = vals[1:5]
    kp_raw = vals[5:]
    if len(kp_raw) % 3 != 0:
        raise ValueError("Expected keypoints as x y vis triples.")
    kps = []
    for i in range(0, len(kp_raw), 3):
        x, y, vis = kp_raw[i : i + 3]
        kps.append((x, y, vis))
    return {"class_id": class_id, "bbox": bbox, "keypoints": kps}


def _compute_homography_from_keypoints(
    image_w: int,
    image_h: int,
    yolo_keypoints: list[tuple[float, float, float]],
    court_points: list[Tuple[float, float]],
    visibility_threshold: float,
) -> tuple[np.ndarray | None, list[dict[str, Any]]]:
    correspondences: list[dict[str, Any]] = []
    for idx, (nx, ny, vis) in enumerate(yolo_keypoints):
        if idx >= len(court_points):
            break
        if vis < visibility_threshold:
            continue
        ix = float(nx) * image_w
        iy = float(ny) * image_h
        cx, cy = court_points[idx]
        correspondences.append(
            {
                "idx": idx,
                "court_xy": [float(cx), float(cy)],
                "image_xy": [float(ix), float(iy)],
                "visibility": float(vis),
            }
        )

    if len(correspondences) < 4:
        return None, correspondences

    court_arr = np.asarray([c["court_xy"] for c in correspondences], dtype=np.float64)
    image_arr = np.asarray([c["image_xy"] for c in correspondences], dtype=np.float64)
    h, _ = cv2.findHomography(court_arr, image_arr, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return h, correspondences


def import_yolo_keypoints_to_manifest(
    images_dir: Path,
    labels_dir: Path,
    manifest_out: Path,
    sport: str,
    split: str,
    side: Literal["left", "right", "full"] | None = None,
    side_from_class: str | None = None,
    class_id: int | None = None,
    visibility_threshold: float = 0.5,
    project_root: Path = Path("."),
    append: bool = True,
) -> Dict[str, Any]:
    images_dir = Path(images_dir).resolve()
    labels_dir = Path(labels_dir).resolve()
    manifest_out = Path(manifest_out).resolve()
    project_root = Path(project_root).resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels_dir not found: {labels_dir}")
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    mapping = _parse_side_mapping(side_from_class)

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    imported = 0
    skipped = 0

    for image_path in _iter_image_files(images_dir):
        label_path = _find_label_file(labels_dir, image_path)
        if not label_path.exists():
            skipped += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            errors.append(f"{image_path.name}: failed to read image")
            continue
        h_img, w_img = image.shape[:2]

        try:
            lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not lines:
                skipped += 1
                continue
            parsed = [_parse_yolo_pose_line(ln) for ln in lines]
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label_path.name}: parse error: {exc}")
            continue

        selected = None
        for obj in parsed:
            if class_id is not None and obj["class_id"] != class_id:
                continue
            selected = obj
            break
        if selected is None:
            skipped += 1
            continue

        side_row = side or mapping.get(int(selected["class_id"]), "left")
        refs = reference_points_for_sport(sport=sport, side=side_row)
        court_pts = [rp.court_xy for rp in refs]
        h_mat, correspondences = _compute_homography_from_keypoints(
            image_w=w_img,
            image_h=h_img,
            yolo_keypoints=selected["keypoints"],
            court_points=court_pts,
            visibility_threshold=visibility_threshold,
        )
        if h_mat is None:
            skipped += 1
            continue

        try:
            frame_rel = image_path.relative_to(project_root).as_posix()
        except ValueError:
            frame_rel = image_path.as_posix()

        row = {
            "frame_path": frame_rel,
            "homography": h_mat.tolist(),
            "split": split,
            "side": side_row,
            "video_id": image_path.parent.name,
            "frame_index": None,
            "num_points": len(correspondences),
            "source": "yolo_keypoints",
            "class_id": int(selected["class_id"]),
            "correspondences": correspondences,
        }
        rows.append(row)
        imported += 1

    mode = "a" if append else "w"
    with manifest_out.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "manifest_out": str(manifest_out),
        "imported": imported,
        "skipped": skipped,
        "num_errors": len(errors),
        "first_errors": errors[:20],
    }
