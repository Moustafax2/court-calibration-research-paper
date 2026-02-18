"""Calibration evaluation utilities."""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import median
from typing import Any, Dict

import cv2
import numpy as np

from src.data.annotation_schema import FrameAnnotation
from src.infer.pipeline import build_frame_processor
from src.utils.config import load_yaml_config


def _resolve(root: Path, p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp.resolve()
    return (root / pp).resolve()


def _load_manifest(manifest_path: Path, split: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            raw = json.loads(t)
            ann = FrameAnnotation.from_dict(raw)
            if split is not None and ann.split != split:
                continue
            rows.append(raw)
    return rows


def _load_assignments(path: Path | None, split: str | None) -> dict[str, int]:
    if path is None or not path.exists():
        return {}
    out: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            row = json.loads(t)
            if split is not None and row.get("split") != split:
                continue
            out[str(row["frame_path"])] = int(row["template_id"])
    return out


def _make_court_mask(length: float, width: float, ppm: float = 30.0, pad: int = 40):
    half_x = length / 2.0
    half_y = width / 2.0
    w = int(round(length * ppm + 2 * pad))
    h = int(round(width * ppm + 2 * pad))
    mask = np.zeros((h, w), dtype=np.uint8)
    x0 = int(round(pad))
    y0 = int(round(pad))
    x1 = int(round(pad + length * ppm))
    y1 = int(round(pad + width * ppm))
    cv2.rectangle(mask, (x0, y0), (x1, y1), color=1, thickness=-1)
    # court -> canvas
    a = np.array(
        [
            [ppm, 0.0, pad + half_x * ppm],
            [0.0, -ppm, pad + half_y * ppm],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return mask, a


def _iou_binary(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    inter = float(np.logical_and(aa, bb).sum())
    union = float(np.logical_or(aa, bb).sum())
    if union <= 0.0:
        return 0.0
    return inter / union


def _iou_entire(h_gt: np.ndarray, h_pred: np.ndarray, court_mask: np.ndarray, a_c2p: np.ndarray) -> float:
    try:
        t_court = np.linalg.inv(h_pred) @ h_gt
    except np.linalg.LinAlgError:
        return 0.0
    t_pix = a_c2p @ t_court @ np.linalg.inv(a_c2p)
    warped = cv2.warpPerspective(
        court_mask.astype(np.uint8),
        t_pix.astype(np.float64),
        (court_mask.shape[1], court_mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return _iou_binary(court_mask, warped)


def _visible_mask_in_court(
    h_img_from_court: np.ndarray,
    image_w: int,
    image_h: int,
    court_mask: np.ndarray,
    a_c2p: np.ndarray,
) -> np.ndarray:
    corners = np.array(
        [[0.0, 0.0], [image_w - 1.0, 0.0], [image_w - 1.0, image_h - 1.0], [0.0, image_h - 1.0]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    try:
        h_court_from_img = np.linalg.inv(h_img_from_court)
    except np.linalg.LinAlgError:
        return np.zeros_like(court_mask, dtype=np.uint8)

    court_pts = cv2.perspectiveTransform(corners, h_court_from_img.astype(np.float64)).reshape(-1, 2)
    homog = np.concatenate([court_pts, np.ones((court_pts.shape[0], 1), dtype=np.float64)], axis=1)
    pix = (a_c2p @ homog.T).T
    pix = pix[:, :2] / np.clip(pix[:, 2:3], 1e-9, None)
    poly = np.round(pix).astype(np.int32)
    out = np.zeros_like(court_mask, dtype=np.uint8)
    cv2.fillPoly(out, [poly], color=1)
    return (out > 0).astype(np.uint8) * (court_mask > 0).astype(np.uint8)


def _iou_part(
    h_gt: np.ndarray,
    h_pred: np.ndarray,
    image_w: int,
    image_h: int,
    court_mask: np.ndarray,
    a_c2p: np.ndarray,
) -> float:
    gt_vis = _visible_mask_in_court(h_gt, image_w=image_w, image_h=image_h, court_mask=court_mask, a_c2p=a_c2p)
    pred_vis = _visible_mask_in_court(h_pred, image_w=image_w, image_h=image_h, court_mask=court_mask, a_c2p=a_c2p)
    return _iou_binary(gt_vis, pred_vis)


def evaluate_calibration(
    config_path: Path,
    ckpt: Path,
    retrieval_ckpt: Path | None = None,
    templates_dir: Path | None = None,
    stn_ckpt: Path | None = None,
    template_homographies: Path | None = None,
    sport: str = "basketball",
    device: str | None = None,
    retrieval_method: str = "embedding",
    max_samples: int | None = None,
) -> Dict[str, Any]:
    cfg = load_yaml_config(config_path)
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("eval", {})
    root = Path(data_cfg.get("project_root", ".")).resolve()
    manifest = _resolve(root, str(data_cfg["manifest"]))
    split = str(data_cfg.get("split", "val"))
    assignments_path = data_cfg.get("assignments")
    assignments = _load_assignments(
        _resolve(root, str(assignments_path)) if assignments_path is not None else None,
        split=split,
    )

    rows = _load_manifest(manifest, split=split)
    if max_samples is not None:
        rows = rows[: max(1, int(max_samples))]
    if not rows:
        raise ValueError(f"No rows found in manifest={manifest} for split={split}.")

    spec, processor = build_frame_processor(
        sport=sport,
        ckpt=Path(ckpt),
        retrieval_ckpt=retrieval_ckpt,
        templates_dir=templates_dir,
        stn_ckpt=stn_ckpt,
        template_homographies_path=template_homographies,
        debug_retrieval=False,
        retrieval_method=retrieval_method,
        overlay_alpha=0.0,
        device=device,
    )
    court_mask, a_c2p = _make_court_mask(length=spec.court_size[0], width=spec.court_size[1], ppm=30.0, pad=40)

    warmup = int(eval_cfg.get("runtime_warmup", 5))
    runtimes_ms: list[float] = []
    iou_entire_list: list[float] = []
    iou_part_list: list[float] = []
    template_hits = 0
    template_total = 0
    pred_count = 0
    failed_frames = 0

    for i, row in enumerate(rows):
        ann = FrameAnnotation.from_dict(row)
        frame_abs = _resolve(root, ann.frame_path.as_posix())
        frame = cv2.imread(str(frame_abs), cv2.IMREAD_COLOR)
        if frame is None:
            failed_frames += 1
            continue

        t0 = time.perf_counter()
        result = processor.process(frame=frame, frame_idx=i)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            runtimes_ms.append(dt_ms)

        meta = result.metadata
        pred_h_text = meta.get("homography_pred")
        if pred_h_text is None:
            failed_frames += 1
            continue

        try:
            h_pred = np.asarray(json.loads(pred_h_text), dtype=np.float64).reshape(3, 3)
        except Exception:
            failed_frames += 1
            continue
        if not np.isfinite(h_pred).all():
            failed_frames += 1
            continue

        h_gt = ann.homography_image_from_court
        iou_entire_list.append(_iou_entire(h_gt=h_gt, h_pred=h_pred, court_mask=court_mask, a_c2p=a_c2p))
        iou_part_list.append(
            _iou_part(
                h_gt=h_gt,
                h_pred=h_pred,
                image_w=frame.shape[1],
                image_h=frame.shape[0],
                court_mask=court_mask,
                a_c2p=a_c2p,
            )
        )
        pred_count += 1

        if assignments:
            gt_tid = assignments.get(ann.frame_path.as_posix())
            pred_tid_text = meta.get("template_id")
            if gt_tid is not None and pred_tid_text is not None:
                template_total += 1
                if int(pred_tid_text) == int(gt_tid):
                    template_hits += 1

    def _mean(xs: list[float]) -> float | None:
        if not xs:
            return None
        return float(np.mean(np.asarray(xs, dtype=np.float64)))

    return {
        "sport": sport,
        "manifest": str(manifest),
        "split": split,
        "num_rows": len(rows),
        "num_predicted": pred_count,
        "num_failed": failed_frames,
        "iou_entire_mean": _mean(iou_entire_list),
        "iou_entire_median": float(median(iou_entire_list)) if iou_entire_list else None,
        "iou_part_mean": _mean(iou_part_list),
        "iou_part_median": float(median(iou_part_list)) if iou_part_list else None,
        "retrieval_top1": (float(template_hits) / float(template_total)) if template_total > 0 else None,
        "retrieval_count": template_total,
        "runtime_ms_mean": _mean(runtimes_ms),
        "runtime_ms_median": float(median(runtimes_ms)) if runtimes_ms else None,
        "runtime_warmup": warmup,
        "processor": processor.__class__.__name__,
    }
