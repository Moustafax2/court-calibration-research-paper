"""Pose dictionary generation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

from src.camera.court_model import build_four_region_model_for_sport, generate_semantic_mask
from src.data.annotation_schema import FrameAnnotation


@dataclass
class PoseDictionaryArtifacts:
    means: np.ndarray
    covariances: np.ndarray
    weights: np.ndarray
    assignments: np.ndarray
    homographies: np.ndarray


def homography_to_pose_vector(h: np.ndarray) -> np.ndarray:
    """Convert a 3x3 homography to an 8D normalized vector."""
    hh = np.asarray(h, dtype=np.float64).copy()
    if abs(hh[2, 2]) < 1e-12:
        hh /= (np.linalg.norm(hh) + 1e-12)
    else:
        hh /= hh[2, 2]
    vec = np.array(
        [
            hh[0, 0],
            hh[0, 1],
            hh[0, 2],
            hh[1, 0],
            hh[1, 1],
            hh[1, 2],
            hh[2, 0],
            hh[2, 1],
        ],
        dtype=np.float64,
    )
    return vec


def pose_vector_to_homography(vec: np.ndarray) -> np.ndarray:
    """Convert an 8D pose vector back to a 3x3 homography."""
    v = np.asarray(vec, dtype=np.float64).reshape(8)
    h = np.array(
        [
            [v[0], v[1], v[2]],
            [v[3], v[4], v[5]],
            [v[6], v[7], 1.0],
        ],
        dtype=np.float64,
    )
    return h


def _load_manifest_rows(
    manifest_path: Path,
    split: str | None,
    allowed_frame_paths: set[str] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            raw = json.loads(text)
            ann = FrameAnnotation.from_dict(raw)
            if split is not None and ann.split != split:
                continue
            frame_key = ann.frame_path.as_posix()
            if allowed_frame_paths is not None and frame_key not in allowed_frame_paths:
                continue
            rows.append(raw)
    if not rows:
        raise ValueError(f"No usable rows found in {manifest_path} for split={split}")
    return rows


def _sample_rows(
    rows: List[Dict[str, Any]],
    max_samples: int | None,
    random_state: int,
) -> List[Dict[str, Any]]:
    if max_samples is None:
        return rows
    k = int(max_samples)
    if k <= 0:
        raise ValueError("max_samples must be > 0 when provided.")
    if k >= len(rows):
        return rows
    rng = np.random.default_rng(int(random_state))
    idx = rng.choice(len(rows), size=k, replace=False)
    idx = np.sort(idx)
    return [rows[int(i)] for i in idx]


def _coverage_from_posterior(prob: np.ndarray, threshold: float) -> float:
    if prob.size == 0:
        return 0.0
    maxp = np.max(prob, axis=1)
    return float(np.mean(maxp >= threshold))


def fit_pose_gmm_auto(
    vectors: np.ndarray,
    min_components: int = 50,
    max_components: int = 260,
    step: int = 10,
    posterior_threshold: float = 0.6,
    random_state: int = 42,
) -> GaussianMixture:
    """Fit a GMM with increasing component count until coverage criterion is met."""
    n = vectors.shape[0]
    low = max(1, min(min_components, n))
    high = max(1, min(max_components, n))
    if low > high:
        low = high

    best = None
    for k in range(low, high + 1, max(1, step)):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=random_state,
            max_iter=300,
        )
        gmm.fit(vectors)
        post = gmm.predict_proba(vectors)
        cov = _coverage_from_posterior(post, posterior_threshold)
        best = gmm
        if cov >= posterior_threshold:
            return gmm
    if best is None:
        raise RuntimeError("Failed to fit GMM pose dictionary.")
    return best


def generate_pose_dictionary(
    manifest_path: Path,
    sport: str,
    output_dir: Path,
    project_root: Path = Path("."),
    split: str = "train",
    template_size: Tuple[int, int] = (960, 540),  # (W, H)
    min_components: int = 50,
    max_components: int = 260,
    step: int = 10,
    posterior_threshold: float = 0.6,
    random_state: int = 42,
    max_samples: int | None = None,
    allowed_frame_paths: set[str] | None = None,
    template_source: str = "medoid",
    min_template_fg_ratio: float = 0.01,
) -> Dict[str, Any]:
    """Generate pose dictionary artifacts from annotated homographies."""
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    project_root = Path(project_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = output_dir / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_manifest_rows(
        manifest_path=manifest_path,
        split=split,
        allowed_frame_paths=allowed_frame_paths,
    )
    rows = _sample_rows(rows=rows, max_samples=max_samples, random_state=random_state)
    vectors = []
    homographies = []
    frame_paths = []
    for row in rows:
        h = np.asarray(row["homography"], dtype=np.float64)
        vectors.append(homography_to_pose_vector(h))
        homographies.append(h)
        frame_paths.append(row["frame_path"])
    vecs = np.asarray(vectors, dtype=np.float64)

    gmm = fit_pose_gmm_auto(
        vectors=vecs,
        min_components=min_components,
        max_components=max_components,
        step=step,
        posterior_threshold=posterior_threshold,
        random_state=random_state,
    )
    assignments = gmm.predict(vecs)

    # Save dictionary arrays.
    np.savez(
        output_dir / "pose_dictionary.npz",
        means=gmm.means_,
        covariances=gmm.covariances_,
        weights=gmm.weights_,
    )

    # Write frame-to-template assignments.
    assignments_path = output_dir / "template_assignments.jsonl"
    with assignments_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            out = {
                "frame_path": row["frame_path"],
                "split": row["split"],
                "template_id": int(assignments[i]),
            }
            f.write(json.dumps(out) + "\n")

    # Generate one semantic template per component.
    # `medoid` uses the nearest real sample to each component mean, which avoids
    # unrealistic templates from averaging homographies directly in vector space.
    regions = build_four_region_model_for_sport(sport)
    width, height = template_size
    written_templates = 0
    near_empty_templates = 0
    low_fg_components_count = 0
    template_homographies: list[list[list[float]]] = []

    def _render_mask_from_h(h_in: np.ndarray, frame_path: str | None = None) -> tuple[np.ndarray, float]:
        h_render = np.asarray(h_in, dtype=np.float64)
        if frame_path is not None:
            fp = Path(frame_path)
            if not fp.is_absolute():
                fp = (project_root / fp).resolve()
            img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if img is not None:
                src_h, src_w = img.shape[:2]
                if src_w > 0 and src_h > 0:
                    sx = float(width) / float(src_w)
                    sy = float(height) / float(src_h)
                    s = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
                    h_render = s @ h_render
        mask_local = generate_semantic_mask(
            image_height=height,
            image_width=width,
            homography_image_from_court=h_render,
            regions=regions,
        )
        fg = float(np.mean(mask_local > 0))
        return mask_local, fg

    for k in range(gmm.n_components):
        mask: np.ndarray | None = None
        fg_ratio = 0.0
        h_k: np.ndarray
        if template_source == "mean":
            h_init = pose_vector_to_homography(gmm.means_[k])
            h_k = np.asarray(h_init, dtype=np.float64)
            mask, fg_ratio = _render_mask_from_h(h_k, frame_path=None)
        else:
            idxs = np.where(assignments == k)[0]
            if idxs.size == 0:
                h_init = pose_vector_to_homography(gmm.means_[k])
                h_k = np.asarray(h_init, dtype=np.float64)
                mask, fg_ratio = _render_mask_from_h(h_k, frame_path=None)
            else:
                mu = gmm.means_[k].reshape(1, -1)
                cluster_vecs = vecs[idxs]
                d = np.linalg.norm(cluster_vecs - mu, axis=1)
                order = np.argsort(d)
                best_mask = None
                best_h = None
                best_fg = -1.0
                for jj in order:
                    rep_idx = int(idxs[int(jj)])
                    cand_h = homographies[rep_idx]
                    cand_frame = str(frame_paths[rep_idx])
                    cand_mask, cand_fg = _render_mask_from_h(cand_h, frame_path=cand_frame)
                    if cand_fg > best_fg:
                        best_fg = cand_fg
                        best_mask = cand_mask
                        best_h = np.asarray(cand_h, dtype=np.float64)
                    if cand_fg >= float(min_template_fg_ratio):
                        break
                assert best_mask is not None and best_h is not None
                mask = best_mask
                h_k = best_h
                fg_ratio = float(best_fg)
        if fg_ratio < float(min_template_fg_ratio):
            low_fg_components_count += 1

        assert mask is not None
        if fg_ratio < 0.005:
            near_empty_templates += 1
        out_path = templates_dir / f"template_{k:04d}.png"
        ok = cv2.imwrite(str(out_path), mask)
        if ok:
            written_templates += 1
        template_homographies.append(h_k.tolist())

    with (output_dir / "template_homographies.json").open("w", encoding="utf-8") as f:
        json.dump(template_homographies, f)

    summary = {
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "split_used": split,
        "num_samples": int(len(rows)),
        "max_samples": int(max_samples) if max_samples is not None else None,
        "allowed_frame_paths_count": (
            int(len(allowed_frame_paths)) if allowed_frame_paths is not None else None
        ),
        "template_source": str(template_source),
        "min_template_fg_ratio": float(min_template_fg_ratio),
        "num_components": int(gmm.n_components),
        "templates_written": int(written_templates),
        "near_empty_templates": int(near_empty_templates),
        "low_fg_components_after_search": int(low_fg_components_count),
        "assignments_path": str(assignments_path),
        "dictionary_npz": str((output_dir / "pose_dictionary.npz")),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
