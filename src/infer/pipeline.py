"""Frame processor abstractions for video inference."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from src.camera.sports import SportSpec, get_sport_spec


@dataclass
class FrameProcessorResult:
    frame_out: np.ndarray
    metadata: Dict[str, str]


class FrameProcessor:
    """Interface for per-frame calibration/inference processing."""

    def process(self, frame: np.ndarray, frame_idx: int) -> FrameProcessorResult:
        raise NotImplementedError


class NoOpFrameProcessor(FrameProcessor):
    """Pass-through processor for smoke testing."""

    def process(self, frame: np.ndarray, frame_idx: int) -> FrameProcessorResult:
        return FrameProcessorResult(
            frame_out=frame,
            metadata={"mode": "noop", "frame_idx": str(frame_idx)},
        )


class SegmentationFrameProcessor(FrameProcessor):
    """Segmentation inference processor using a trained U-Net checkpoint."""

    CLASS_COLORS = {
        0: (90, 90, 90),      # background
        1: (70, 200, 70),     # half-court
        2: (255, 170, 30),    # three-pt
        3: (60, 80, 255),     # key
    }
    CLASS_NAMES = {
        0: "background",
        1: "half_court",
        2: "three_pt",
        3: "key",
    }

    def __init__(
        self,
        ckpt_path: Path,
        overlay_alpha: float = 0.45,
        device: str | None = None,
        retrieval_ckpt_path: Path | None = None,
        templates_dir: Path | None = None,
        stn_ckpt_path: Path | None = None,
        template_homographies_path: Path | None = None,
        sport: str = "basketball",
    ) -> None:
        import torch
        from src.camera.pose_dictionary import pose_vector_to_homography
        from src.models.stn_homography import STNHomographyRegressor
        from src.models.unet_segmentation import UNetSegmentation
        from src.models.siamese_pose import SiameseRetrievalModel

        self._torch = torch
        self._pose_vector_to_h = pose_vector_to_homography
        self.overlay_alpha = float(np.clip(overlay_alpha, 0.0, 1.0))
        self.sport = sport

        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        cfg = checkpoint.get("config", {})
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})

        self.input_h = int(data_cfg.get("image_height", 512))
        self.input_w = int(data_cfg.get("image_width", 512))
        self.num_classes = int(model_cfg.get("num_classes", 4))

        self.model = UNetSegmentation(
            in_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=self.num_classes,
            base_channels=int(model_cfg.get("base_channels", 32)),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # Optional retrieval branch.
        self.retrieval_enabled = False
        self.retrieval_model = None
        self.template_ids: list[int] = []
        self.template_embeddings = None
        self.template_embed_dim = None
        self.templates_dir = templates_dir
        if retrieval_ckpt_path is not None and templates_dir is not None:
            retrieval_ckpt_path = Path(retrieval_ckpt_path)
            templates_dir = Path(templates_dir)
            if retrieval_ckpt_path.exists() and templates_dir.exists():
                r_ckpt = torch.load(str(retrieval_ckpt_path), map_location="cpu")
                r_cfg = r_ckpt.get("config", {})
                r_model_cfg = r_cfg.get("model", {})
                self.retrieval_model = SiameseRetrievalModel(
                    in_channels=int(r_model_cfg.get("in_channels", 1)),
                    embedding_dim=int(r_model_cfg.get("embedding_dim", 128)),
                    base_channels=int(r_model_cfg.get("base_channels", 32)),
                ).to(self.device)
                self.retrieval_model.load_state_dict(r_ckpt["model_state_dict"])
                self.retrieval_model.eval()

                self.template_ids = sorted(
                    int(p.stem.split("_")[-1]) for p in templates_dir.glob("template_*.png")
                )
                if self.template_ids:
                    self.template_embeddings = self._precompute_template_embeddings()
                    self.template_embed_dim = int(self.template_embeddings.shape[1])
                    self.retrieval_enabled = True

        # Optional STN refinement branch.
        self.stn_enabled = False
        self.stn_model = None
        self.stn_input_h = self.input_h
        self.stn_input_w = self.input_w
        self.template_homographies = None
        if (
            stn_ckpt_path is not None
            and template_homographies_path is not None
            and self.retrieval_enabled
        ):
            stn_ckpt_path = Path(stn_ckpt_path)
            template_homographies_path = Path(template_homographies_path)
            if stn_ckpt_path.exists() and template_homographies_path.exists():
                s_ckpt = torch.load(str(stn_ckpt_path), map_location="cpu")
                s_cfg = s_ckpt.get("config", {})
                s_model_cfg = s_cfg.get("model", {})
                s_data_cfg = s_cfg.get("data", {})
                self.stn_input_h = int(s_data_cfg.get("image_height", self.input_h))
                self.stn_input_w = int(s_data_cfg.get("image_width", self.input_w))
                self.stn_model = STNHomographyRegressor(
                    in_channels=int(s_model_cfg.get("in_channels", 2)),
                    base_channels=int(s_model_cfg.get("base_channels", 32)),
                ).to(self.device)
                self.stn_model.load_state_dict(s_ckpt["model_state_dict"])
                self.stn_model.eval()

                with template_homographies_path.open("r", encoding="utf-8") as f:
                    mats = json.load(f)
                self.template_homographies = [np.asarray(m, dtype=np.float64) for m in mats]
                self.stn_enabled = True

    def _predict_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        torch = self._torch
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        x_t = torch.from_numpy(x).float().to(self.device)

        with torch.no_grad():
            logits = self.model(x_t)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        mask = cv2.resize(pred, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask

    def _normalize_mask_for_retrieval(self, mask: np.ndarray) -> np.ndarray:
        m = mask.astype(np.float32)
        denom = max(1.0, float(np.max(m)))
        return m / denom

    def _embed_mask(self, mask: np.ndarray) -> np.ndarray:
        torch = self._torch
        if self.retrieval_model is None:
            raise RuntimeError("Retrieval model is not initialized.")
        h, w = self.input_h, self.input_w
        resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        x = self._normalize_mask_for_retrieval(resized)[None, None, ...]  # NCHW
        x_t = torch.from_numpy(x).float().to(self.device)
        with torch.no_grad():
            emb = self.retrieval_model.encoder(x_t).cpu().numpy()
        return emb[0]

    def _precompute_template_embeddings(self) -> np.ndarray:
        embs = []
        for tid in self.template_ids:
            p = Path(self.templates_dir) / f"template_{tid:04d}.png"
            mask = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            if mask.ndim == 3:
                mask = mask[..., 0]
            embs.append(self._embed_mask(mask))
        if not embs:
            raise RuntimeError("No valid template embeddings could be computed.")
        return np.asarray(embs, dtype=np.float32)

    def _retrieve_template(self, mask: np.ndarray) -> tuple[int, float]:
        q = self._embed_mask(mask).astype(np.float32)[None, :]
        diffs = self.template_embeddings - q
        dists = np.linalg.norm(diffs, axis=1)
        idx = int(np.argmin(dists))
        return int(self.template_ids[idx]), float(dists[idx])

    def _predict_relative_h(self, mask_pred: np.ndarray, template_id: int) -> np.ndarray:
        torch = self._torch
        if self.stn_model is None:
            raise RuntimeError("STN model is not initialized.")
        tp = Path(self.templates_dir) / f"template_{template_id:04d}.png"
        tmask = cv2.imread(str(tp), cv2.IMREAD_UNCHANGED)
        if tmask is None:
            raise FileNotFoundError(f"Template mask missing: {tp}")
        if tmask.ndim == 3:
            tmask = tmask[..., 0]

        a = cv2.resize(mask_pred, (self.stn_input_w, self.stn_input_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        b = cv2.resize(tmask, (self.stn_input_w, self.stn_input_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        a /= max(1.0, float(np.max(a)))
        b /= max(1.0, float(np.max(b)))
        x = np.stack([a, b], axis=0)[None, ...]  # NCHW
        x_t = torch.from_numpy(x).float().to(self.device)
        with torch.no_grad():
            rel_vec = self.stn_model(x_t).squeeze(0).cpu().numpy()
        return self._pose_vector_to_h(rel_vec)

    def _draw_court_outline(self, image: np.ndarray, h_final: np.ndarray) -> None:
        # Center-origin basketball dimensions.
        half_x, half_y = 14.351, 7.6454
        outline = np.array(
            [[-half_x, half_y], [half_x, half_y], [half_x, -half_y], [-half_x, -half_y]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(outline, h_final).reshape(-1, 2)
        pts = np.round(warped).astype(np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=(255, 255, 0), thickness=2)

    def _colorize(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape[:2]
        out = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in self.CLASS_COLORS.items():
            out[mask == cls_id] = color
        return out

    def _draw_legend(self, image: np.ndarray) -> np.ndarray:
        canvas = image.copy()
        x0, y0 = 20, 20
        box_h = 22
        box_w = 24
        pad_y = 7
        cv2.rectangle(canvas, (x0 - 10, y0 - 10), (x0 + 280, y0 + 4 * (box_h + pad_y) + 8), (20, 20, 20), -1)
        cv2.putText(canvas, "Segmentation", (x0, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        yy = y0 + 10
        for cls_id in [0, 1, 2, 3]:
            color = self.CLASS_COLORS.get(cls_id, (255, 255, 255))
            name = self.CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            cv2.rectangle(canvas, (x0, yy), (x0 + box_w, yy + box_h), color, -1)
            cv2.rectangle(canvas, (x0, yy), (x0 + box_w, yy + box_h), (255, 255, 255), 1)
            cv2.putText(canvas, f"{cls_id}: {name}", (x0 + box_w + 10, yy + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            yy += box_h + pad_y
        return canvas

    def process(self, frame: np.ndarray, frame_idx: int) -> FrameProcessorResult:
        mask = self._predict_mask(frame)
        color_mask = self._colorize(mask)
        out = cv2.addWeighted(frame, 1.0 - self.overlay_alpha, color_mask, self.overlay_alpha, 0.0)
        out = self._draw_legend(out)
        cv2.putText(
            out,
            f"frame={frame_idx}",
            (20, out.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        metadata = {"mode": "segmentation", "classes": ",".join(str(int(v)) for v in np.unique(mask))}

        if self.retrieval_enabled:
            template_id, dist = self._retrieve_template(mask)
            cv2.putText(
                out,
                f"template={template_id} dist={dist:.3f}",
                (20, out.shape[0] - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            metadata["mode"] = "segmentation+retrieval"
            metadata["template_id"] = str(template_id)
            metadata["template_dist"] = f"{dist:.6f}"

            if self.stn_enabled and self.template_homographies is not None:
                if 0 <= template_id < len(self.template_homographies):
                    h_template = self.template_homographies[template_id]
                    h_rel = self._predict_relative_h(mask, template_id)
                    h_final = h_rel @ h_template
                    self._draw_court_outline(out, h_final)
                    metadata["mode"] = "segmentation+retrieval+stn"
        return FrameProcessorResult(frame_out=out, metadata=metadata)


def build_frame_processor(
    sport: str,
    ckpt: Path | None,
    overlay_alpha: float = 0.45,
    device: str | None = None,
    retrieval_ckpt: Path | None = None,
    templates_dir: Path | None = None,
    stn_ckpt: Path | None = None,
    template_homographies_path: Path | None = None,
) -> Tuple[SportSpec, FrameProcessor]:
    """Build a sport-specific frame processor."""
    spec = get_sport_spec(sport)
    if ckpt is None:
        return spec, NoOpFrameProcessor()
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return spec, SegmentationFrameProcessor(
        ckpt_path=Path(ckpt),
        overlay_alpha=overlay_alpha,
        device=device,
        retrieval_ckpt_path=retrieval_ckpt,
        templates_dir=templates_dir,
        stn_ckpt_path=stn_ckpt,
        template_homographies_path=template_homographies_path,
        sport=sport,
    )
